#include "laser_mapping.hpp"


LaserMapping::LaserMapping(const LaserMappingConfig &config) : config_(config) {
    preprocess_.reset(new PointCloudPreprocess());
    p_imu_.reset(new ImuProcess());
    RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "laser mapping construct");
}

LaserMapping::~LaserMapping() {
    scan_down_body_ = nullptr;
    scan_undistort_ = nullptr;
    scan_down_world_ = nullptr;
    RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "laser mapping deconstruct");
}

bool LaserMapping::Init() {
    preprocess_->Blind() = config_.preprocess__blind;
    preprocess_->TimeScale() = config_.preprocess__time_scale;
    preprocess_->NumScans() = config_.preprocess__num_scans;
    preprocess_->PointFilterNum() = config_.preprocess__point_filter_num;
    preprocess_->FeatureEnabled() = config_.preprocess__feature_extract_enable;
    if (config_.preprocess__lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
    } else if (config_.preprocess__lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
    } else if (config_.preprocess__lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
    } else {
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"), "unknown lidar_type");
        return false;
    }

    ivox_options_.resolution_ = config_.ivox__grid_resolution;
    if (config_.ivox__nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (config_.ivox__nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (config_.ivox__nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (config_.ivox__nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"), "unknown ivox_nearby_type, use NEARBY18");
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    voxel_scan_.setLeafSize(config_.filter_size_surf_min, config_.filter_size_surf_min, config_.filter_size_surf_min);
    
    p_imu_->SetExtrinsic(VecFromArray<double>(config_.mapping__extrinsic_t), MatFromArray<double>(config_.mapping__extrinsic_r));
    p_imu_->SetGyrCov(V3D(config_.gyr_cov, config_.gyr_cov, config_.gyr_cov));
    p_imu_->SetAccCov(V3D(config_.acc_cov, config_.acc_cov, config_.acc_cov));
    p_imu_->SetGyrBiasCov(V3D(config_.b_gyr_cov, config_.b_gyr_cov, config_.b_gyr_cov));
    p_imu_->SetAccBiasCov(V3D(config_.b_acc_cov, config_.b_acc_cov, config_.b_acc_cov));

     // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        config_.num_max_iterations , epsi.data());

    return true;
}


void LaserMapping::Run() {
    if (!SyncPackages()) {
        return;
    }

    /// IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);
    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"), "No point, skip this scan!");
        return;
    }

    /// the first scan
    if (flg_first_scan_) {
        ivox_->AddPoints(scan_undistort_->points);
        first_lidar_time_ = measures_.lidar_bag_time_;
        flg_first_scan_ = false;
        return;
    }
    flg_EKF_inited_ = (measures_.lidar_bag_time_ - first_lidar_time_) >= INIT_TIME;

    /// downsample
    Timer::Evaluate(
        [&, this]() {
            voxel_scan_.setInputCloud(scan_undistort_);
            voxel_scan_.filter(*scan_down_body_);
        },
        "Downsample PointCloud");

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"), "Too few points %d / %ld, skip this scan!", cur_pts, scan_undistort_->size());
        return;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);
    residuals_.resize(cur_pts, 0);
    point_selected_surf_.resize(cur_pts, true);
    plane_coef_.resize(cur_pts, V4F::Zero());

    // ICP and iterated Kalman filter update
    Timer::Evaluate(
        [&, this]() {
            // iterated state estimation
            double solve_H_time = 0;
            // update the observation model, will call nn and point-to-plane residual computation
            kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            // save the state
            state_point_ = kf_.get_x();
            euler_cur_ = SO3ToEuler(state_point_.rot);
            pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
        },
        "IEKF Solve and Update");

    // update local map
    Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");

    RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "    [ mapping ]: In num: %ld downsamp %d Map grid num: %ld effect num : %d",
                scan_undistort_->points.size(), cur_pts, ivox_->NumValidGrids(), effect_feat_num_);

    // Debug variables
    frame_num_++;
}

void LaserMapping::StandardPCLCallBack(const sensor_msgs::msg::PointCloud2::UniquePtr &msg) {
    mtx_buffer_.lock();
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (rclcpp::Time(msg->header.stamp).seconds() < last_timestamp_lidar_) {
                RCLCPP_ERROR(rclcpp::get_logger("laser_mapping"), "lidar loop back, clear buffer");
                lidar_buffer_.clear();
            }

            PointCloudType::Ptr ptr(new PointCloudType());
            preprocess_->Process(msg, ptr);
            lidar_buffer_.push_back(ptr);
            time_buffer_.push_back(rclcpp::Time(msg->header.stamp).seconds());
            last_timestamp_lidar_ = rclcpp::Time(msg->header.stamp).seconds();
        },
        "Preprocess (Standard)");
    mtx_buffer_.unlock();
}

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg) {
    mtx_buffer_.lock();
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (rclcpp::Time(msg->header.stamp).seconds() < last_timestamp_lidar_) {
                RCLCPP_WARN(rclcpp::get_logger("laser_mapping"), "lidar loop back, clear buffer");
                lidar_buffer_.clear();
            }

            last_timestamp_lidar_ = rclcpp::Time(msg->header.stamp).seconds();

            if (!config_.time_sync_en && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 && !imu_buffer_.empty() &&
                !lidar_buffer_.empty()) {
                RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "IMU and LiDAR not Synced, IMU time: %f, lidar header time: %f",
                            last_timestamp_imu_, last_timestamp_lidar_);
            }

            if (config_.time_sync_en && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
                !imu_buffer_.empty()) {
                timediff_set_flg_ = true;
                timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
                RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "Self sync IMU and LiDAR, time diff is %f", timediff_lidar_wrt_imu_);
            }

            PointCloudType::Ptr ptr(new PointCloudType());
            preprocess_->Process(msg, ptr);
            lidar_buffer_.emplace_back(ptr);
            time_buffer_.emplace_back(last_timestamp_lidar_);
        },
        "Preprocess (Livox)");

    mtx_buffer_.unlock();
}

void LaserMapping::IMUCallBack(const sensor_msgs::msg::Imu::UniquePtr &msg_in) {
    publish_count_++;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu_) > 0.1 && config_.time_sync_en) {
        msg->header.stamp = rclcpp::Time((timediff_lidar_wrt_imu_ + rclcpp::Time(msg_in->header.stamp).seconds()) * 1e9);
    }

    double timestamp = rclcpp::Time(msg->header.stamp).seconds();

    mtx_buffer_.lock();
    if (timestamp < last_timestamp_imu_) {
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"), "imu loop back, clear buffer");
        imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.emplace_back(msg);
    mtx_buffer_.unlock();
}

bool LaserMapping::SyncPackages() {
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed_) {
        measures_.lidar_ = lidar_buffer_.front();
        measures_.lidar_bag_time_ = time_buffer_.front();

        if (measures_.lidar_->points.size() <= 1) {
            RCLCPP_WARN(rclcpp::get_logger("laser_mapping"), "Too few input point cloud!");
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
        } else if (measures_.lidar_->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_) {
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
        } else {
            scan_num_++;
            lidar_end_time_ = measures_.lidar_bag_time_ + measures_.lidar_->points.back().curvature / double(1000);
            lidar_mean_scantime_ +=
                (measures_.lidar_->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
        }

        measures_.lidar_end_time_ = lidar_end_time_;
        lidar_pushed_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    double imu_time = rclcpp::Time(imu_buffer_.front()->header.stamp).seconds();
    measures_.imu_.clear();
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
        imu_time = rclcpp::Time(imu_buffer_.front()->header.stamp).seconds();
        if (imu_time > lidar_end_time_) break;
        measures_.imu_.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;
    return true;
}

void LaserMapping::PrintState(const state_ikfom &s) {
    RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "state r: %f %f %f %f, t: %f %f %f, off r: %f %f %f %f, t: %f %f %f",
                s.rot.coeffs()[0], s.rot.coeffs()[1], s.rot.coeffs()[2], s.rot.coeffs()[3], s.pos[0], s.pos[1], s.pos[2],
                s.offset_R_L_I.coeffs()[0], s.offset_R_L_I.coeffs()[1], s.offset_R_L_I.coeffs()[2], s.offset_R_L_I.coeffs()[3],
                s.offset_T_L_I[0], s.offset_T_L_I[1], s.offset_T_L_I[2]);
}

void LaserMapping::MapIncremental() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));

        /* decide if need add to map */
        PointType &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty() && flg_EKF_inited_) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / config_.filter_size_map_min).array().floor() + 0.5) * config_.filter_size_map_min;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * config_.filter_size_map_min &&
                fabs(dis_2_center.y()) > 0.5 * config_.filter_size_map_min &&
                fabs(dis_2_center.z()) > 0.5 * config_.filter_size_map_min) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            bool need_add = true;
            float dist = calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
                    if (calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}

/**
 * Lidar point cloud registration
 * will be called by the eskf custom observation model
 * compute point-to-plane residual here
 * @param s kf state
 * @param ekfom_data H matrix
 */
void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    int cnt_pts = scan_down_body_->size();

    std::vector<size_t> index(cnt_pts);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    Timer::Evaluate(
        [&, this]() {
            auto R_wl = (s.rot * s.offset_R_L_I).cast<float>();
            auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

            /** closest surface search and residual computation **/
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_->points[i];
                PointType &point_world = scan_down_world_->points[i];

                /* transform to world frame */
                V3F p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                auto &points_near = nearest_points_[i];
                if (ekfom_data.converge) {
                    /** Find the closest surfaces in the map **/
                    ivox_->GetClosestPoint(point_world, points_near, NUM_MATCH_POINTS);
                    point_selected_surf_[i] = points_near.size() >= MIN_NUM_MATCH_POINTS;
                    if (point_selected_surf_[i]) {
                        point_selected_surf_[i] =
                            esti_plane(plane_coef_[i], points_near, config_.esti_plane_threshold);
                    }
                }

                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);

                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        point_selected_surf_[i] = true;
                        residuals_[i] = pd2;
                    }
                }
            });
        },
        "    ObsModel (Lidar Match)");

    effect_feat_num_ = 0;

    corr_pts_.resize(cnt_pts);
    corr_norm_.resize(cnt_pts);
    for (int i = 0; i < cnt_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
            corr_pts_[effect_feat_num_][3] = residuals_[i];

            effect_feat_num_++;
        }
    }
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        ekfom_data.valid = false;
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"), "No Effective Points!");
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            ekfom_data.h.resize(effect_feat_num_);

            index.resize(effect_feat_num_);
            const M3F off_R = s.offset_R_L_I.toRotationMatrix().cast<float>();
            const V3F off_t = s.offset_T_L_I.cast<float>();
            const M3F Rt = s.rot.toRotationMatrix().transpose().cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                V3F point_this_be = corr_pts_[i].head<3>();
                M3F point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
                V3F point_this = off_R * point_this_be + off_t;
                M3F point_crossmat = SKEW_SYM_MATRIX(point_this);

                /*** get the normal vector of closest surface/corner ***/
                V3F norm_vec = corr_norm_[i].head<3>();

                /*** calculate the Measurement Jacobian matrix H ***/
                V3F C(Rt * norm_vec);
                V3F A(point_crossmat * C);

                if (config_.mapping__extrinsic_est_en) {
                    V3F B(point_be_crossmat * off_R.transpose() * C);
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
                        B[1], B[2], C[0], C[1], C[2];
                } else {
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0;
                }

                /*** Measurement: distance to the closest surface/corner ***/
                ekfom_data.h(i) = -corr_pts_[i][3];
            });
        },
        "    ObsModel (IEKF Build Jacobian)");
}

///////////////////////////  private method /////////////////////////////////////////////////////////////////////
template <typename T>
void LaserMapping::SetPosestamp(T &out) {
    out.pose.position.x = state_point_.pos(0);
    out.pose.position.y = state_point_.pos(1);
    out.pose.position.z = state_point_.pos(2);
    out.pose.orientation.x = state_point_.rot.coeffs()[0];
    out.pose.orientation.y = state_point_.rot.coeffs()[1];
    out.pose.orientation.z = state_point_.rot.coeffs()[2];
    out.pose.orientation.w = state_point_.rot.coeffs()[3];
}

void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LaserMapping::PointBodyToWorld(const V3F &pi, PointType *const po) {
    V3D p_body(pi.x(), pi.y(), pi.z());
    V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = std::abs(po->z);
}

void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}
