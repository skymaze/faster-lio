#pragma once

#include <rclcpp/rclcpp.hpp>
#include <execution>
#include <fstream>
#include "utils.hpp"
#include "options.h"

#include <pcl/filters/voxel_grid.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

#include <condition_variable>
#include <thread>

#include "imu_processing.hpp"
#include "ivox3d/ivox3d.h"
#include "pointcloud_preprocess.h"


struct LaserMappingConfig {
    int num_max_iterations;
    float esti_plane_threshold;
    bool time_sync_en;
    double filter_size_surf_min;
    double filter_size_map_min;
    double gyr_cov;
    double acc_cov;
    double b_gyr_cov;
    double b_acc_cov;
    double preprocess__blind;
    float preprocess__time_scale;
    int preprocess__lidar_type;
    int preprocess__num_scans;
    int preprocess__point_filter_num;
    bool preprocess__feature_extract_enable;
    bool mapping__extrinsic_est_en;
    std::vector<double> mapping__extrinsic_t;
    std::vector<double> mapping__extrinsic_r;
    float ivox__grid_resolution;
    int ivox__nearby_type;
};

class LaserMapping {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

#ifdef IVOX_NODE_TYPE_PHC
    using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>;
#else
    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;
#endif

    LaserMapping(const LaserMappingConfig &config);
    ~LaserMapping();

    bool Init();

    bool Run();

    // callbacks of lidar and imu
    void StandardPCLCallBack(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
    void LivoxPCLCallBack(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg);
    void IMUCallBack(const sensor_msgs::msg::Imu::UniquePtr &msg_in);

    // sync lidar with imu
    bool SyncPackages();

    /// interface of mtk, customized obseravtion model
    void ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);

    double GetLidarEndTime() const { return lidar_end_time_; }
    auto GetP() const { return kf_.get_P(); }

    template <typename T>
    void SetPosestamp(T &out);

    PointCloudType::Ptr GetFrameWorld(bool dense = false);
    PointCloudType::Ptr GetFrameBody();
    PointCloudType::Ptr GetFrameEffectWorld();

   private:
    void PointBodyToWorld(PointType const *pi, PointType *const po);
    void PointBodyToWorld(const V3F &pi, PointType *const po);
    void PointBodyLidarToIMU(PointType const *const pi, PointType *const po);

    void MapIncremental();

    void PrintState(const state_ikfom &s);

   private:
    LaserMappingConfig config_;
    /// modules
    IVoxType::Options ivox_options_;
    std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process

    /// point clouds data
    CloudPtr scan_undistort_{new PointCloudType()};   // scan after undistortion
    CloudPtr scan_down_body_{new PointCloudType()};   // downsampled scan in body
    CloudPtr scan_down_world_{new PointCloudType()};  // downsampled scan in world
    std::vector<PointVector> nearest_points_;         // nearest points of current scan
    VV4F corr_pts_;                           // inlier pts
    VV4F corr_norm_;                          // inlier plane norms
    pcl::VoxelGrid<PointType> voxel_scan_;            // voxel filter for current scan
    std::vector<float> residuals_;                    // point-to-plane residuals
    std::vector<bool> point_selected_surf_;           // selected points
    VV4F plane_coef_;                         // plane coeffs

    std::mutex mtx_buffer_;
    std::deque<double> time_buffer_;
    std::deque<PointCloudType::Ptr> lidar_buffer_;
    std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer_;

    /// options
    double timediff_lidar_wrt_imu_ = 0.0;
    double last_timestamp_lidar_ = 0;
    double lidar_end_time_ = 0;
    double last_timestamp_imu_ = -1.0;
    double first_lidar_time_ = 0.0;
    bool lidar_pushed_ = false;

    /// statistics and flags ///
    int scan_count_ = 0;
    int publish_count_ = 0;
    bool flg_first_scan_ = true;
    bool flg_EKF_inited_ = false;
    int pcd_index_ = 0;
    double lidar_mean_scantime_ = 0.0;
    int scan_num_ = 0;
    bool timediff_set_flg_ = false;
    int effect_feat_num_ = 0, frame_num_ = 0;

    ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
    MeasureGroup measures_;                    // sync IMU and lidar scan
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;  // esekf
    state_ikfom state_point_;                          // ekf current state
    vect3 pos_lidar_;                                  // lidar position after eskf update
    V3D euler_cur_ = V3D::Zero();      // rotation in euler angles
};
