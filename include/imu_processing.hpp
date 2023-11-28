#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cmath>
#include <deque>
#include <fstream>

#include "common_lib.hpp"
#include "so3_math.h"
#include "use_ikfom.hpp"
#include "utils.hpp"


constexpr int MAX_INI_COUNT = 20;

/// IMU Process and undistortion
class ImuProcess {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();
    ~ImuProcess();

    void Reset();
    void SetExtrinsic(const V3D &transl, const M3D &rot);
    void SetGyrCov(const V3D &scaler);
    void SetAccCov(const V3D &scaler);
    void SetGyrBiasCov(const V3D &b_g);
    void SetAccBiasCov(const V3D &b_a);
    void Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                 PointCloudType::Ptr pcl_un_);
    static bool TimeList(const PointType &x, const PointType &y) { return (x.curvature < y.curvature); };

    std::ofstream fout_imu_;
    Eigen::Matrix<double, 12, 12> Q_;
    V3D cov_acc_;
    V3D cov_gyr_;
    V3D cov_acc_scale_;
    V3D cov_gyr_scale_;
    V3D cov_bias_gyr_;
    V3D cov_bias_acc_;

   private:
    void IMUInit(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
    void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                      PointCloudType &pcl_out);

    PointCloudType::Ptr cur_pcl_un_;
    sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;
    std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> v_imu_;
    std::vector<Pose6D> IMUpose_;
    std::vector<M3D> v_rot_pcl_;
    M3D Lidar_R_wrt_IMU_;
    V3D Lidar_T_wrt_IMU_;
    V3D mean_acc_;
    V3D mean_gyr_;
    V3D angvel_last_;
    V3D acc_s_last_;
    double last_lidar_end_time_ = 0;
    int init_iter_num_ = 1;
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
};
