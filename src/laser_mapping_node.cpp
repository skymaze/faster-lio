#include <rclcpp/rclcpp.hpp>
#include "laser_mapping.hpp"


class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode()
    : Node("laser_mapping_node")
    {
        this->declare_parameter<bool>("publish.path_publish_en", true);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.scan_effect_pub_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<std::string>("publish.tf_imu_frame", "body");
        this->declare_parameter<std::string>("publish.tf_world_frame", "camera_init");

        this->declare_parameter<bool>("pcd.save_pcd_en", false);
        this->declare_parameter<int>("pcd.save_interval", -1);

        this->declare_parameter<int>("num_max_iterations", 4);
        this->declare_parameter<double>("filter_size_surf_min", 0.5);
        this->declare_parameter<double>("filter_size_map_min", 0.0);
        this->declare_parameter<float>("esti_plane_threshold", 0.1);

        this->declare_parameter<float>("ivox.grid_resolution", 0.2);
        this->declare_parameter<int>("ivox.nearby_type", 18);

        this->declare_parameter<std::string>("common.lidar_topic", "/livox/lidar");
        this->declare_parameter<std::string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);

        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<float>("preprocess.time_scale", 1e-3);
        this->declare_parameter<int>("preprocess.lidar_type", 1);
        this->declare_parameter<int>("preprocess.num_scans", 16);
        this->declare_parameter<int>("preprocess.point_filter_num", 2);
        this->declare_parameter<bool>("preprocess.feature_extract_enable", false);

        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", false);
        this->declare_parameter<std::vector<double>>("mapping.extrinsic_t", {0, 0, 0});
        this->declare_parameter<std::vector<double>>("mapping.extrinsic_r", {0, 0, 0});


        auto config = LaserMappingConfig();

        this->get_parameter_or<int>("num_max_iterations", config.num_max_iterations, 4);
        this->get_parameter_or<double>("filter_size_surf_min", config.filter_size_surf_min, 0.5);
        this->get_parameter_or<double>("filter_size_map_min", config.filter_size_map_min, 0.0);
        this->get_parameter_or<float>("esti_plane_threshold", config.esti_plane_threshold, 0.1);

        this->get_parameter_or<float>("ivox.grid_resolution", config.ivox__grid_resolution, 0.2);
        this->get_parameter_or<int>("ivox.nearby_type", config.ivox__nearby_type, 18);

        this->get_parameter_or<bool>("common.time_sync_en", config.time_sync_en, false);

        this->get_parameter_or<double>("preprocess.blind", config.preprocess__blind, 0.01);
        this->get_parameter_or<float>("preprocess.time_scale", config.preprocess__time_scale, 1e-3);
        this->get_parameter_or<int>("preprocess.lidar_type", config.preprocess__lidar_type, 1);
        this->get_parameter_or<int>("preprocess.num_scans", config.preprocess__num_scans, 16);
        this->get_parameter_or<int>("preprocess.point_filter_num", config.preprocess__point_filter_num, 2);
        this->get_parameter_or<bool>("preprocess.feature_extract_enable", config.preprocess__feature_extract_enable, false);
        
        this->get_parameter_or<double>("mapping.gyr_cov", config.gyr_cov, 0.1);
        this->get_parameter_or<double>("mapping.acc_cov", config.acc_cov, 0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov", config.b_gyr_cov, 0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov", config.b_acc_cov, 0.0001);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", config.mapping__extrinsic_est_en, false);
        this->get_parameter_or<std::vector<double>>("mapping.extrinsic_t", config.mapping__extrinsic_t, std::vector<double>());
        this->get_parameter_or<std::vector<double>>("mapping.extrinsic_r", config.mapping__extrinsic_r, std::vector<double>());

        laser_mapping_ = std::make_shared<LaserMapping>(config);

        laser_mapping_->Init();

        if (config.preprocess__lidar_type == 1) {
            RCLCPP_INFO(this->get_logger(), "Livox Lidar");
            livox_pcl_sub_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                this->get_parameter("common.lidar_topic").as_string(),
                200000,
                std::bind(&LaserMapping::LivoxPCLCallBack, laser_mapping_, std::placeholders::_1)
            );
        } else {
            RCLCPP_INFO(this->get_logger(), "Standard Lidar");
            standard_pcl_sub_= this->create_subscription<sensor_msgs::msg::PointCloud2>(
                this->get_parameter("common.lidar_topic").as_string(),
                200000,
                std::bind(&LaserMapping::StandardPCLCallBack, laser_mapping_, std::placeholders::_1)
            );
        }
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            this->get_parameter("common.imu_topic").as_string(),
            200000,
            std::bind(&LaserMapping::IMUCallBack, laser_mapping_, std::placeholders::_1)
        );

        timer_ = this->create_wall_timer(std::chrono::milliseconds(1), std::bind(&LaserMappingNode::run, this));
    }
private:
    std::shared_ptr<LaserMapping> laser_mapping_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr standard_pcl_sub_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_pcl_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    void run() const
    {
        laser_mapping_->Run();
    }

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaserMappingNode>());
  rclcpp::shutdown();
  return 0;
}