#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "laser_mapping.hpp"
#include <vector>


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
        this->declare_parameter<bool>("publish.scan_body_pub_en", true);
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

        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        path_publish_en_ = this->get_parameter("publish.path_publish_en").as_bool();
        if (path_publish_en_) {
            RCLCPP_INFO(this->get_logger(), "Path Publish Enabled");
            path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        }

        scan_publish_en_ = this->get_parameter("publish.scan_publish_en").as_bool();
        if (scan_publish_en_) {
            RCLCPP_INFO(this->get_logger(), "Scan Publish Enabled");
            scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        }
        dense_publish_en_ = this->get_parameter("publish.dense_publish_en").as_bool();
        
        scan_effect_pub_en_ = this->get_parameter("publish.scan_effect_pub_en").as_bool();
        if (scan_effect_pub_en_) {
            RCLCPP_INFO(this->get_logger(), "Scan Effect Publish Enabled");
            scan_effect_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        }

        scan_body_pub_en_ = this->get_parameter("publish.scan_body_pub_en").as_bool();
        if (scan_body_pub_en_) {
            RCLCPP_INFO(this->get_logger(), "Scan Bodyframe Publish Enabled");
            scan_body_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        }
    }
private:
    std::shared_ptr<LaserMapping> laser_mapping_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr standard_pcl_sub_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_pcl_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Odometry Publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    // transform
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    bool path_publish_en_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    std::vector<geometry_msgs::msg::PoseStamped> poses_;

    bool scan_publish_en_;
    bool dense_publish_en_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_pub_;
   
    bool scan_effect_pub_en_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_effect_pub_;

    bool scan_body_pub_en_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_body_pub_;

    void run()
    {
        if (laser_mapping_->Run()) {
            // Publish Odometry
            nav_msgs::msg::Odometry odom_aft_mapped_;
            odom_aft_mapped_.header.frame_id = "camera_init";
            odom_aft_mapped_.header.stamp = rclcpp::Time(static_cast<int64_t>(laser_mapping_->GetLidarEndTime() * 1e9));
            odom_aft_mapped_.child_frame_id = "body";
            laser_mapping_->SetPosestamp(odom_aft_mapped_.pose);
            odom_pub_->publish(odom_aft_mapped_);

            auto P = laser_mapping_->GetP();
            for (int i = 0; i < 6; i ++)
            {
                int k = i < 3 ? i + 3 : i - 3;
                odom_aft_mapped_.pose.covariance[i*6 + 0] = P(k, 3);
                odom_aft_mapped_.pose.covariance[i*6 + 1] = P(k, 4);
                odom_aft_mapped_.pose.covariance[i*6 + 2] = P(k, 5);
                odom_aft_mapped_.pose.covariance[i*6 + 3] = P(k, 0);
                odom_aft_mapped_.pose.covariance[i*6 + 4] = P(k, 1);
                odom_aft_mapped_.pose.covariance[i*6 + 5] = P(k, 2);
            }

            geometry_msgs::msg::TransformStamped trans;
            trans.header.stamp = rclcpp::Time(static_cast<int64_t>(laser_mapping_->GetLidarEndTime() * 1e9));
            trans.header.frame_id = "camera_init";
            trans.child_frame_id = "body";
            trans.transform.translation.x = odom_aft_mapped_.pose.pose.position.x;
            trans.transform.translation.y = odom_aft_mapped_.pose.pose.position.y;
            trans.transform.translation.z = odom_aft_mapped_.pose.pose.position.z;
            trans.transform.rotation.w = odom_aft_mapped_.pose.pose.orientation.w;
            trans.transform.rotation.x = odom_aft_mapped_.pose.pose.orientation.x;
            trans.transform.rotation.y = odom_aft_mapped_.pose.pose.orientation.y;
            trans.transform.rotation.z = odom_aft_mapped_.pose.pose.orientation.z;
            tf_broadcaster_->sendTransform(trans);

            if (path_publish_en_) {
                geometry_msgs::msg::PoseStamped pose;
                laser_mapping_->SetPosestamp(pose);
                pose.header.frame_id =  "camera_init";
                pose.header.stamp = rclcpp::Time(static_cast<int64_t>(laser_mapping_->GetLidarEndTime() * 1e9));
                poses_.push_back(pose);
                nav_msgs::msg::Path path_;
                path_.poses = poses_;
                path_.header.frame_id = "camera_init";
                path_.header.stamp = rclcpp::Time(static_cast<int64_t>(laser_mapping_->GetLidarEndTime() * 1e9));
                path_pub_->publish(path_);
            }

            if (scan_publish_en_) {
                PointCloudType::Ptr scan = laser_mapping_->GetFrameWorld(dense_publish_en_);
                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*scan, msg);
                msg.header.frame_id = "camera_init";
                msg.header.stamp = rclcpp::Time(static_cast<int64_t>(laser_mapping_->GetLidarEndTime() * 1e9));
                scan_pub_->publish(msg);
            }

            if (scan_effect_pub_en_) {
                PointCloudType::Ptr scan = laser_mapping_->GetFrameEffectWorld();
                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*scan, msg);
                msg.header.frame_id = "camera_init";
                msg.header.stamp = rclcpp::Time(static_cast<int64_t>(laser_mapping_->GetLidarEndTime() * 1e9));
                scan_effect_pub_->publish(msg);
            }

            if (scan_body_pub_en_) {
                PointCloudType::Ptr scan = laser_mapping_->GetFrameBody();
                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*scan, msg);
                msg.header.frame_id = "body";
                msg.header.stamp = rclcpp::Time(static_cast<int64_t>(laser_mapping_->GetLidarEndTime() * 1e9));
                scan_body_pub_->publish(msg);
            }
        }
    }

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaserMappingNode>());
  rclcpp::shutdown();
  return 0;
}