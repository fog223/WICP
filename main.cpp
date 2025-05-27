#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include "icp_3d.h"

int main(int argc, char **argv) {
    // 读取点云
    pcl::PointCloud<Icp::PointType>::Ptr gt_source(new pcl::PointCloud<Icp::PointType>);
    pcl::PointCloud<Icp::PointType>::Ptr source(new pcl::PointCloud<Icp::PointType>);
    pcl::PointCloud<Icp::PointType>::Ptr target(new pcl::PointCloud<Icp::PointType>);
    // pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/HS1/1_sampled_reg.ply", *gt_source);
    // pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/HS1/1_sampled_reg_trans.ply", *source);
    // pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/HS1/2_sampled_reg.ply", *target);
    pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/HS2/1_sampled_reg.ply", *gt_source);
    pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/HS2/1_sampled_reg_trans.ply", *source);
    pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/HS2/2_sampled_reg.ply", *target);
    // pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/Park/140706_10_sampled.ply", *gt_source);
    // pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/Park/140706_10_sampled_trans.ply", *source);
    // pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/Park/140706_11_sampled.ply", *target);
    // pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/Campus/04_sampled.ply", *gt_source);
    // pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/Campus/04_sampled_trans.ply", *source);
    // pcl::io::loadPLYFile<Icp::PointType>("E:/Data/WICP/Campus/05_sampled.ply", *target);

    Icp::WIcp3d icp3d_;
    icp3d_.SetTarget(target);
    icp3d_.SetSource(source);
    icp3d_.SetGtSource(gt_source);
    Icp::Mat4d inipose = Icp::Mat4d::Identity();
    icp3d_.Comparison(inipose);
    return 0;
}
