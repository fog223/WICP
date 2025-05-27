//
// Created by fog on 2025/3/15.
//

#ifndef ICP_3D_H
#define ICP_3D_H

#include <numeric>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace Icp {
    using Vec3f = Eigen::Vector3f;
    using Vec3d = Eigen::Vector3d;
    using Vec4f = Eigen::Vector4f;
    using Vec4d = Eigen::Vector4d;
    using Vec6f = Eigen::Matrix<float, 6, 1>;
    using Vec6d = Eigen::Matrix<double, 6, 1>;
    using Mat3f = Eigen::Matrix3f;
    using Mat3d = Eigen::Matrix3d;
    using Mat4f = Eigen::Matrix4f;
    using Mat4d = Eigen::Matrix4d;
    using Mat6f = Eigen::Matrix<float, 6, 6>;
    using Mat6d = Eigen::Matrix<double, 6, 6>;

    // 定义系统中用到的点和点云类型
    using PointType = pcl::PointXYZ;
    using PointCloudType = pcl::PointCloud<PointType>;
    using CloudPtr = PointCloudType::Ptr;

    // 点云到Eigen的常用的转换函数
    inline Vec3f ToVec3f(const PointType &pt) { return pt.getVector3fMap(); }
    inline Vec3d ToVec3d(const PointType &pt) { return pt.getVector3fMap().cast<double>(); }

    template<typename S>
    inline PointType ToPointType(const Eigen::Matrix<S, 3, 1> &pt) {
        PointType p;
        p.x = pt.x();
        p.y = pt.y();
        p.z = pt.z();
        return p;
    }

    template<typename S>
    bool FitLine(std::vector<Eigen::Matrix<S, 3, 1> > &data, Eigen::Matrix<S, 3, 1> &origin,
                 Eigen::Matrix<S, 3, 1> &dir,
                 double eps = 0.2) {
        if (data.size() < 2) {
            return false;
        }

        origin = std::accumulate(data.begin(), data.end(), Eigen::Matrix<S, 3, 1>::Zero().eval()) / data.size();

        Eigen::MatrixXd Y(data.size(), 3);
        for (int i = 0; i < data.size(); ++i) {
            Y.row(i) = (data[i] - origin).transpose();
        }

        Eigen::JacobiSVD svd(Y, Eigen::ComputeFullV);
        dir = svd.matrixV().col(0);

        // check eps
        for (const auto &d: data) {
            if (dir.cross(d - origin).squaredNorm() > eps) {
                return false;
            }
        }

        return true;
    }

    template<typename S>
    bool FitPlane(std::vector<Eigen::Matrix<S, 3, 1> > &data, Eigen::Matrix<S, 4, 1> &plane_coeffs, double eps = 1e-2) {
        if (data.size() < 3) {
            return false;
        }

        Eigen::MatrixXd A(data.size(), 4);
        for (int i = 0; i < data.size(); ++i) {
            A.row(i).head<3>() = data[i].transpose();
            A.row(i)[3] = 1.0;
        }

        Eigen::JacobiSVD svd(A, Eigen::ComputeThinV);
        plane_coeffs = svd.matrixV().col(3);

        // check error eps
        for (int i = 0; i < data.size(); ++i) {
            double err = plane_coeffs.template head<3>().dot(data[i]) + plane_coeffs[3];
            if (err * err > eps) {
                return false;
            }
        }

        return true;
    }

    /**
     * 3D 形式的ICP
     * 先SetTarget, 再SetSource, 然后调用Align方法获取位姿
     *
     * ICP 求解R,t 将source点云配准到target点云上
     * 如果 p 是source点云中的点，那么R*p+t就得到target中的配对点
     */
    class WIcp3d {
    public:
        struct Options {
            int max_iteration_ = 50; // 最大迭代次数
            double overlap_ = 0.3; // 拒绝策略，重叠度约束
            double max_nn_distance_ = 0.1; // 点到点最近邻查找时阈值
            double max_line_distance_ = 0.05; // 点线最近邻查找时阈值
            double max_plane_distance_ = 0.05; // 平面最近邻查找时阈值
            int min_effective_pts_ = 10; // 最近邻点数阈值
            double rms_ = 1e-6; // 收敛判定条件

            std::string csv_filename_ = "result.csv";
        };

        struct MatchPair {
            Vec3d p; // 转换后的源点
            Vec3d n1;
            Vec3d q; // 目标中的对应点
            Vec3d n2;
            double dis2;
            float weight;

            /// Sort
            /// \param b
            /// \return
            bool
            operator<(const MatchPair &b) const {
                return dis2 < b.dis2;
            }
        };

        WIcp3d() {
        }

        WIcp3d(Options options) : options_(options) {
        }

        /// 设置目标的Scan
        void SetTarget(CloudPtr target) {
            target_ = target;
            kdtree_->setInputCloud(target_);
        }

        /// 正确转换后的source
        void SetGtSource(CloudPtr gt_source) {
            gt_source_ = gt_source;
        }

        /// 设置被配准的Scan
        void SetSource(CloudPtr source) {
            source_ = source;
        }

        void SetGroundTruth(const Mat4d &gt_pose) {
            gt_pose_ = gt_pose;
            gt_set_ = true;
        }

        /// 点到点
        bool AlignP2P(Mat4d &init_pose, int reject_method = 1);

        /// 点到线段
        bool AlignP2Line(Mat4d &init_pose, int reject_method = 1);

        /// 点到平面
        bool AlignP2Plane(Mat4d &init_pose, int reject_method = 1);

        bool AlignWP2P(Mat4d &init_pose);

        bool SensitivityAnalysis(Mat4d &init_pose);
        bool Comparison(Mat4d &init_pose);

    private:
        pcl::search::KdTree<PointType>::Ptr kdtree_{new pcl::search::KdTree<PointType>};

        CloudPtr target_ = nullptr;
        CloudPtr source_ = nullptr;
        CloudPtr gt_source_ = nullptr; // 用于对比计算误差

        bool gt_set_ = false; // 真值是否设置
        Mat4d gt_pose_;

        Options options_;
    };
} // namespace Icp

#endif  // ICP_3D_H
