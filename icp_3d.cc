#include "icp_3d.h"

#include <execution>
#include <spdlog/spdlog.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d_omp.h>

namespace Icp {
    bool WIcp3d::AlignP2P(Mat4d &init_pose, int reject_method) {
        spdlog::info("aligning with point to point");
        assert(target_!=nullptr && source_!=nullptr);

        std::ofstream outfile(options_.csv_filename_);
        // Write the header row
        outfile << "time, error\n";
        // 计算初始逐点误差
        double pointswise_error = 0;
        for (int idx = 0; idx < source_->size(); ++idx) {
            Vec3d ps = init_pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                       + init_pose.block<3, 1>(0, 3);
            Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
            pointswise_error += (gt_ps - ps).norm();
        }
        pointswise_error /= source_->size();
        spdlog::info("pointswise_error: {}", pointswise_error);
        outfile << 0 << "," << pointswise_error << "\n";

        double final_rms = std::numeric_limits<double>::max();
        Mat4d pose = init_pose;
        double time_start = clock();
        double time_lost = 0.0;
        for (int iter = 0; iter < options_.max_iteration_; ++iter) {
            // 对点的索引，预先生成
            std::vector<int> index(source_->points.size());
            for (int i = 0; i < index.size(); ++i) {
                index[i] = i;
            }

            std::vector<MatchPair> matches(source_->size());
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
                auto p = ToVec3d(source_->points[idx]);
                Vec3d ps = pose.block<3, 3>(0, 0) * p + pose.block<3, 1>(0, 3);
                std::vector<int> nn; // 存储查询近邻点索引
                std::vector<float> dist; // 存储近邻点对应距离的平方
                kdtree_->nearestKSearch(ToPointType(ps), 1, nn, dist); // 这里取1个最近邻

                if (!nn.empty()) {
                    Vec3d q = ToVec3d(target_->points[nn[0]]);
                    double dis2 = (q - ps).norm();

                    // 有效点对
                    matches[idx].p = ps;
                    matches[idx].q = q;
                    matches[idx].dis2 = dis2;
                }
            });

            // 均值计算
            double rms = 0;
            Vec3d centerP = Vec3d::Zero();
            Vec3d centerQ = Vec3d::Zero();
            std::sort(matches.begin(), matches.end());
            int effnum = 0;
            if (reject_method == 1) {
                effnum = options_.overlap_ * matches.size();
                for (int idx = 0; idx < effnum; ++idx) {
                    centerP += matches[idx].p; // 变换后的源点累加
                    centerQ += matches[idx].q; // 目标点累加
                    rms += matches[idx].dis2;
                }
            }
            if (reject_method == 2) {
                for (int idx = 0; idx < matches.size(); ++idx) {
                    if (matches[idx].dis2 < options_.max_nn_distance_) {
                        effnum++;
                        centerP += matches[idx].p; // 变换后的源点累加
                        centerQ += matches[idx].q; // 目标点累加
                        rms += matches[idx].dis2;
                    } else
                        break;
                }
            }

            rms /= effnum;
            rms = sqrt(rms);
            centerP /= effnum;
            centerQ /= effnum;

            // 构建协方差矩阵
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (int idx = 0; idx < effnum; ++idx) {
                Vec3d p_prime = matches[idx].p - centerP;
                Vec3d q_prime = matches[idx].q - centerQ;
                H += p_prime * q_prime.transpose();
            }

            // SVD分解
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            // 计算旋转矩阵和平移向量
            Mat3d R = V * U.transpose();
            if (R.determinant() < 0) {
                R = -R;
            }
            Vec3d t = centerQ - R * centerP;

            // 更新姿态
            pose.block<3, 1>(0, 3) = R * pose.block<3, 1>(0, 3) + t;
            pose.block<3, 3>(0, 0) = R * pose.block<3, 3>(0, 0);
            spdlog::info("iter: {}, eff: {}, rms: {}", iter, effnum, rms);

            if (fabs(rms - final_rms) < options_.rms_) {
                spdlog::info("converged.");
                break;
            }
            final_rms = rms;

            double time_end = clock();
            double itertime = (double) (time_end - time_start) / CLOCKS_PER_SEC;
            itertime -= time_lost;

            // 计算逐点误差
            double pointswise_error = 0;
            for (int idx = 0; idx < source_->size(); ++idx) {
                Vec3d ps = pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                           + pose.block<3, 1>(0, 3);
                Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
                pointswise_error += (gt_ps - ps).norm();
            }
            pointswise_error /= source_->size();
            spdlog::info("pointswise_error: {}", pointswise_error);
            outfile << itertime << "," << pointswise_error << "\n";

            // 去除计算逐点误差的时间
            double time_end2 = clock();
            time_lost += (double) (time_end2 - time_end) / CLOCKS_PER_SEC;
        }
        outfile.close();

        if (gt_set_) {
            Eigen::AngleAxisd aaDiff(gt_pose_.block<3, 3>(0, 0).transpose() * pose.block<3, 3>(0, 0));
            double angleDiff = aaDiff.angle();

            // Convert radians to angles
            auto R_error = angleDiff * 180.0 / M_PI;
            //计算平移误差
            auto t_error = (gt_pose_.block<3, 1>(0, 3) - pose.block<3, 1>(0, 3)).norm();

            spdlog::info("R_error: {}", R_error);
            spdlog::info("t_error: {}", t_error);

            // 计算逐点误差
            double pointswise_error = 0;
            for (int idx = 0; idx < source_->size(); ++idx) {
                Vec3d ps = pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                           + pose.block<3, 1>(0, 3);
                Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
                pointswise_error += (gt_ps - ps).norm();
            }
            pointswise_error /= source_->size();
            spdlog::info("pointswise_error: {}", pointswise_error);
        }

        std::cout << pose << std::endl;
        init_pose = pose;
        return true;
    }

    bool WIcp3d::AlignP2Line(Mat4d &init_pose, int reject_method) {
        spdlog::info("aligning with point to point");
        assert(target_!=nullptr && source_!=nullptr);

        std::ofstream outfile(options_.csv_filename_);
        // Write the header row
        outfile << "p2line\n";
        double final_rms = std::numeric_limits<double>::max();
        Mat4d pose = init_pose;
        for (int iter = 0; iter < options_.max_iteration_; ++iter) {
            // 对点的索引，预先生成
            std::vector<int> index(source_->points.size());
            for (int i = 0; i < index.size(); ++i) {
                index[i] = i;
            }

            std::vector<MatchPair> matches(source_->size());
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
                auto p = ToVec3d(source_->points[idx]);
                Vec3d ps = pose.block<3, 3>(0, 0) * p + pose.block<3, 1>(0, 3);
                std::vector<int> nn; // 存储查询近邻点索引
                std::vector<float> dist; // 存储近邻点对应距离的平方
                kdtree_->nearestKSearch(ToPointType(ps), 5, nn, dist); // 这里取5个最近邻

                if (nn.size() > 3) {
                    // convert to eigen
                    std::vector<Vec3d> nn_eigen;
                    for (int i = 0; i < 5; ++i) {
                        nn_eigen.emplace_back(ToVec3d(target_->points[nn[i]]));
                    }

                    Vec3d d, p0;
                    if (!FitLine(nn_eigen, p0, d, options_.max_line_distance_)) {
                        matches[idx].p = ps;
                        matches[idx].q = ToVec3d(target_->points[nn[0]]);
                        matches[idx].dis2 = (matches[idx].q - matches[idx].p).norm();;
                        return;
                    }
                    double t = (ps - p0).dot(d) / d.dot(d);
                    Vec3d q = p0 + t * d; // 在直线上的垂足点
                    double dis2 = (q - ps).norm();

                    // 有效点对
                    matches[idx].p = ps;
                    matches[idx].q = q;
                    matches[idx].dis2 = dis2;
                }
            });

            // 均值计算
            double rms = 0;
            Vec3d centerP = Vec3d::Zero();
            Vec3d centerQ = Vec3d::Zero();
            std::sort(matches.begin(), matches.end());
            int effnum = 0;
            if (reject_method == 1) {
                effnum = options_.overlap_ * matches.size();
                for (int idx = 0; idx < effnum; ++idx) {
                    centerP += matches[idx].p; // 变换后的源点累加
                    centerQ += matches[idx].q; // 目标点累加
                    rms += matches[idx].dis2;
                }
            }
            if (reject_method == 2) {
                for (int idx = 0; idx < matches.size(); ++idx) {
                    if (matches[idx].dis2 < options_.max_line_distance_) {
                        effnum++;
                        centerP += matches[idx].p; // 变换后的源点累加
                        centerQ += matches[idx].q; // 目标点累加
                        rms += matches[idx].dis2;
                    } else
                        break;
                }
            }

            rms /= effnum;
            rms = sqrt(rms);
            centerP /= effnum;
            centerQ /= effnum;

            // 构建协方差矩阵
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (int idx = 0; idx < effnum; ++idx) {
                Vec3d p_prime = matches[idx].p - centerP;
                Vec3d q_prime = matches[idx].q - centerQ;
                H += p_prime * q_prime.transpose();
            }

            // SVD分解
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            // 计算旋转矩阵和平移向量
            Mat3d R = V * U.transpose();
            if (R.determinant() < 0) {
                R = -R;
            }
            Vec3d t = centerQ - R * centerP;

            // 更新姿态
            pose.block<3, 1>(0, 3) = R * pose.block<3, 1>(0, 3) + t;
            pose.block<3, 3>(0, 0) = R * pose.block<3, 3>(0, 0);
            spdlog::info("iter: {}, eff: {}, rms: {}", iter, effnum, rms);

            if (fabs(rms - final_rms) < options_.rms_) {
                spdlog::info("converged.");
                break;
            }
            final_rms = rms;

            // 计算逐点误差
            double pointswise_error = 0;
            for (int idx = 0; idx < source_->size(); ++idx) {
                Vec3d ps = pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                           + pose.block<3, 1>(0, 3);
                Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
                pointswise_error += (gt_ps - ps).norm();
            }
            pointswise_error /= source_->size();
            spdlog::info("pointswise_error: {}", pointswise_error);
            outfile << pointswise_error << "\n";
        }
        outfile.close();

        if (gt_set_) {
            Eigen::AngleAxisd aaDiff(gt_pose_.block<3, 3>(0, 0).transpose() * pose.block<3, 3>(0, 0));
            double angleDiff = aaDiff.angle();

            // Convert radians to angles
            auto R_error = angleDiff * 180.0 / M_PI;
            //计算平移误差
            auto t_error = (gt_pose_.block<3, 1>(0, 3) - pose.block<3, 1>(0, 3)).norm();

            spdlog::info("R_error: {}", R_error);
            spdlog::info("t_error: {}", t_error);

            // 计算逐点误差
            double pointswise_error = 0;
            for (int idx = 0; idx < source_->size(); ++idx) {
                Vec3d ps = pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                           + pose.block<3, 1>(0, 3);
                Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
                pointswise_error += (gt_ps - ps).norm();
            }
            pointswise_error /= source_->size();
            spdlog::info("pointswise_error: {}", pointswise_error);
        }

        std::cout << pose << std::endl;
        init_pose = pose;
        return true;
    }

    bool WIcp3d::AlignP2Plane(Mat4d &init_pose, int reject_method) {
        spdlog::info("aligning with point to point");
        assert(target_!=nullptr && source_!=nullptr);

        std::ofstream outfile(options_.csv_filename_);
        // Write the header row
        outfile << "p2plane\n";
        double final_rms = std::numeric_limits<double>::max();
        Mat4d pose = init_pose;
        for (int iter = 0; iter < options_.max_iteration_; ++iter) {
            // 对点的索引，预先生成
            std::vector<int> index(source_->points.size());
            for (int i = 0; i < index.size(); ++i) {
                index[i] = i;
            }

            std::vector<MatchPair> matches(source_->size());
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
                auto p = ToVec3d(source_->points[idx]);
                Vec3d ps = pose.block<3, 3>(0, 0) * p + pose.block<3, 1>(0, 3);
                std::vector<int> nn; // 存储查询近邻点索引
                std::vector<float> dist; // 存储近邻点对应距离的平方
                kdtree_->nearestKSearch(ToPointType(ps), 5, nn, dist); // 这里取5个最近邻

                if (nn.size() > 3) {
                    // convert to eigen
                    std::vector<Vec3d> nn_eigen;
                    for (int i = 0; i < 5; ++i) {
                        nn_eigen.emplace_back(ToVec3d(target_->points[nn[i]]));
                    }

                    Vec4d n;
                    if (!FitPlane(nn_eigen, n)) {
                        matches[idx].p = ps;
                        matches[idx].q = ToVec3d(target_->points[nn[0]]);
                        matches[idx].dis2 = (matches[idx].q - matches[idx].p).norm();;
                        return;
                    }
                    double t = n.head<3>().dot(ps) + n[3]; // 法向量是单位向量，模长为1
                    Vec3d q = ps - t * n.head<3>(); // 在平面上的垂足点
                    double dis2 = (q - ps).norm();

                    matches[idx].p = ps;
                    matches[idx].q = q;
                    matches[idx].dis2 = dis2;
                }
            });

            // 均值计算
            double rms = 0;
            Vec3d centerP = Vec3d::Zero();
            Vec3d centerQ = Vec3d::Zero();
            std::sort(matches.begin(), matches.end());
            int effnum = 0;
            if (reject_method == 1) {
                effnum = options_.overlap_ * matches.size();
                for (int idx = 0; idx < effnum; ++idx) {
                    centerP += matches[idx].p; // 变换后的源点累加
                    centerQ += matches[idx].q; // 目标点累加
                    rms += matches[idx].dis2;
                }
            }
            if (reject_method == 2) {
                for (int idx = 0; idx < matches.size(); ++idx) {
                    if (matches[idx].dis2 < options_.max_plane_distance_) {
                        effnum++;
                        centerP += matches[idx].p; // 变换后的源点累加
                        centerQ += matches[idx].q; // 目标点累加
                        rms += matches[idx].dis2;
                    } else
                        break;
                }
            }

            rms /= effnum;
            rms = sqrt(rms);
            centerP /= effnum;
            centerQ /= effnum;

            // 构建协方差矩阵
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (int idx = 0; idx < effnum; ++idx) {
                Vec3d p_prime = matches[idx].p - centerP;
                Vec3d q_prime = matches[idx].q - centerQ;
                H += p_prime * q_prime.transpose();
            }

            // SVD分解
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            // 计算旋转矩阵和平移向量
            Mat3d R = V * U.transpose();
            if (R.determinant() < 0) {
                R = -R;
            }
            Vec3d t = centerQ - R * centerP;

            // 更新姿态
            pose.block<3, 1>(0, 3) = R * pose.block<3, 1>(0, 3) + t;
            pose.block<3, 3>(0, 0) = R * pose.block<3, 3>(0, 0);
            spdlog::info("iter: {}, eff: {}, rms: {}", iter, effnum, rms);

            if (fabs(rms - final_rms) < options_.rms_) {
                spdlog::info("converged.");
                break;
            }
            if (rms > final_rms) {
                spdlog::info("rms up.");
                break;
            }
            final_rms = rms;

            // 计算逐点误差
            double pointswise_error = 0;
            for (int idx = 0; idx < source_->size(); ++idx) {
                Vec3d ps = pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                           + pose.block<3, 1>(0, 3);
                Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
                pointswise_error += (gt_ps - ps).norm();
            }
            pointswise_error /= source_->size();
            spdlog::info("pointswise_error: {}", pointswise_error);
            outfile << pointswise_error << "\n";
        }
        outfile.close();

        if (gt_set_) {
            Eigen::AngleAxisd aaDiff(gt_pose_.block<3, 3>(0, 0).transpose() * pose.block<3, 3>(0, 0));
            double angleDiff = aaDiff.angle();

            // Convert radians to angles
            auto R_error = angleDiff * 180.0 / M_PI;
            //计算平移误差
            auto t_error = (gt_pose_.block<3, 1>(0, 3) - pose.block<3, 1>(0, 3)).norm();

            spdlog::info("R_error: {}", R_error);
            spdlog::info("t_error: {}", t_error);

            // 计算逐点误差
            double pointswise_error = 0;
            for (int idx = 0; idx < source_->size(); ++idx) {
                Vec3d ps = pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                           + pose.block<3, 1>(0, 3);
                Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
                pointswise_error += (gt_ps - ps).norm();
            }
            pointswise_error /= source_->size();
            spdlog::info("pointswise_error: {}", pointswise_error);
        }

        std::cout << pose << std::endl;
        init_pose = pose;
        return true;
    }

    bool WIcp3d::AlignWP2P(Mat4d &init_pose) {
        spdlog::info("aligning with weight point to point");
        spdlog::info("overlap: {}", options_.overlap_);
        assert(target_!=nullptr && source_!=nullptr);

        std::ofstream outfile(options_.csv_filename_);
        // Write the header row
        outfile << "time, error\n";
        // 计算初始逐点误差
        double pointswise_error = 0;
        for (int idx = 0; idx < source_->size(); ++idx) {
            Vec3d ps = init_pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                       + init_pose.block<3, 1>(0, 3);
            Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
            pointswise_error += (gt_ps - ps).norm();
        }
        pointswise_error /= source_->size();
        spdlog::info("pointswise_error: {}", pointswise_error);
        outfile << 0 << "," << pointswise_error << "\n";
        double time_start = clock();
        double time_lost = 0.0;

        double final_rms = std::numeric_limits<double>::max();
        Mat4d pose = init_pose;
        double total_findtime = 0;
        double total_svdtime = 0;
        for (int iter = 0; iter < options_.max_iteration_; ++iter) {
            // 对点的索引，预先生成
            std::vector<int> index(source_->points.size());
            for (int i = 0; i < index.size(); ++i) {
                index[i] = i;
            }

            double time_match_start = clock();
            std::vector<MatchPair> matches(source_->size());
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
                auto p = ToVec3d(source_->points[idx]);
                Vec3d ps = pose.block<3, 3>(0, 0) * p + pose.block<3, 1>(0, 3);
                std::vector<int> nn; // 存储查询近邻点索引
                std::vector<float> dist; // 存储近邻点对应距离的平方
                kdtree_->nearestKSearch(ToPointType(ps), 1, nn, dist); // 这里取1个最近邻
                Vec3d q = ToVec3d(target_->points[nn[0]]);
                double dis2 = (q - ps).norm();
                // 点对
                matches[idx].p = ps;
                matches[idx].q = q;
                matches[idx].dis2 = dis2;
            });
            double time_match_end = clock();
            double match_time = (double) (time_match_end - time_match_start) / CLOCKS_PER_SEC;
            total_findtime += match_time;

            // 均值计算
            double time_rt_start = clock();
            double rms = 0;
            double totalweight = 0;
            Vec3d centerP = Vec3d::Zero();
            Vec3d centerQ = Vec3d::Zero();
            int effnum = options_.overlap_ * matches.size();
            std::sort(matches.begin(), matches.end());
            for (int idx = 0; idx < effnum; ++idx) {
                matches[idx].weight = 10 * matches[idx].dis2 / matches[effnum - 1].dis2;
                // matches[idx].weight = 1;
                // matches[idx].weight = 1 - (matches[idx].dis2 / matches[effnum - 1].dis2);
                centerP += matches[idx].weight * matches[idx].p; // 变换后的源点累加
                centerQ += matches[idx].weight * matches[idx].q; // 目标点累加
                rms += matches[idx].weight * matches[idx].dis2;
                totalweight += matches[idx].weight;
            }

            rms /= totalweight;
            rms = sqrt(rms);
            centerP /= totalweight;
            centerQ /= totalweight;

            // 构建协方差矩阵
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (int idx = 0; idx < effnum; ++idx) {
                Vec3d p_prime = matches[idx].p - centerP;
                Vec3d q_prime = matches[idx].q - centerQ;
                H += matches[idx].weight * p_prime * q_prime.transpose();
            }

            // SVD分解
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            // 计算旋转矩阵和平移向量
            Mat3d R = V * U.transpose();
            if (R.determinant() < 0) {
                R = -R;
            }
            Vec3d t = centerQ - R * centerP;

            double time_rt_end = clock();
            double rt_time = (double) (time_rt_end - time_rt_start) / CLOCKS_PER_SEC;
            total_svdtime += rt_time;

            // 更新姿态
            pose.block<3, 1>(0, 3) = R * pose.block<3, 1>(0, 3) + t;
            pose.block<3, 3>(0, 0) = R * pose.block<3, 3>(0, 0);
            spdlog::info("iter: {}, rms: {}", iter, rms);

            // if (fabs(rms - final_rms) < options_.rms_) {
            //     spdlog::info("converged.");
            //     break;
            // }
            final_rms = rms;

            double time_end = clock();
            double itertime = (double) (time_end - time_start) / CLOCKS_PER_SEC;
            itertime -= time_lost;

            // 计算逐点误差
            double pointswise_error = 0;
            for (int idx = 0; idx < source_->size(); ++idx) {
                Vec3d ps = pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                           + pose.block<3, 1>(0, 3);
                Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
                pointswise_error += (gt_ps - ps).norm();
            }
            pointswise_error /= source_->size();
            spdlog::info("pointswise_error: {}", pointswise_error);
            outfile << itertime << "," << pointswise_error << "\n";

            // 去除计算逐点误差的时间
            double time_end2 = clock();
            time_lost += (double) (time_end2 - time_end) / CLOCKS_PER_SEC;
        }
        outfile.close();

        spdlog::info("find time: {}", total_findtime / options_.max_iteration_);
        spdlog::info("svd time: {}", total_svdtime / options_.max_iteration_);

        if (gt_set_) {
            Eigen::AngleAxisd aaDiff(gt_pose_.block<3, 3>(0, 0).transpose() * pose.block<3, 3>(0, 0));
            double angleDiff = aaDiff.angle();

            // Convert radians to angles
            auto R_error = angleDiff * 180.0 / M_PI;
            //计算平移误差
            auto t_error = (gt_pose_.block<3, 1>(0, 3) - pose.block<3, 1>(0, 3)).norm();

            spdlog::info("R_error: {}", R_error);
            spdlog::info("t_error: {}", t_error);

            // 计算逐点误差
            double pointswise_error = 0;
            for (int idx = 0; idx < source_->size(); ++idx) {
                Vec3d ps = pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                           + pose.block<3, 1>(0, 3);
                Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
                pointswise_error += (gt_ps - ps).norm();
            }
            pointswise_error /= source_->size();
            spdlog::info("pointswise_error: {}", pointswise_error);
        }

        std::cout << pose << std::endl;
        init_pose = pose;
        return true;
    }

    bool WIcp3d::SensitivityAnalysis(Mat4d &init_pose) {
        spdlog::info("aligning with weight point to point");
        spdlog::info("overlap: {}", options_.overlap_);
        assert(target_!=nullptr && source_!=nullptr);

        std::ofstream outfile(options_.csv_filename_);
        // Write the header row
        outfile << "time, error\n";

        for (int i = 1; i <= 10; ++i) {
            spdlog::info("SensitivityAnalysis iteration: {}", i);
            double time_start = clock();
            double final_rms = std::numeric_limits<double>::max();
            Mat4d pose = init_pose;
            for (int iter = 0; iter < options_.max_iteration_; ++iter) {
                // 对点的索引，预先生成
                std::vector<int> index(source_->points.size());
                for (int i = 0; i < index.size(); ++i) {
                    index[i] = i;
                }

                std::vector<MatchPair> matches(source_->size());
                std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
                    auto p = ToVec3d(source_->points[idx]);
                    Vec3d ps = pose.block<3, 3>(0, 0) * p + pose.block<3, 1>(0, 3);
                    std::vector<int> nn; // 存储查询近邻点索引
                    std::vector<float> dist; // 存储近邻点对应距离的平方
                    kdtree_->nearestKSearch(ToPointType(ps), 1, nn, dist); // 这里取1个最近邻
                    Vec3d q = ToVec3d(target_->points[nn[0]]);
                    double dis2 = (q - ps).norm();
                    // 点对
                    matches[idx].p = ps;
                    matches[idx].q = q;
                    matches[idx].dis2 = dis2;
                });

                // 均值计算
                double rms = 0;
                double totalweight = 0;
                Vec3d centerP = Vec3d::Zero();
                Vec3d centerQ = Vec3d::Zero();
                int effnum = options_.overlap_ * matches.size();
                std::sort(matches.begin(), matches.end());
                for (int idx = 0; idx < effnum; ++idx) {
                    matches[idx].weight = 5 * i * matches[idx].dis2 / matches[effnum - 1].dis2;
                    // matches[idx].weight = 1;
                    // matches[idx].weight = 1 - (matches[idx].dis2 / matches[effnum - 1].dis2);
                    centerP += matches[idx].weight * matches[idx].p; // 变换后的源点累加
                    centerQ += matches[idx].weight * matches[idx].q; // 目标点累加
                    rms += matches[idx].weight * matches[idx].dis2;
                    totalweight += matches[idx].weight;
                }

                rms /= totalweight;
                rms = sqrt(rms);
                centerP /= totalweight;
                centerQ /= totalweight;

                // 构建协方差矩阵
                Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
                for (int idx = 0; idx < effnum; ++idx) {
                    Vec3d p_prime = matches[idx].p - centerP;
                    Vec3d q_prime = matches[idx].q - centerQ;
                    H += matches[idx].weight * p_prime * q_prime.transpose();
                }

                // SVD分解
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3d U = svd.matrixU();
                Eigen::Matrix3d V = svd.matrixV();

                // 计算旋转矩阵和平移向量
                Mat3d R = V * U.transpose();
                if (R.determinant() < 0) {
                    R = -R;
                }
                Vec3d t = centerQ - R * centerP;

                // 更新姿态
                pose.block<3, 1>(0, 3) = R * pose.block<3, 1>(0, 3) + t;
                pose.block<3, 3>(0, 0) = R * pose.block<3, 3>(0, 0);
                spdlog::info("iter: {}, rms: {}", iter, rms);

                // if (fabs(rms - final_rms) < options_.rms_) {
                //     spdlog::info("converged.");
                //     break;
                // }
                final_rms = rms;
            }

            double time_end = clock();
            double cos_time = (double) (time_end - time_start) / CLOCKS_PER_SEC;

            // 计算逐点误差
            double pointswise_error = 0;
            for (int idx = 0; idx < source_->size(); ++idx) {
                Vec3d ps = pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                           + pose.block<3, 1>(0, 3);
                Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
                pointswise_error += (gt_ps - ps).norm();
            }
            pointswise_error /= source_->size();
            spdlog::info("pointswise_error: {}", pointswise_error);
            outfile << cos_time << "," << pointswise_error << "\n";
        }

        outfile.close();
        return true;
    }

    bool WIcp3d::Comparison(Mat4d &init_pose) {
        spdlog::info("aligning with weight point to point");
        spdlog::info("overlap: {}", options_.overlap_);
        assert(target_!=nullptr && source_!=nullptr);

        spdlog::info("Normal Estimation in progress...");
        pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
        ne.setSearchMethod(tree);
        ne.setKSearch(30);
        ne.setInputCloud(target_);
        ne.compute(*target_normals);
        ne.setInputCloud(source_);
        ne.compute(*source_normals);

        std::ofstream outfile(options_.csv_filename_);
        // Write the header row
        outfile << "error\n";
        // 计算初始逐点误差
        double pointswise_error = 0;
        for (int idx = 0; idx < source_->size(); ++idx) {
            Vec3d ps = init_pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                       + init_pose.block<3, 1>(0, 3);
            Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
            pointswise_error += (gt_ps - ps).norm();
        }
        pointswise_error /= source_->size();
        spdlog::info("pointswise_error: {}", pointswise_error);
        outfile << pointswise_error << "\n";

        double final_rms = std::numeric_limits<double>::max();
        Mat4d pose = init_pose;
        for (int iter = 0; iter < options_.max_iteration_; ++iter) {
            // 对点的索引，预先生成
            std::vector<int> index(source_->points.size());
            for (int i = 0; i < index.size(); ++i) {
                index[i] = i;
            }

            std::vector<MatchPair> matches(source_->size());
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
                auto p = ToVec3d(source_->points[idx]);
                Vec3d ps = pose.block<3, 3>(0, 0) * p + pose.block<3, 1>(0, 3);
                std::vector<int> nn; // 存储查询近邻点索引
                std::vector<float> dist; // 存储近邻点对应距离的平方
                kdtree_->nearestKSearch(ToPointType(ps), 1, nn, dist); // 这里取1个最近邻
                Vec3d q = ToVec3d(target_->points[nn[0]]);
                double dis2 = (q - ps).norm();
                // 点对
                matches[idx].p = ps;
                matches[idx].n1 =
                        pose.block<3, 3>(0, 0)
                        * source_normals->points[idx].getNormalVector3fMap().cast<double>();
                matches[idx].q = q;
                matches[idx].n2 = target_normals->points[nn[0]].getNormalVector3fMap().cast<double>();
                matches[idx].dis2 = dis2;
            });

            // 均值计算
            double rms = 0;
            double totalweight = 0;
            Vec3d centerP = Vec3d::Zero();
            Vec3d centerQ = Vec3d::Zero();
            int effnum = options_.overlap_ * matches.size();
            std::sort(matches.begin(), matches.end());
            for (int idx = 0; idx < effnum; ++idx) {
                // matches[idx].weight = 1;
                // matches[idx].weight = 1 - (matches[idx].dis2 / matches[effnum - 1].dis2);
                // matches[idx].weight = matches[idx].n1.dot(matches[idx].n2);
                matches[idx].weight = 10 * matches[idx].dis2 / matches[effnum - 1].dis2;
                centerP += matches[idx].weight * matches[idx].p; // 变换后的源点累加
                centerQ += matches[idx].weight * matches[idx].q; // 目标点累加
                rms += matches[idx].weight * matches[idx].dis2;
                totalweight += matches[idx].weight;
            }

            rms /= totalweight;
            rms = sqrt(rms);
            centerP /= totalweight;
            centerQ /= totalweight;

            // 构建协方差矩阵
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (int idx = 0; idx < effnum; ++idx) {
                Vec3d p_prime = matches[idx].p - centerP;
                Vec3d q_prime = matches[idx].q - centerQ;
                H += matches[idx].weight * p_prime * q_prime.transpose();
            }

            // SVD分解
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            // 计算旋转矩阵和平移向量
            Mat3d R = V * U.transpose();
            if (R.determinant() < 0) {
                R = -R;
            }
            Vec3d t = centerQ - R * centerP;

            // 更新姿态
            pose.block<3, 1>(0, 3) = R * pose.block<3, 1>(0, 3) + t;
            pose.block<3, 3>(0, 0) = R * pose.block<3, 3>(0, 0);
            spdlog::info("iter: {}, rms: {}", iter, rms);
            // if (fabs(rms - final_rms) < options_.rms_) {
            //     spdlog::info("converged.");
            //     break;
            // }
            final_rms = rms;

            // 计算逐点误差
            double pointswise_error = 0;
            for (int idx = 0; idx < source_->size(); ++idx) {
                Vec3d ps = pose.block<3, 3>(0, 0) * ToVec3d(source_->points[idx])
                           + pose.block<3, 1>(0, 3);
                Vec3d gt_ps = ToVec3d(gt_source_->points[idx]);
                pointswise_error += (gt_ps - ps).norm();
            }
            pointswise_error /= source_->size();
            spdlog::info("pointswise_error: {}", pointswise_error);
            outfile << pointswise_error << "\n";
        }
        outfile.close();

        std::cout << pose << std::endl;
        init_pose = pose;
        return true;
    }
} // namespace Icp
