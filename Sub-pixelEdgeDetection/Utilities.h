#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class GradientUtils
 * @brief (Prompt 6.1) 存放梯度计算方法的公共扩展类。
 */
class GradientUtils {
public:
    /**
     * @brief 应用 Sobel 算子 (论文 3.3.1 节选择的方法)。
     * @param input 预处理后的灰度图像。
     * @param dx 导数的x阶。
     * @param dy 导数的y阶。
     * @param ksize Sobel核的大小。
     * @return cv::Mat 梯度图 (16位有符号，CV_16S，以避免溢出)。
     */
    static cv::Mat applySobel(const cv::Mat& input, int dx, int dy, int ksize = 3);

    // (可扩展) 在此添加其他梯度算子
    // static cv::Mat applyPrewitt(const cv::Mat& input, int dx, int dy);
    // static cv::Mat applyRoberts(const cv::Mat& input);
};

/**
 * @class Utilities
 * @brief 存放各种计算方法的公共类。
 */
class Utilities {
public:
    /**
     * @brief 计算 RMS 梯度 (论文 3.3.2 节)。
     * @param gradImg 梯度图 (例如来自 Sobel)。
     * @param direction 0 表示计算 X 方向 (逐列)，1 表示计算 Y 方向 (逐行)。
     * @return std::vector<double> 包含 RMS 梯度值的一维向量。
     */
    static std::vector<double> calculateRMSGradient(const cv::Mat& gradImg, int direction);

    /**
     * @brief 计算 RMS 灰度 (用于亚像素模型拟合, 如图 4.2)。
     * @param grayImg 灰度图。
     * @param direction 0 表示计算 X 方向 (逐列)，1 表示计算 Y 方向 (逐行)。
     * @return std::vector<double> 包含 RMS 灰度值的一维向量。
     */
    static std::vector<double> calculateRMSGray(const cv::Mat& grayImg, int direction);

    /**
     * @brief 计算斯皮尔曼相关系数 (Prompt 6.3, 论文 3.3.3 节)。
     * @param v1 向量1。
     * @param v2 向量2 (必须与 v1 大小相同)。
     * @return double 相关系数, 范围 [-1.0, 1.0]。
     */
    static double calculateSpearman(const std::vector<double>& v1, const std::vector<double>& v2);

    /**
     * @brief 查找向量中的峰值（局部最大值）。
     * @param data 输入向量。
     * @param minPeakDistance 峰值之间的最小距离。
     * @return std::vector<int> 峰值位置的索引。
     */
    static std::vector<int> findPeaks(const std::vector<double>& data, int minPeakDistance = 10);
};
