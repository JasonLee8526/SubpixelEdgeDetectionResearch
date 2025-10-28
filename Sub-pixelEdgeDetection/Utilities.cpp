#include "Utilities.h"
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>

// --- GradientUtils 实现 ---

cv::Mat GradientUtils::applySobel(const cv::Mat& input, int dx, int dy, int ksize) {
    cv::Mat grad;
    // 使用 16 位有符号整数 (CV_16S) 来存储梯度，以防止 8 位溢出
    cv::Sobel(input, grad, CV_16S, dx, dy, ksize);
    return grad;
}

// --- Utilities 实现 ---

std::vector<double> Utilities::calculateRMSGradient(const cv::Mat& gradImg, int direction) {
    CV_Assert(gradImg.type() == CV_16S); // 期望 16 位有符号梯度图

    if (direction == 0) { // X 方向 (逐列)
        int cols = gradImg.cols;
        int rows = gradImg.rows;
        std::vector<double> rms_gradient_x(cols);

        for (int j = 0; j < cols; ++j) {
            double sum_sq = 0.0;
            for (int i = 0; i < rows; ++i) {
                double val = static_cast<double>(gradImg.at<short>(i, j));
                sum_sq += val * val;
            }
            rms_gradient_x[j] = std::sqrt(sum_sq / rows);
        }
        return rms_gradient_x;
    }
    else { // Y 方向 (逐行)
        int cols = gradImg.cols;
        int rows = gradImg.rows;
        std::vector<double> rms_gradient_y(rows);

        for (int i = 0; i < rows; ++i) {
            double sum_sq = 0.0;
            for (int j = 0; j < cols; ++j) {
                double val = static_cast<double>(gradImg.at<short>(i, j));
                sum_sq += val * val;
            }
            rms_gradient_y[i] = std::sqrt(sum_sq / cols);
        }
        return rms_gradient_y;
    }
}

std::vector<double> Utilities::calculateRMSGray(const cv::Mat& grayImg, int direction) {
    CV_Assert(grayImg.type() == CV_8UC1); // 期望 8 位灰度图

    if (direction == 0) { // X 方向 (逐列)
        int cols = grayImg.cols;
        int rows = grayImg.rows;
        std::vector<double> rms_gray_x(cols);

        for (int j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (int i = 0; i < rows; ++i) {
                sum += static_cast<double>(grayImg.at<uchar>(i, j));
            }
            rms_gray_x[j] = sum / rows; // 注意：论文中"RMS 灰度"似乎是指平均灰度
        }
        return rms_gray_x;
    }
    else { // Y 方向 (逐行)
        int cols = grayImg.cols;
        int rows = grayImg.rows;
        std::vector<double> rms_gray_y(rows);

        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols; ++j) {
                sum += static_cast<double>(grayImg.at<uchar>(i, j));
            }
            rms_gray_y[i] = sum / cols;
        }
        return rms_gray_y;
    }
}


// --- 斯皮尔曼相关性 (Prompt 6.3) ---

// 辅助结构体，用于排序和获取原始索引
struct RankedItem {
    double value;
    int index;
    int rank;
};

// 比较函数
bool compareByValue(const RankedItem& a, const RankedItem& b) {
    return a.value < b.value;
}

bool compareByIndex(const RankedItem& a, const RankedItem& b) {
    return a.index < b.index;
}

// 获取等级向量
std::vector<double> getRanks(const std::vector<double>& v) {
    std::vector<RankedItem> items(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        items[i] = { v[i], static_cast<int>(i), 0 };
    }

    // 按值排序
    std::sort(items.begin(), items.end(), compareByValue);

    // 分配等级
    for (size_t i = 0; i < items.size(); ++i) {
        items[i].rank = static_cast<int>(i) + 1;
    }

    // 按原始索引排序
    std::sort(items.begin(), items.end(), compareByIndex);

    std::vector<double> ranks(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        ranks[i] = static_cast<double>(items[i].rank);
    }
    return ranks;
}

double Utilities::calculateSpearman(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size() || v1.empty()) {
        return 0.0;
    }

    std::vector<double> ranks_v1 = getRanks(v1);
    std::vector<double> ranks_v2 = getRanks(v2);

    size_t n = v1.size();
    double sum_d_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double d = ranks_v1[i] - ranks_v2[i];
        sum_d_sq += d * d;
    }

    // 论文公式 3.23
    return 1.0 - (6.0 * sum_d_sq) / (static_cast<double>(n) * (n * n - 1.0));
}

std::vector<int> Utilities::findPeaks(const std::vector<double>& data, int minPeakDistance) {
    std::vector<int> peaks;
    if (data.empty()) return peaks;

    for (int i = 1; i < data.size() - 1; ++i) {
        if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
            // 检查与上一个峰值的距离
            if (peaks.empty() || (i - peaks.back()) >= minPeakDistance) {
                peaks.push_back(i);
            }
            else if (data[i] > data[peaks.back()]) {
                // 如果这个峰值更高，替换上一个太近的峰值
                peaks.pop_back();
                peaks.push_back(i);
            }
        }
    }
    return peaks;
}
