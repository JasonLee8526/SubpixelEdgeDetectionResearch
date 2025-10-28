#include "Utilities.h"
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>

// --- GradientUtils ʵ�� ---

cv::Mat GradientUtils::applySobel(const cv::Mat& input, int dx, int dy, int ksize) {
    cv::Mat grad;
    // ʹ�� 16 λ�з������� (CV_16S) ���洢�ݶȣ��Է�ֹ 8 λ���
    cv::Sobel(input, grad, CV_16S, dx, dy, ksize);
    return grad;
}

// --- Utilities ʵ�� ---

std::vector<double> Utilities::calculateRMSGradient(const cv::Mat& gradImg, int direction) {
    CV_Assert(gradImg.type() == CV_16S); // ���� 16 λ�з����ݶ�ͼ

    if (direction == 0) { // X ���� (����)
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
    else { // Y ���� (����)
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
    CV_Assert(grayImg.type() == CV_8UC1); // ���� 8 λ�Ҷ�ͼ

    if (direction == 0) { // X ���� (����)
        int cols = grayImg.cols;
        int rows = grayImg.rows;
        std::vector<double> rms_gray_x(cols);

        for (int j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (int i = 0; i < rows; ++i) {
                sum += static_cast<double>(grayImg.at<uchar>(i, j));
            }
            rms_gray_x[j] = sum / rows; // ע�⣺������"RMS �Ҷ�"�ƺ���ָƽ���Ҷ�
        }
        return rms_gray_x;
    }
    else { // Y ���� (����)
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


// --- ˹Ƥ��������� (Prompt 6.3) ---

// �����ṹ�壬��������ͻ�ȡԭʼ����
struct RankedItem {
    double value;
    int index;
    int rank;
};

// �ȽϺ���
bool compareByValue(const RankedItem& a, const RankedItem& b) {
    return a.value < b.value;
}

bool compareByIndex(const RankedItem& a, const RankedItem& b) {
    return a.index < b.index;
}

// ��ȡ�ȼ�����
std::vector<double> getRanks(const std::vector<double>& v) {
    std::vector<RankedItem> items(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        items[i] = { v[i], static_cast<int>(i), 0 };
    }

    // ��ֵ����
    std::sort(items.begin(), items.end(), compareByValue);

    // ����ȼ�
    for (size_t i = 0; i < items.size(); ++i) {
        items[i].rank = static_cast<int>(i) + 1;
    }

    // ��ԭʼ��������
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

    // ���Ĺ�ʽ 3.23
    return 1.0 - (6.0 * sum_d_sq) / (static_cast<double>(n) * (n * n - 1.0));
}

std::vector<int> Utilities::findPeaks(const std::vector<double>& data, int minPeakDistance) {
    std::vector<int> peaks;
    if (data.empty()) return peaks;

    for (int i = 1; i < data.size() - 1; ++i) {
        if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
            // �������һ����ֵ�ľ���
            if (peaks.empty() || (i - peaks.back()) >= minPeakDistance) {
                peaks.push_back(i);
            }
            else if (data[i] > data[peaks.back()]) {
                // ��������ֵ���ߣ��滻��һ��̫���ķ�ֵ
                peaks.pop_back();
                peaks.push_back(i);
            }
        }
    }
    return peaks;
}
