#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class GradientUtils
 * @brief (Prompt 6.1) ����ݶȼ��㷽���Ĺ�����չ�ࡣ
 */
class GradientUtils {
public:
    /**
     * @brief Ӧ�� Sobel ���� (���� 3.3.1 ��ѡ��ķ���)��
     * @param input Ԥ�����ĻҶ�ͼ��
     * @param dx ������x�ס�
     * @param dy ������y�ס�
     * @param ksize Sobel�˵Ĵ�С��
     * @return cv::Mat �ݶ�ͼ (16λ�з��ţ�CV_16S���Ա������)��
     */
    static cv::Mat applySobel(const cv::Mat& input, int dx, int dy, int ksize = 3);

    // (����չ) �ڴ���������ݶ�����
    // static cv::Mat applyPrewitt(const cv::Mat& input, int dx, int dy);
    // static cv::Mat applyRoberts(const cv::Mat& input);
};

/**
 * @class Utilities
 * @brief ��Ÿ��ּ��㷽���Ĺ����ࡣ
 */
class Utilities {
public:
    /**
     * @brief ���� RMS �ݶ� (���� 3.3.2 ��)��
     * @param gradImg �ݶ�ͼ (�������� Sobel)��
     * @param direction 0 ��ʾ���� X ���� (����)��1 ��ʾ���� Y ���� (����)��
     * @return std::vector<double> ���� RMS �ݶ�ֵ��һά������
     */
    static std::vector<double> calculateRMSGradient(const cv::Mat& gradImg, int direction);

    /**
     * @brief ���� RMS �Ҷ� (����������ģ�����, ��ͼ 4.2)��
     * @param grayImg �Ҷ�ͼ��
     * @param direction 0 ��ʾ���� X ���� (����)��1 ��ʾ���� Y ���� (����)��
     * @return std::vector<double> ���� RMS �Ҷ�ֵ��һά������
     */
    static std::vector<double> calculateRMSGray(const cv::Mat& grayImg, int direction);

    /**
     * @brief ����˹Ƥ�������ϵ�� (Prompt 6.3, ���� 3.3.3 ��)��
     * @param v1 ����1��
     * @param v2 ����2 (������ v1 ��С��ͬ)��
     * @return double ���ϵ��, ��Χ [-1.0, 1.0]��
     */
    static double calculateSpearman(const std::vector<double>& v1, const std::vector<double>& v2);

    /**
     * @brief ���������еķ�ֵ���ֲ����ֵ����
     * @param data ����������
     * @param minPeakDistance ��ֵ֮�����С���롣
     * @return std::vector<int> ��ֵλ�õ�������
     */
    static std::vector<int> findPeaks(const std::vector<double>& data, int minPeakDistance = 10);
};
