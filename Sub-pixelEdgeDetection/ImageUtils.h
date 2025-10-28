#pragma once

#include <opencv2/opencv.hpp>

/**
 * @class FilterUtils
 * @brief (Prompt 5) ����˲����뷽���Ĺ�����չ�ࡣ
 */
class FilterUtils {
public:
    /**
     * @brief Ӧ����ֵ�˲� (����3.2.2��ѡ��ķ���)��
     * @param input ����ͼ��
     * @param kernelSize �˲����˴�С (����������)��
     * @return cv::Mat �˲����ͼ��
     */
    static cv::Mat applyMedianFilter(const cv::Mat& input, int kernelSize = 3);
};

/**
 * @class EnhancementUtils
 * @brief (Prompt 5) ���ͼ����ǿ�����Ĺ�����չ�ࡣ
 */
class EnhancementUtils {
public:
    /**
     * @brief Ӧ��ֱ��ͼ���⻯ (����3.2.3���ᵽ�ķ���)��
     * @param input ����ͼ�� (������8λ�Ҷ�ͼ)��
     * @return cv::Mat ��ǿ���ͼ��
     */
    static cv::Mat applyHistogramEqualization(const cv::Mat& input);
};

/**
 * @class ImagePreprocessor
 * @brief (Prompt 5) ͼ��Ԥ�������ࡣ
 *
 * ������ø��ֹ�������ִ��������Ԥ�������̡�
 */
class ImagePreprocessor {
public:
    /**
     * @brief ִ��������Ԥ�������� (����3.2��)��
     * @param inputImage ԭʼ����ͼ�� (�����ǲ�ɫ��Ҷ�)��
     * @return cv::Mat Ԥ������ɵ�8λ�Ҷ�ͼ��
     */
    cv::Mat preprocess(const cv::Mat& inputImage);

private:
    /**
     * @brief ȷ��ͼ����8λ�Ҷ�ͼ (����3.2.1��)��
     * @param input ����ͼ��
     * @return cv::Mat 8λ�Ҷ�ͼ��
     */
    cv::Mat ensureGrayscale(const cv::Mat& input);
};

