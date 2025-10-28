#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "SubPixelModel.h"

/**
 * @struct CoarseEdges
 * @brief �洢8���ߵ����ؼ��ֶ�λ�����
 */
struct CoarseEdges {
    int x1, x2, x3, x4;
    int y1, y2, y3, y4;
};

/**
 * @struct FineEdges
 * @brief �洢8���ߵ������ؼ�����λ�����
 */
struct FineEdges {
    double x1, x2, x3, x4;
    double y1, y2, y3, y4;
};

/**
 * @class Localization
 * @brief (Prompt 6, 7) ����ֶ�λ�;���λ�����ࡣ
 */
class Localization {
public:
    Localization() : m_templateWindow(15), m_subPixelWindow(20) {}

    /**
     * @brief (Prompt 6.2) �ӱ�׼ͼ�񴴽� RMS �ݶ�ģ�塣
     * @param standardImage һ�� (0,0) ���ı�׼ͼ��
     * @param preprocessor ͼ��Ԥ������ʵ����
     * @return bool �Ƿ�ɹ�����ģ�塣
     */
    bool createTemplate(const cv::Mat& standardImage, class ImagePreprocessor& preprocessor);

    /**
     * @brief (Prompt 6) ִ�����ؼ��ֶ�λ��
     * @param processedImg Ԥ�����Ĵ���ͼ��
     * @return CoarseEdges 8���ߵ����ؼ�λ�á�
     */
    CoarseEdges coarseLocalization(const cv::Mat& processedImg);

    /**
     * @brief (Prompt 7) ִ�������ؼ�����λ��
     * @param originalImg *δ*��ǿ��ԭʼ�Ҷ�ͼ (���ڻҶ�ģ�����)��
     * @param gradImgX X�����ݶ�ͼ (�����ݶ�ģ�����)��
     * @param gradImgY Y�����ݶ�ͼ��
     * @param coarseEdges �ֶ�λ�����
     * @param modelType Ҫʹ�õ�ģ�� (0=Sigmoid, 1=Quadratic, ...)��
     * @return FineEdges 8���ߵ������ؼ�λ�á�
     */
    FineEdges fineLocalization(
        const cv::Mat& originalImg,
        const cv::Mat& gradImgX,
        const cv::Mat& gradImgY,
        const CoarseEdges& coarseEdges,
        int modelType);

    /**
     * @brief (Prompt 7.2) ��ȡ����������ϡ�
     * @param rmsData RMS ���� (�ҶȻ��ݶ�)��
     * @param center �ֶ�λ�����ĵ㡣
     * @return std::vector<double> 41�����ݵ㡣
     */
    std::vector<double> extractData(const std::vector<double>& rmsData, int center);


private:
    std::vector<double> m_templateRMS_X; ///< X�����ģ��
    std::vector<double> m_templateRMS_Y; ///< Y�����ģ��
    int m_templateWindow; ///< (���� 3.3.3) ģ���ȡ����
    int m_subPixelWindow; ///< (���� 4.1) �����ؽ�ȡ���� (20)

    /**
     * @brief (Prompt 6.3, 6.4) �������ģ��ƥ��λ�úͷ�ֵ��
     * @param rmsData ����ͼ��� RMS �ݶȡ�
     * @param templateData ģ�� RMS �ݶȡ�
     * @return std::vector<int> 4����ֵ������λ�á�
     */
    std::vector<int> findTemplateMatchingPeaks(const std::vector<double>& rmsData, const std::vector<double>& templateData);
};
