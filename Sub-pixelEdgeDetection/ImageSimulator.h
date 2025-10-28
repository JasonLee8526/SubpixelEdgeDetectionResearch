#pragma once

#include <opencv2/opencv.hpp>
#include <string>

/**
 * @class ImageSimulator
 * @brief (Prompt 4) ģ�����������еı�׼��Բ�׿̱��ͼ��
 *
 * ������������x, y�����΢��(um)����
 * ����һ������ģ����Ե��8λ�Ҷ�ͼ��
 */
class ImageSimulator {
public:
    /**
     * @brief ���캯������ʼ��ת�����ӡ�
     * @param imgWidth ͼ���� (������766���ض�Ӧ50um)��
     * @param physicalWidth_um ͼ������������ (um)��
     */
    ImageSimulator(int imgWidth = 766, double physicalWidth_um = 50.0);

    /**
     * @brief ����һ��ģ����׿̱��ͼ��
     *
     * (Prompt 4)
     * 1. ������ɫ���� (800x600)��
     * 2. ���ƻ�ɫ���
     * 3. �����������ڿ򣬲���������������ƫ�ơ�
     * 4. ������ͼ��Ӧ�ø�˹ģ����ģ��������Ե��
     *
     * @param width ͼ���� (����)��
     * @param height ͼ��߶� (����)��
     * @param errorX_um X������׿���� (΢��)��
     * @param errorY_um Y������׿���� (΢��)��
     * @return cv::Mat 8λ�Ҷ�ͼ�� (CV_8UC1)��
     */
    cv::Mat generateStandardWaferImage(int width, int height, double errorX_um, double errorY_um);

    // ����ת������
    double PIX_TO_UM_FACTOR; ///< ���ص�΢�׵�ת������
    double UM_TO_PIX_FACTOR; ///< ΢�׵����ص�ת������
};
