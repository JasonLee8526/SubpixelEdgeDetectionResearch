#include "ImageUtils.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

// --- FilterUtils ʵ�� ---

cv::Mat FilterUtils::applyMedianFilter(const cv::Mat& input, int kernelSize) {
    CV_Assert(kernelSize % 2 == 1); // �˱���������
    cv::Mat output;
    cv::medianBlur(input, output, kernelSize);
    return output;
}

// --- EnhancementUtils ʵ�� ---

cv::Mat EnhancementUtils::applyHistogramEqualization(const cv::Mat& input) {
    CV_Assert(input.type() == CV_8UC1); // ������8λ�Ҷ�ͼ
    cv::Mat output;
    cv::equalizeHist(input, output);
    return output;
}

// --- ImagePreprocessor ʵ�� ---

cv::Mat ImagePreprocessor::ensureGrayscale(const cv::Mat& input) {
    if (input.channels() == 3 || input.channels() == 4) {
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    else if (input.type() != CV_8UC1) {
        cv::Mat gray8;
        input.convertTo(gray8, CV_8UC1);
        return gray8;
    }
    return input.clone();
}

cv::Mat ImagePreprocessor::preprocess(const cv::Mat& inputImage) {
    // 1. �Ҷȿռ�任 (���� 3.2.1)
    cv::Mat grayImg = ensureGrayscale(inputImage);

    // 2. �˲����� (���� 3.2.2) - ����ѡ������ֵ�˲�
    cv::Mat filteredImg = FilterUtils::applyMedianFilter(grayImg, 3);

    // 3. ͼ����ǿ (���� 3.2.3) - �����ᵽ��ֱ��ͼ���⻯
    cv::Mat enhancedImg = EnhancementUtils::applyHistogramEqualization(filteredImg);

    // std::cout << "[Preprocessor] ͼ��Ԥ������ɡ�" << std::endl;
    return enhancedImg;
}

