#include "ImageUtils.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

// --- FilterUtils 实现 ---

cv::Mat FilterUtils::applyMedianFilter(const cv::Mat& input, int kernelSize) {
    CV_Assert(kernelSize % 2 == 1); // 核必须是奇数
    cv::Mat output;
    cv::medianBlur(input, output, kernelSize);
    return output;
}

// --- EnhancementUtils 实现 ---

cv::Mat EnhancementUtils::applyHistogramEqualization(const cv::Mat& input) {
    CV_Assert(input.type() == CV_8UC1); // 必须是8位灰度图
    cv::Mat output;
    cv::equalizeHist(input, output);
    return output;
}

// --- ImagePreprocessor 实现 ---

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
    // 1. 灰度空间变换 (论文 3.2.1)
    cv::Mat grayImg = ensureGrayscale(inputImage);

    // 2. 滤波降噪 (论文 3.2.2) - 论文选择了中值滤波
    cv::Mat filteredImg = FilterUtils::applyMedianFilter(grayImg, 3);

    // 3. 图像增强 (论文 3.2.3) - 论文提到了直方图均衡化
    cv::Mat enhancedImg = EnhancementUtils::applyHistogramEqualization(filteredImg);

    // std::cout << "[Preprocessor] 图像预处理完成。" << std::endl;
    return enhancedImg;
}

