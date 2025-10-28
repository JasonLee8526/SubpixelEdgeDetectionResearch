#include "ImageSimulator.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

ImageSimulator::ImageSimulator(int imgWidth, double physicalWidth_um) {
    // (Prompt 4.2) 根据论文中的尺寸（766 像素 = 50 µm）计算转换因子
    UM_TO_PIX_FACTOR = static_cast<double>(imgWidth) / physicalWidth_um;
    PIX_TO_UM_FACTOR = physicalWidth_um / static_cast<double>(imgWidth);

    std::cout << "[ImageSimulator] 初始化: " << UM_TO_PIX_FACTOR
        << " 像素/微米, " << PIX_TO_UM_FACTOR
        << " 微米/像素" << std::endl;
}

cv::Mat ImageSimulator::generateStandardWaferImage(int width, int height, double errorX_um, double errorY_um) {
    // (Prompt 4.1) 创建一个黑色背景的8位灰度图像
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);

    // 定义框的尺寸 (像素) - 这些是示例值
    int outerBoxSize = 400;
    int innerBoxSize = 200;
    int boxThickness = 20; // 框的厚度

    // 定义灰度值
    uchar outerBoxGray = 100; // (Prompt 4.3) 灰色外框
    uchar innerBoxGray = 200; // (Prompt 4.3) 稍亮的内框

    // (Prompt 4.4) 计算误差偏移（像素）
    int errorX_px = static_cast<int>(std::round(errorX_um * UM_TO_PIX_FACTOR));
    int errorY_px = static_cast<int>(std::round(errorY_um * UM_TO_PIX_FACTOR));

    // 计算中心点
    cv::Point center(width / 2, height / 2);

    // (Prompt 4.3) 绘制外框 (前一层)
    cv::Point outerTopLeft(center.x - outerBoxSize / 2, center.y - outerBoxSize / 2);
    cv::Point outerBottomRight(center.x + outerBoxSize / 2, center.y + outerBoxSize / 2);
    cv::rectangle(img, outerTopLeft, outerBottomRight, cv::Scalar(outerBoxGray), boxThickness);

    // (Prompt 4.4) 绘制偏移后的内框 (当前层)
    cv::Point innerTopLeft(center.x - innerBoxSize / 2 + errorX_px, center.y - innerBoxSize / 2 + errorY_px);
    cv::Point innerBottomRight(center.x + innerBoxSize / 2 + errorX_px, center.y + innerBoxSize / 2 + errorY_px);
    cv::rectangle(img, innerTopLeft, innerBottomRight, cv::Scalar(innerBoxGray), boxThickness);

    // (Prompt 4.5) 重要：使用高斯模糊模拟论文中的模糊边缘
    // 这是为了模拟图4.9所示的非理想S型边缘
    cv::Mat blurredImg;
    // 使用一个 5x5 的核。这个值可以调整以匹配论文中的模糊程度
    cv::GaussianBlur(img, blurredImg, cv::Size(5, 5), 0);

    return blurredImg;
}
