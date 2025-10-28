#pragma once

#include <opencv2/opencv.hpp>
#include <string>

/**
 * @class ImageSimulator
 * @brief (Prompt 4) 模拟生成论文中的标准晶圆套刻标记图像。
 *
 * 该类根据输入的x, y方向的微米(um)级误差，
 * 生成一个带有模糊边缘的8位灰度图像。
 */
class ImageSimulator {
public:
    /**
     * @brief 构造函数，初始化转换因子。
     * @param imgWidth 图像宽度 (论文中766像素对应50um)。
     * @param physicalWidth_um 图像代表的物理宽度 (um)。
     */
    ImageSimulator(int imgWidth = 766, double physicalWidth_um = 50.0);

    /**
     * @brief 生成一张模拟的套刻标记图像。
     *
     * (Prompt 4)
     * 1. 创建黑色背景 (800x600)。
     * 2. 绘制灰色外框。
     * 3. 绘制稍亮的内框，并根据误差参数进行偏移。
     * 4. 对整个图像应用高斯模糊以模拟非理想边缘。
     *
     * @param width 图像宽度 (像素)。
     * @param height 图像高度 (像素)。
     * @param errorX_um X方向的套刻误差 (微米)。
     * @param errorY_um Y方向的套刻误差 (微米)。
     * @return cv::Mat 8位灰度图像 (CV_8UC1)。
     */
    cv::Mat generateStandardWaferImage(int width, int height, double errorX_um, double errorY_um);

    // 公共转换因子
    double PIX_TO_UM_FACTOR; ///< 像素到微米的转换因子
    double UM_TO_PIX_FACTOR; ///< 微米到像素的转换因子
};
