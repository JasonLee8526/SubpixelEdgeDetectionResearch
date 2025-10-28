#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "SubPixelModel.h"

/**
 * @struct CoarseEdges
 * @brief 存储8条边的像素级粗定位结果。
 */
struct CoarseEdges {
    int x1, x2, x3, x4;
    int y1, y2, y3, y4;
};

/**
 * @struct FineEdges
 * @brief 存储8条边的亚像素级精定位结果。
 */
struct FineEdges {
    double x1, x2, x3, x4;
    double y1, y2, y3, y4;
};

/**
 * @class Localization
 * @brief (Prompt 6, 7) 负责粗定位和精定位的主类。
 */
class Localization {
public:
    Localization() : m_templateWindow(15), m_subPixelWindow(20) {}

    /**
     * @brief (Prompt 6.2) 从标准图像创建 RMS 梯度模板。
     * @param standardImage 一张 (0,0) 误差的标准图像。
     * @param preprocessor 图像预处理器实例。
     * @return bool 是否成功创建模板。
     */
    bool createTemplate(const cv::Mat& standardImage, class ImagePreprocessor& preprocessor);

    /**
     * @brief (Prompt 6) 执行像素级粗定位。
     * @param processedImg 预处理后的待测图像。
     * @return CoarseEdges 8条边的像素级位置。
     */
    CoarseEdges coarseLocalization(const cv::Mat& processedImg);

    /**
     * @brief (Prompt 7) 执行亚像素级精定位。
     * @param originalImg *未*增强的原始灰度图 (用于灰度模型拟合)。
     * @param gradImgX X方向梯度图 (用于梯度模型拟合)。
     * @param gradImgY Y方向梯度图。
     * @param coarseEdges 粗定位结果。
     * @param modelType 要使用的模型 (0=Sigmoid, 1=Quadratic, ...)。
     * @return FineEdges 8条边的亚像素级位置。
     */
    FineEdges fineLocalization(
        const cv::Mat& originalImg,
        const cv::Mat& gradImgX,
        const cv::Mat& gradImgY,
        const CoarseEdges& coarseEdges,
        int modelType);

    /**
     * @brief (Prompt 7.2) 提取数据用于拟合。
     * @param rmsData RMS 向量 (灰度或梯度)。
     * @param center 粗定位的中心点。
     * @return std::vector<double> 41个数据点。
     */
    std::vector<double> extractData(const std::vector<double>& rmsData, int center);


private:
    std::vector<double> m_templateRMS_X; ///< X方向的模板
    std::vector<double> m_templateRMS_Y; ///< Y方向的模板
    int m_templateWindow; ///< (论文 3.3.3) 模板截取窗口
    int m_subPixelWindow; ///< (论文 4.1) 亚像素截取窗口 (20)

    /**
     * @brief (Prompt 6.3, 6.4) 查找最佳模板匹配位置和峰值。
     * @param rmsData 待测图像的 RMS 梯度。
     * @param templateData 模板 RMS 梯度。
     * @return std::vector<int> 4个峰值的像素位置。
     */
    std::vector<int> findTemplateMatchingPeaks(const std::vector<double>& rmsData, const std::vector<double>& templateData);
};
