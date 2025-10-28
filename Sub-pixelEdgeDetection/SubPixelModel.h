#pragma once

#include <vector>

/**
 * @struct FitResult
 * @brief (Prompt 7) 存储亚像素拟合的结果。
 */
struct FitResult {
    double edge_position; ///< 亚像素边缘位置
    double r_squared;     ///< R-Squared (R²) 拟合优度
    bool success;         ///< 拟合是否成功

    FitResult() : edge_position(0.0), r_squared(0.0), success(false) {}
};

/**
 * @class SubPixelModel
 * @brief (Prompt 7 & 9) 存放所有亚像素模型的类。
 *
 * 包含论文第4章对比的6种模型：
 * 1. 灰度矩模型 (Gray Moment)
 * 2. 空间矩模型 (Spatial Moment)
 * 3. Sigmoid 模型
 * 4. 反正切模型 (Arctan)
 * 5. 二次多项式模型 (Quadratic)
 * 6. 高斯模型 (Gaussian)
 *
 * 所有模型的方法都是并行的，可以相互替换。
 */
class SubPixelModel {
public:
    /**
     * @brief 计算 R-Squared (R²) 拟合优度 (论文 4.1 节, 公式 4.1)。
     * @param y_data 原始数据点。
     * @param y_fit 拟合后的模型数据点。
     * @return double R-Squared 值。
     */
    static double calculateRSquared(const std::vector<double>& y_data, const std::vector<double>& y_fit);

    // --- 灰度模型 (论文 4.2 节) ---

    /**
     * @brief 1. 灰度矩模型 (论文 4.2.1 节, 公式 4.8)。
     * @param grayData 截取的41个灰度数据点。
     * @return FitResult 拟合结果 (直接计算，非拟合)。
     */
    static FitResult calcGrayMoment(const std::vector<double>& grayData);

    /**
     * @brief 2. 空间矩模型 (论文 4.2.1 节, 公式 4.14)。
     * @param grayData 截取的41个灰度数据点。
     * @return FitResult 拟合结果 (直接计算，非拟合)。
     */
    static FitResult calcSpatialMoment(const std::vector<double>& grayData);

    /**
     * @brief 3. Sigmoid 模型 (论文 4.2.2 节, 公式 4.15)。
     * (Prompt 7.3) 使用 Eigen::LevenbergMarquardt 进行非线性拟合。
     * @param grayData 截取的41个灰度数据点。
     * @return FitResult 拟合结果，edge_position 为参数 'c'。
     */
    static FitResult fitSigmoid(const std::vector<double>& grayData);

    /**
     * @brief 4. 反正切模型 (论文 4.2.2 节, 公式 4.16)。
     * (Prompt 7.1) 使用 Eigen::LevenbergMarquardt 进行非线性拟合。
     * @param grayData 截取的41个灰度数据点。
     * @return FitResult 拟合结果，edge_position 为参数 '-c/b'。
     */
    static FitResult fitArctan(const std::vector<double>& grayData);

    // --- 梯度模型 (论文 4.3 节) ---

    /**
     * @brief 5. 二次多项式模型 (论文 4.3 节, 公式 4.17)。
     * @param gradData 截取的41个梯度数据点。
     * @return FitResult 拟合结果，edge_position 为参数 '-b/2a'。
     */
    static FitResult fitQuadratic(const std::vector<double>& gradData);

    /**
     * @brief 6. 高斯模型 (论文 4.3 节, 公式 4.18)。
     * (Prompt 7.1) 通过对数变换 (公式 4.23) 转换为二次多项式拟合。
     * @param gradData 截取的41个梯度数据点。
     * @return FitResult 拟合结果，edge_position 为参数 'b'。
     */
    static FitResult fitGaussian(const std::vector<double>& gradData);
};
