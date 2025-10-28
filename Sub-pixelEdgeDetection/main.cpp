#include <iostream>
#include <vector>
#include <string>
#include <iomanip> // 用于 std::setw, std::setprecision
#include <numeric>

#include "ImageSimulator.h"
#include "ImageUtils.h" 
#include "Localization.h"
#include "Utilities.h"

/**
 * @struct ExperimentResult
 * @brief 存储一组实验的结果（平均误差和方差）。
 */
struct ExperimentResult {
    double mean_error_x = 0.0;
    double mean_error_y = 0.0;
    double variance_x = 0.0;
    double variance_y = 0.0;
};

/**
 * @brief (Prompt 8) 计算一组误差数据的平均值和方差。
 * @param errors 误差数据向量。
 * @param out_mean [out] 平均值。
 * @param out_variance [out] 方差。
 */
void calculateStatistics(const std::vector<double>& errors, double& out_mean, double& out_variance) {
    if (errors.empty()) {
        out_mean = 0.0;
        out_variance = 0.0;
        return;
    }

    double sum = std::accumulate(errors.begin(), errors.end(), 0.0);
    out_mean = sum / errors.size();

    double sq_sum = 0.0;
    for (double err : errors) {
        sq_sum += (err - out_mean) * (err - out_mean);
    }
    // 使用 n-1 作为分母（样本方差），如果需要总体方差则用 errors.size()
    if (errors.size() > 1) {
        out_variance = sq_sum / (errors.size() - 1);
    }
    else {
        out_variance = 0.0; // 无法计算方差
    }
}

/**
 * @brief (Prompt 8) 运行单次测量流程。
 *
 * 图像生成 -> 预处理 -> 粗定位 -> 精定位 -> 计算误差
 */
bool runSingleMeasurement(
    ImageSimulator& simulator,
    ImagePreprocessor& preprocessor,
    Localization& localizer,
    double true_dx_um,
    double true_dy_um,
    int modelType,
    double& out_error_x,
    double& out_error_y)
{
    // 1. 生成图像 (Prompt 8.1)
    // 论文使用 766x576，我们使用 800x600 保持一致性
    cv::Mat testImg = simulator.generateStandardWaferImage(800, 600, true_dx_um, true_dy_um);

    // 准备用于拟合的原始灰度图 (无增强)
    // (论文 4.1 节指出拟合是在预处理后的数据上，但灰度模型应在未增强的数据上拟合效果更好)
    // 我们遵循论文 3.2 节的流程，使用预处理后的数据
    cv::Mat processedImg = preprocessor.preprocess(testImg);

    // 准备用于灰度模型拟合的"原始"灰度图 (仅滤波，不增强)
    cv::Mat grayImg;
    if (testImg.channels() != 1) cv::cvtColor(testImg, grayImg, cv::COLOR_BGR2GRAY);
    else grayImg = testImg.clone();
    cv::medianBlur(grayImg, grayImg, 3); // 仅做中值滤波


    // 准备梯度图
    cv::Mat gradX = GradientUtils::applySobel(processedImg, 1, 0);
    cv::Mat gradY = GradientUtils::applySobel(processedImg, 0, 1);

    // 3. 粗定位 (Prompt 8.1)
    CoarseEdges coarseEdges = localizer.coarseLocalization(processedImg);
    if (coarseEdges.x1 == -1 || coarseEdges.y1 == -1) {
        std::cerr << "粗定位失败！" << std::endl;
        return false;
    }

    // 4. 精定位 (Prompt 8.1)
    // 注意：传递 grayImg 用于灰度模型, 传递 gradX/Y 用于梯度模型
    FineEdges fineEdges = localizer.fineLocalization(grayImg, gradX, gradY, coarseEdges, modelType);

    // 5. 得到结果 (Prompt 8.1)
    // (论文 6.3.1 节, 公式 6.1)
    // dx = ( (x1+x4) - (x2+x3) ) / 2
    double measured_dx_px = ((fineEdges.x1 + fineEdges.x4) - (fineEdges.x2 + fineEdges.x3)) / 2.0;
    double measured_dy_px = ((fineEdges.y1 + fineEdges.y4) - (fineEdges.y2 + fineEdges.y3)) / 2.0;

    double measured_dx_um = measured_dx_px * simulator.PIX_TO_UM_FACTOR;
    double measured_dy_um = measured_dy_px * simulator.PIX_TO_UM_FACTOR;

    // 计算误差
    out_error_x = measured_dx_um - true_dx_um;
    out_error_y = measured_dy_um - true_dy_um;

    return true;
}


int main() {
    std::cout << "--- 复现论文第6章实验 (Prompt 8) ---" << std::endl;

    // (Prompt 8.1) 初始化所有类
    ImageSimulator simulator(766, 50.0); // (论文 6.1 节, 766px = 50um)
    ImagePreprocessor preprocessor;
    Localization localizer;

    // 创建 (0,0) 模板
    std::cout << "正在创建 (0,0) 模板..." << std::endl;
    cv::Mat templateImg = simulator.generateStandardWaferImage(800, 600, 0.0, 0.0);
    if (!localizer.createTemplate(templateImg, preprocessor)) {
        std::cerr << "无法启动实验：模板创建失败。" << std::endl;
        return -1;
    }
    std::cout << "模板创建完毕，开始模拟实验..." << std::endl;

    // (Prompt 8) 定义实验组 (来自论文 6.1 节)
    std::vector<std::pair<double, double>> errorGroups = {
        {0.0, 0.0},
        {-0.5, -1.0},
        {-1.0, 0.0},
        {-1.0, 0.1},
        {-1.0, -0.1},
        {-1.0, 0.5},
        {-1.0, 1.0},
        {-1.0, -1.0}
    };

    int runsPerGroup = 5; // (论文 6.1 节, 每组5张图像)
    int modelToTest = 0;  // 0 = Sigmoid (论文首选, 表 6.5)

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n--- 正在复现 Sigmoid 模型 (表 6.5 和 6.9) ---" << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
    std::cout << std::setw(15) << "标准误差 (um)"
        << std::setw(18) << "平均误差 X (um)"
        << std::setw(18) << "平均误差 Y (um)"
        << std::setw(18) << "方差 X (10e-5)"
        << std::setw(18) << "方差 Y (10e-5)"
        << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;

    std::vector<double> total_errors_x;
    std::vector<double> total_errors_y;

    for (const auto& group : errorGroups) {
        double true_dx = group.first;
        double true_dy = group.second;

        std::vector<double> group_errors_x;
        std::vector<double> group_errors_y;

        for (int i = 0; i < runsPerGroup; ++i) {
            double error_x, error_y;
            if (runSingleMeasurement(simulator, preprocessor, localizer, true_dx, true_dy, modelToTest, error_x, error_y)) {
                group_errors_x.push_back(error_x);
                group_errors_y.push_back(error_y);
                total_errors_x.push_back(std::abs(error_x)); // 记录绝对误差以计算总平均误差
                total_errors_y.push_back(std::abs(error_y)); // 记录绝对误差以计算总平均误差
            }
        }

        double mean_x, var_x, mean_y, var_y;
        calculateStatistics(group_errors_x, mean_x, var_x);
        calculateStatistics(group_errors_y, mean_y, var_y);

        std::string group_str = "(" + std::to_string(true_dx) + ", " + std::to_string(true_dy) + ")";
        std::cout << std::setw(15) << group_str
            << std::setw(18) << mean_x
            << std::setw(18) << mean_y
            << std::setw(18) << var_x * 100000.0 // 转换为 10e-5
            << std::setw(18) << var_y * 100000.0 // 转换为 10e-5
            << std::endl;
    }

    // (Prompt 8.1) 计算总平均值和方差
    // 注意：表 6.5 的“平均值”是“平均误差”(Mean Error)，而不是“平均绝对误差”(Mean Absolute Error)
    // 论文中的平均误差很小（0.004801 um），这表明正负误差抵消了。
    // 我们计算总的“平均绝对误差”和总的“方差”

    double total_mean_abs_error_x, total_var_x_all;
    double total_mean_abs_error_y, total_var_y_all;

    // 计算总平均绝对误差
    calculateStatistics(total_errors_x, total_mean_abs_error_x, total_var_x_all);
    calculateStatistics(total_errors_y, total_mean_abs_error_y, total_var_y_all);

    // 论文表 6.9 的方差是组内方差的平均值，还是所有样本的总方差？这不明确。
    // 我们这里显示总的平均绝对误差和总的方差。

    std::cout << "----------------------------------------------------------------------------------" << std::endl;
    std::cout << std::setw(15) << "总平均绝对误差"
        << std::setw(18) << total_mean_abs_error_x
        << std::setw(18) << total_mean_abs_error_y
        << std::setw(18) << "---"
        << std::setw(18) << "---"
        << std::endl;
    std::cout << std::setw(15) << "总样本方差"
        << std::setw(18) << "---"
        << std::setw(18) << "---"
        << std::setw(18) << total_var_x_all * 100000.0
        << std::setw(18) << total_var_y_all * 100000.0
        << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;

    std::cout << "\n实验完成。请按 Enter 键退出..." << std::endl;
    std::cin.get();

    return 0;
}

