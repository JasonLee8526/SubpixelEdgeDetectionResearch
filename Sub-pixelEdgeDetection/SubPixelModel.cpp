// (�޸� 4): ���� _USE_MATH_DEFINES ���� <cmath> ������ M_PI
// ���������� include ֮ǰ
#define _USE_MATH_DEFINES
#include <cmath>

#include "SubPixelModel.h"
#include <numeric>
#include <iostream>

// (Prompt 7.3) ���� Eigen ���Ŀ�ͷ������Ż�ģ��
#include <Eigen/Core>
// (�޸� C2027): ���� SVD ģ���Զ��� Eigen::BDCSVD
#include <Eigen/SVD> 
#include <unsupported/Eigen/NonLinearOptimization>

// --- R-Squared ���� ---

double SubPixelModel::calculateRSquared(const std::vector<double>& y_data, const std::vector<double>& y_fit) {
    if (y_data.size() != y_fit.size() || y_data.empty()) return 0.0;

    double sum_y = std::accumulate(y_data.begin(), y_data.end(), 0.0);
    double mean_y = sum_y / y_data.size();

    double ss_total = 0.0; // ��ƽ����
    double ss_res = 0.0;   // �в�ƽ����

    for (size_t i = 0; i < y_data.size(); ++i) {
        ss_total += (y_data[i] - mean_y) * (y_data[i] - mean_y);
        ss_res += (y_data[i] - y_fit[i]) * (y_data[i] - y_fit[i]);
    }

    if (ss_total < 1e-9) return 1.0; // ������ϻ���

    // (���� ��ʽ 4.1)
    return 1.0 - (ss_res / ss_total);
}

// --- ģ�� 1: �ҶȾ� ---

FitResult SubPixelModel::calcGrayMoment(const std::vector<double>& grayData) {
    // (���� 4.2.1 ��)
    int n = static_cast<int>(grayData.size());
    double m1 = 0, m2 = 0, m3 = 0;
    for (double val : grayData) {
        m1 += val;
        m2 += val * val;
        m3 += val * val * val;
    }
    m1 /= n;
    m2 /= n;
    m3 /= n;

    double sigma_sq = m2 - m1 * m1;
    if (std::abs(sigma_sq) < 1e-9) return FitResult(); // ��ĸΪ0

    double sigma = std::sqrt(sigma_sq);
    double s = (m3 + 2 * m1 * m1 * m1 - 3 * m1 * m2) / (sigma_sq * sigma);

    double P1 = 0.5 * (1.0 + s * std::sqrt(1.0 / (4.0 + s * s)));
    double d = n * P1; // ���Ĺ�ʽ 4.8

    FitResult res;
    res.edge_position = d;
    res.success = true;
    // R-Squared ������û�����壬��Ϊ��ֱ�Ӽ���
    return res;
}

// --- ģ�� 2: �ռ�� ---

FitResult SubPixelModel::calcSpatialMoment(const std::vector<double>& grayData) {
    // (���� 4.2.1 ��)
    int n = static_cast<int>(grayData.size());
    double m0 = 0, m1 = 0, m2 = 0;
    double x_norm_step = 2.0 / (n - 1); // �� [0, n-1] ��һ���� [-1, 1]

    for (int i = 0; i < n; ++i) {
        double x_norm = -1.0 + i * x_norm_step;
        double f_x = grayData[i];
        m0 += f_x;
        m1 += x_norm * f_x;
        m2 += x_norm * x_norm * f_x;
    }

    if (std::abs(m1) < 1e-9) return FitResult(); // ��ĸΪ0

    // ���Ĺ�ʽ 4.14
    double d_norm = (3.0 * m2 - m0) / (2.0 * m1);

    // ����һ���� d ת������������
    double d = (d_norm + 1.0) * (n - 1) / 2.0;

    FitResult res;
    res.edge_position = d;
    res.success = true;
    return res;
}


// --- ģ�� 3: Sigmoid (���������) ---
// (Prompt 7.3) ���� Eigen Levenberg-Marquardt ����� Functor
// S(x) = a / (1 + exp(-b*(x-c))) + d
struct SigmoidFunctor {
    const std::vector<double>& x_data;
    const std::vector<double>& y_data;

    SigmoidFunctor(const std::vector<double>& x, const std::vector<double>& y) : x_data(x), y_data(y) {}

    // ���� params: [a, b, c, d]
    // fvec: �в����� [model_y(i) - y_data(i)]
    int operator()(const Eigen::VectorXd& params, Eigen::VectorXd& fvec) const {
        double a = params(0), b = params(1), c = params(2), d = params(3);
        for (int i = 0; i < x_data.size(); ++i) {
            double x = x_data[i];
            double exp_term = std::exp(-b * (x - c));
            double model_y = a / (1.0 + exp_term) + d;
            fvec(i) = model_y - y_data[i];
        }
        return 0;
    }

    // fjac: �ſɱȾ���
    int df(const Eigen::VectorXd& params, Eigen::MatrixXd& fjac) const {
        double a = params(0), b = params(1), c = params(2), d = params(3);
        for (int i = 0; i < x_data.size(); ++i) {
            double x = x_data[i];
            double exp_term = std::exp(-b * (x - c));
            double denom = 1.0 + exp_term;
            double denom_sq = denom * denom;

            // df/da
            fjac(i, 0) = 1.0 / denom;
            // df/db
            fjac(i, 1) = a * (x - c) * exp_term / denom_sq;
            // df/dc
            fjac(i, 2) = -a * b * exp_term / denom_sq;
            // df/dd
            fjac(i, 3) = 1.0;
        }
        return 0;
    }

    // ����������� (a, b, c, d)
    int m_inputs{ 4 };
    // ���ݵ����� (41)
    int m_values{ static_cast<int>(x_data.size()) };

    // �� LevenbergMarquardt ʹ��
    int inputs() const { return m_inputs; }
    int values() const { return m_values; }
};

FitResult SubPixelModel::fitSigmoid(const std::vector<double>& grayData) {
    int n = static_cast<int>(grayData.size());
    if (n == 0) return FitResult();

    std::vector<double> x_data(n);
    std::iota(x_data.begin(), x_data.end(), 0.0); // x = 0, 1, ..., 40

    // ���� Functor
    SigmoidFunctor functor(x_data, grayData);
    Eigen::LevenbergMarquardt<SigmoidFunctor> lm(functor);

    // ���ó�ʼ����
    Eigen::VectorXd params(4);
    params(3) = grayData[0]; // d = min gray
    params(0) = grayData.back() - grayData[0]; // a = max_gray - min_gray
    params(1) = 0.5; // b (slope) - ��һ��
    params(2) = (double)n / 2.0; // c (center)

    // ������С��
    Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(params);

    FitResult res;
    // (�޸� 3): �����ȷ�ĳɹ�״̬
    if (status == Eigen::LevenbergMarquardtSpace::Status::RelativeErrorTooSmall ||
        status == Eigen::LevenbergMarquardtSpace::Status::RelativeReductionTooSmall ||
        status == Eigen::LevenbergMarquardtSpace::Status::RelativeErrorAndReductionTooSmall)
    {
        res.edge_position = params(2); // c (���Ĺ�ʽ 4.15)
        res.success = true;

        // ���� R-Squared
        std::vector<double> y_fit(n);
        Eigen::VectorXd fvec(n);
        functor(params, fvec);
        // y_fit = model_y. fvec = model_y - y_data => model_y = fvec + y_data
        for (int i = 0; i < n; ++i) y_fit[i] = fvec(i) + grayData[i];
        res.r_squared = calculateRSquared(grayData, y_fit);
    }

    return res;
}


// --- ģ�� 4: ������ (���������) ---
// A(x) = a * atan(b*x + c) + d
struct ArctanFunctor {
    const std::vector<double>& x_data;
    const std::vector<double>& y_data;

    ArctanFunctor(const std::vector<double>& x, const std::vector<double>& y) : x_data(x), y_data(y) {}

    // params: [a, b, c, d]
    int operator()(const Eigen::VectorXd& params, Eigen::VectorXd& fvec) const {
        double a = params(0), b = params(1), c = params(2), d = params(3);
        for (int i = 0; i < x_data.size(); ++i) {
            double model_y = a * std::atan(b * x_data[i] + c) + d;
            fvec(i) = model_y - y_data[i];
        }
        return 0;
    }

    int df(const Eigen::VectorXd& params, Eigen::MatrixXd& fjac) const {
        double a = params(0), b = params(1), c = params(2), d = params(3);
        for (int i = 0; i < x_data.size(); ++i) {
            double x = x_data[i];
            double inner = b * x + c;
            double denom = 1.0 + inner * inner;

            fjac(i, 0) = std::atan(inner);       // df/da
            fjac(i, 1) = a * x / denom;          // df/db
            fjac(i, 2) = a / denom;              // df/dc
            fjac(i, 3) = 1.0;                    // df/dd
        }
        return 0;
    }

    int m_inputs{ 4 };
    int m_values{ static_cast<int>(x_data.size()) };
    int inputs() const { return m_inputs; }
    int values() const { return m_values; }
};

FitResult SubPixelModel::fitArctan(const std::vector<double>& grayData) {
    int n = static_cast<int>(grayData.size());
    if (n == 0) return FitResult();

    std::vector<double> x_data(n);
    std::iota(x_data.begin(), x_data.end(), 0.0);

    ArctanFunctor functor(x_data, grayData);
    Eigen::LevenbergMarquardt<ArctanFunctor> lm(functor);

    Eigen::VectorXd params(4);
    // ��ʼ�²�
    double min_y = grayData[0];
    double max_y = grayData.back();
    // (�޸� 4): M_PI �����Ѷ���
    params(0) = (max_y - min_y) / M_PI; // a 
    params(1) = 0.5; // b
    params(2) = -params(1) * (double)n / 2.0; // c
    params(3) = (max_y + min_y) / 2.0; // d

    Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(params);

    FitResult res;
    // (�޸� 3): �����ȷ�ĳɹ�״̬
    if (status == Eigen::LevenbergMarquardtSpace::Status::RelativeErrorTooSmall ||
        status == Eigen::LevenbergMarquardtSpace::Status::RelativeReductionTooSmall ||
        status == Eigen::LevenbergMarquardtSpace::Status::RelativeErrorAndReductionTooSmall)
    {
        double b = params(1), c = params(2);
        if (std::abs(b) > 1e-9) {
            res.edge_position = -c / b; // (���� �� 4.7)
            res.success = true;

            std::vector<double> y_fit(n);
            Eigen::VectorXd fvec(n);
            functor(params, fvec);
            // y_fit = model_y. fvec = model_y - y_data => model_y = fvec + y_data
            for (int i = 0; i < n; ++i) y_fit[i] = fvec(i) + grayData[i];
            res.r_squared = calculateRSquared(grayData, y_fit);
        }
    }
    return res;
}


// --- ģ�� 5 & 6 �������� (������С������϶��ζ���ʽ) ---
// P(x) = a*x^2 + b*x + c
FitResult fitQuadraticInternal(const std::vector<double>& x_data, const std::vector<double>& y_data) {
    int n = static_cast<int>(x_data.size());
    if (n == 0) return FitResult();

    Eigen::MatrixXd A(n, 3);
    Eigen::VectorXd B(n);

    for (int i = 0; i < n; ++i) {
        double x = x_data[i];
        A(i, 0) = x * x; // a
        A(i, 1) = x;     // b
        A(i, 2) = 1;     // c
        B(i) = y_data[i];
    }

    // (�޸� C2027) �������ڿ��Թ����ˣ���Ϊ <Eigen/SVD> �Ѿ�������
    Eigen::VectorXd coeffs = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);

    double a = coeffs(0), b = coeffs(1), c = coeffs(2);

    FitResult res;
    if (std::abs(a) < 1e-9) return res; // a Ϊ 0�����Ƕ���

    res.edge_position = -b / (2.0 * a); // ��ֵ�� (���� �� 4.7)
    res.success = true;

    // ���� R-Squared
    std::vector<double> y_fit(n);
    for (int i = 0; i < n; ++i) {
        double x = x_data[i];
        y_fit[i] = a * x * x + b * x + c;
    }
    res.r_squared = SubPixelModel::calculateRSquared(y_data, y_fit);
    return res;
}

// --- ģ�� 5: ���ζ���ʽ ---

FitResult SubPixelModel::fitQuadratic(const std::vector<double>& gradData) {
    int n = static_cast<int>(gradData.size());
    std::vector<double> x_data(n);
    std::iota(x_data.begin(), x_data.end(), 0.0);
    return fitQuadraticInternal(x_data, gradData);
}

// --- ģ�� 6: ��˹ ---

FitResult SubPixelModel::fitGaussian(const std::vector<double>& gradData) {
    // (���� 4.3 ��, ��ʽ 4.19 - 4.23)
    // G(x) = a * exp(...)
    // F(x) = ln(G(x)) = a0 + a1*x + a2*x^2
    int n = static_cast<int>(gradData.size());
    std::vector<double> x_data(n);
    std::vector<double> y_log_data(n);
    double min_val = 1e-6; // ���� log(0)

    for (int i = 0; i < n; ++i) {
        x_data[i] = static_cast<double>(i);
        y_log_data[i] = std::log(std::max(gradData[i], min_val));
    }

    // ��� F(x) = a2*x^2 + a1*x + a0
    FitResult quadratic_res = fitQuadraticInternal(x_data, y_log_data);

    if (!quadratic_res.success) return FitResult();

    // ���´� coeffs ��ȡ a0, a1, a2
    Eigen::MatrixXd A(n, 3);
    Eigen::VectorXd B(n);
    for (int i = 0; i < n; ++i) {
        A(i, 0) = x_data[i] * x_data[i]; A(i, 1) = x_data[i]; A(i, 2) = 1;
        B(i) = y_log_data[i];
    }
    // (�޸� C2027) �������ڿ��Թ�����
    Eigen::VectorXd coeffs = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    double a2 = coeffs(0), a1 = coeffs(1);

    FitResult res;
    if (std::abs(a2) < 1e-9) return res;

    // (���� 4.21, 4.22)
    // b = a1 / (-2 * a2)
    res.edge_position = a1 / (-2.0 * a2); // ��ֵ�� 'b'
    res.success = true;

    // R-Squared Ӧ����ԭʼ�ռ���㣬������ log �ռ�
    double c_sq = -1.0 / a2;
    if (c_sq <= 0) { // ȷ�� c_sq Ϊ��
        res.success = false;
        return res;
    }
    double b = res.edge_position;
    double a = std::exp(coeffs(2) + b * b / c_sq); // a0 = ln(a) - b^2/c^2 => ln(a) = a0 + b^2/c^2

    std::vector<double> y_fit(n);
    for (int i = 0; i < n; ++i) {
        y_fit[i] = a * std::exp(-(x_data[i] - b) * (x_data[i] - b) / c_sq);
    }
    res.r_squared = calculateRSquared(gradData, y_fit);

    return res;
}

