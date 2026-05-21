/*
=================================================================
  GCA-ND  —  Geometric Coordinate Ascent (N-Dimensional)
  C++17 + OpenMP  |  Gradyan/türev YOK
  
  Desteklenen Loss Fonksiyonları:
    0 = MSE   (Mean Squared Error)
    1 = MAE   (Mean Absolute Error)        ← OLS'nin bozulduğu alan
    2 = HUBER (Huber Loss, delta=1.0)      ← outlier-robust
    3 = MSE + L1 (Lasso regularization)   ← sparse çözüm
    4 = MSE + L2 (Ridge regularization)

  Kullanım:
    ./gca_nd <loss_id> <lambda> < data.csv
    
    Veri formatı (stdin, CSV):
      İlk satır: m n          (satır sayısı, özellik sayısı)
      Sonraki m satır: x1 x2 ... xn y
    
    Çıktı (stdout, CSV):
      theta_0 theta_1 ... theta_n mse mae r2 time_ms iterations
=================================================================
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <string>
#include <sstream>
#include <cassert>
#include <omp.h>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

// ─── Yardımcı ─────────────────────────────────────────────
inline double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// ─── Loss Fonksiyonları ───────────────────────────────────
enum LossType { MSE=0, MAE=1, HUBER=2, LASSO=3, RIDGE=4 };

struct LossConfig {
    LossType type;
    double lambda;     // regularization katsayısı
    double huber_delta = 1.0;
};

// Tek örnek için residual katkısı
inline double sample_loss(double r, const LossConfig& cfg) {
    double abs_r = std::abs(r);
    switch (cfg.type) {
        case MSE:   return r * r;
        case MAE:   return abs_r;
        case HUBER:
            if (abs_r <= cfg.huber_delta)
                return 0.5 * r * r;
            else
                return cfg.huber_delta * (abs_r - 0.5 * cfg.huber_delta);
        case LASSO: return r * r;   // reg terimi ayrıca eklenir
        case RIDGE: return r * r;
    }
    return r * r;
}

// Tam f(θ) = −[loss + reg]
double objective(const Vec& theta,
                 const Mat& X,   // X bias dahil (m x n+1)
                 const Vec& y,
                 const LossConfig& cfg) {
    int m = (int)y.size();
    int p = (int)theta.size();
    double total = 0.0;

    #pragma omp parallel for reduction(+:total) schedule(static)
    for (int i = 0; i < m; ++i) {
        double pred = 0.0;
        for (int j = 0; j < p; ++j) pred += X[i][j] * theta[j];
        total += sample_loss(y[i] - pred, cfg);
    }
    total /= m;

    // Regularization (bias hariç)
    if (cfg.type == LASSO) {
        double reg = 0.0;
        for (int j = 1; j < p; ++j) reg += std::abs(theta[j]);
        total += cfg.lambda * reg;
    } else if (cfg.type == RIDGE) {
        double reg = 0.0;
        for (int j = 1; j < p; ++j) reg += theta[j] * theta[j];
        total += cfg.lambda * reg;
    }

    return -total;   // maksimizasyon için negatif
}

// ─── 1D Level-Set Taraması ────────────────────────────────
// j. ekseni serbest bırak, grid üzerinde en iyi noktayı bul
double scan_1d(Vec theta,          // kopya (değiştirilecek)
               int j,
               double lo, double hi,
               int n_grid,
               const Mat& X,
               const Vec& y,
               const LossConfig& cfg) {
    double best_val = theta[j];
    double best_f   = objective(theta, X, y, cfg);
    double step     = (hi - lo) / (n_grid - 1);

    // OpenMP ile grid noktalarını paralel değerlendir
    #pragma omp parallel
    {
        Vec local_theta = theta;
        double local_best_val = best_val;
        double local_best_f   = best_f;

        #pragma omp for schedule(static)
        for (int k = 0; k < n_grid; ++k) {
            double v = lo + k * step;
            local_theta[j] = v;
            double f = objective(local_theta, X, y, cfg);
            if (f > local_best_f) {
                local_best_f   = f;
                local_best_val = v;
            }
        }

        #pragma omp critical
        {
            if (local_best_f > best_f) {
                best_f   = local_best_f;
                best_val = local_best_val;
            }
        }
    }
    return best_val;
}

// ─── GCA-ND Ana Algoritması ───────────────────────────────
struct GCAResult {
    Vec    theta;
    double mse;
    double mae;
    double r2;
    double time_ms;
    int    iterations;
    std::vector<double> f_history;
};

GCAResult gca_fit(const Mat& X_raw,   // bias EKLENMEMİŞ (m x n)
                  const Vec& y,
                  const LossConfig& cfg,
                  int    n_grid    = 500,
                  int    max_outer = 50,
                  double tol       = 1e-7) {
    int m = (int)y.size();
    int n = (int)X_raw[0].size();
    int p = n + 1;   // bias + n ağırlık

    // Bias sütunu ekle
    Mat X(m, Vec(p));
    for (int i = 0; i < m; ++i) {
        X[i][0] = 1.0;
        for (int j = 0; j < n; ++j) X[i][j+1] = X_raw[i][j];
    }

    // Otomatik arama aralıkları
    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / m;
    double y_var  = 0.0;
    for (double v : y) y_var += (v-y_mean)*(v-y_mean);
    double y_std = std::sqrt(y_var / m);

    std::vector<std::pair<double,double>> ranges(p);
    ranges[0] = {y_mean - 3*y_std, y_mean + 3*y_std};   // bias
    for (int j = 1; j < p; ++j) {
        double xmean = 0.0;
        for (int i = 0; i < m; ++i) xmean += X[i][j];
        xmean /= m;
        double xvar = 0.0;
        for (int i = 0; i < m; ++i) xvar += (X[i][j]-xmean)*(X[i][j]-xmean);
        double xstd = std::sqrt(xvar / m + 1e-12);
        double scale = y_std / xstd;
        ranges[j] = {-3*scale, 3*scale};
    }

    // Başlangıç: aralık ortaları
    Vec theta(p);
    for (int j = 0; j < p; ++j)
        theta[j] = (ranges[j].first + ranges[j].second) / 2.0;

    auto t_start = std::chrono::high_resolution_clock::now();

    GCAResult res;
    res.f_history.push_back(objective(theta, X, y, cfg));

    for (int outer = 0; outer < max_outer; ++outer) {
        Vec theta_old = theta;

        for (int j = 0; j < p; ++j) {
            double lo  = ranges[j].first;
            double hi  = ranges[j].second;
            double span = hi - lo;
            // Dinamik pencere: logaritmik daralma
            double shrink = std::pow(0.55, outer);
            double w = std::max(span * shrink, span * 0.005);
            double lo_dyn = std::max(lo, theta[j] - w);
            double hi_dyn = std::min(hi, theta[j] + w);

            theta[j] = scan_1d(theta, j, lo_dyn, hi_dyn, n_grid, X, y, cfg);
        }

        res.f_history.push_back(objective(theta, X, y, cfg));

        // Yakınsama kontrolü
        double delta = 0.0;
        for (int j = 0; j < p; ++j)
            delta = std::max(delta, std::abs(theta[j] - theta_old[j]));
        if (delta < tol) {
            res.iterations = outer + 1;
            break;
        }
        res.iterations = outer + 1;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    res.theta = theta;

    // MSE, MAE, R²
    double ss_res = 0.0, ss_tot = 0.0, mae_sum = 0.0;
    for (int i = 0; i < m; ++i) {
        double pred = dot(X[i], theta);
        double r    = y[i] - pred;
        ss_res += r * r;
        mae_sum += std::abs(r);
        ss_tot  += (y[i] - y_mean) * (y[i] - y_mean);
    }
    res.mse = ss_res / m;
    res.mae = mae_sum / m;
    res.r2  = 1.0 - ss_res / (ss_tot + 1e-12);

    return res;
}

// ─── Main: stdin'den veri oku, stdout'a sonuç yaz ─────────
int main(int argc, char* argv[]) {
    LossConfig cfg;
    cfg.type   = MSE;
    cfg.lambda = 0.01;

    if (argc >= 2) cfg.type   = static_cast<LossType>(std::stoi(argv[1]));
    if (argc >= 3) cfg.lambda = std::stod(argv[2]);

    // Veri oku
    int m, n;
    std::cin >> m >> n;
    Mat X(m, Vec(n));
    Vec y(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) std::cin >> X[i][j];
        std::cin >> y[i];
    }

    GCAResult res = gca_fit(X, y, cfg, 500, 50, 1e-7);

    // Çıktı: theta... mse mae r2 time_ms iterations
    for (double v : res.theta) std::cout << v << " ";
    std::cout << res.mse << " " << res.mae << " "
              << res.r2  << " " << res.time_ms << " "
              << res.iterations << "\n";

    // f_history (ikinci satır)
    for (double v : res.f_history) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
