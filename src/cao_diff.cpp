#include <Eigen/Dense>
#include <NISE/random.hpp>
#include <NISE/threading/threadpool.hpp>
#include <NISE/utils.hpp>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::RowVectorXcd;
using Eigen::RowVectorXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;
using DiagonalMatrixXcd =
    Eigen::DiagonalMatrix<Eigen::dcomplex, Eigen::Dynamic>;

using namespace std::complex_literals;

// hbar / (hc * 100 cm/m) * 1e15 fs/s [cm^-1 fs]
const double hbar_cm1_fs = 1 / (2 * M_PI * 299792458.0 * 100) * 1e15;

void updateTrajectory(VectorXd &Hf, RandomGenerator &rnd, Params const &p)
{
    Hf *= std::exp(-p.lam * p.dt);

    double A = std::sqrt(1 - std::exp(-2 * p.dt * p.lam));

    for (int i = 0; i < p.N; ++i)
        Hf[i] += A * rnd.RandomGaussian(0, p.sig);
}

double calcEffectiveSigma(Params p)
{
    // Hamiltonian (site basis) in cm^-1
    MatrixXd H = MatrixXd::Zero(p.N, p.N);
    H.diagonal(1) = VectorXd::Constant(p.N - 1, p.J);
    H.diagonal(-1) = VectorXd::Constant(p.N - 1, p.J);

    // p.nTimeSteps = 100000;
    RandomGenerator rnd;
    VectorXd Hf = VectorXd::Zero(p.N, p.N);
    for (int i = 0; i < p.N; ++i) // Prepare the starting disorder
        Hf[i] = rnd.RandomGaussian(0, p.sig);
    H.diagonal() = Hf;

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(p.N);

    ArrayXd meanM2 = ArrayXd::Zero(2 * p.N);
    Eigen::Map<ArrayXd> mean(meanM2.data(), p.N);

    // sum of squared distance from the mean
    Eigen::Map<ArrayXd> m2(meanM2.data() + p.N, p.N);

    for (int ti = 0; ti < p.nTimeSteps; ++ti) {
        solver.computeFromTridiagonal(H.diagonal(), H.diagonal(-1));
        VectorXd const &w = solver.eigenvalues();
        for (int ni = 0; ni < p.N; ++ni) {
            // Welford's algorithm for mean and variance
            double d1 = w[ni] - mean[ni];
            mean[ni] += d1 / (ti + 1);
            double d2 = w[ni] - mean[ni];
            m2[ni] += d1 * d2;
        }

        updateTrajectory(Hf, rnd, p);
        H.diagonal() = Hf;
    }
    // m2 /= p.nTimeSteps - 1;
    m2 = (m2 / (p.nTimeSteps - 1)).sqrt();

    // double var = 0;
    // for (int ni = 1; ni < p.N - 1; ++ni)
    //     var += (m2[ni] - var) / ni;
    ArrayXd::Index maxc, minc;
    mean.minCoeff(&minc);
    mean.maxCoeff(&maxc);
    if (minc != 0 or maxc != p.N - 1) {
        std::cerr << "Error!" << '\n';
        throw;
    }

    double hw = std::sqrt(2 * std::log(2));
    double left = mean[1] - hw * m2[1];
    double right = mean[p.N - 2] + hw * m2[p.N - 2];

    // return std::sqrt(var);
    std::cout << "Gamma_eff / J = "
              << std::pow((right - left) / (2 * hw), 2) /
                     (hbar_cm1_fs * p.lam * p.J)
              << '\n';
    std::cout << "Gamma / J = " << p.sig * p.sig / (hbar_cm1_fs * p.lam * p.J)
              << '\n';
    return (right - left) / (2 * hw);
}

double calcCao(Params const &p, bool periodic)
{
    // Constant part of the Hamiltonian (site basis) in cm^-1
    MatrixXd H0 = MatrixXd::Zero(p.N, p.N);
    H0.diagonal(1) = VectorXd::Constant(p.N - 1, p.J);
    H0.diagonal(-1) = VectorXd::Constant(p.N - 1, p.J);

    // Flux operator j(u) (site basis) in units of R [fs^-1]
    MatrixXcd js = MatrixXcd::Zero(p.N, p.N);
    js.diagonal(1) = VectorXcd::Constant(p.N - 1, 1i * p.J / hbar_cm1_fs);
    js.diagonal(-1) = VectorXcd::Constant(p.N - 1, -1i * p.J / hbar_cm1_fs);

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(p.N);
    if (periodic) {
        H0(0, p.N - 1) = p.J;
        H0(p.N - 1, 0) = p.J;
        js(0, p.N - 1) = -1i * p.J;
        js(p.N - 1, 0) = 1i * p.J;
        solver.compute(H0);
    }
    else
        solver.computeFromTridiagonal(H0.diagonal(), H0.diagonal(-1));

    MatrixXd const &v = solver.eigenvectors();
    VectorXd const &w = solver.eigenvalues();

    // j(u) in eigenbasis in units of R [fs^-1]
    MatrixXcd je = v.adjoint() * js * v;

    // Bath hom. line width [cm^-1]
    double gamma = p.sig * p.sig / (p.lam * hbar_cm1_fs);

    // Diffusion constant in units of R^2 [fs^-1]
    double D = 0;

    for (int mu = 0; mu < p.N; ++mu) {
        for (int nu = 0; nu < p.N; ++nu) {
            double omega = w[mu] - w[nu];
            D += hbar_cm1_fs * std::norm(je(mu, nu)) * gamma /
                 (std::pow(gamma, 2) + std::pow(omega, 2));
        }
    }

    D /= p.N;

    return D;
}

double calcCaoIterative(Params p, bool periodic = false)
{
    double D = calcCao(p, periodic);
    double prevD = D - 1;
    p.N = 3000;

    while (std::abs(D - prevD) / D > 1e-4) {
        prevD = D;
        p.N += 500;
        D = calcCao(p, periodic);
        std::cout << "D = " << D << ", N = " << p.N
                  << ", diff = " << std::abs(D - prevD) << '\n';
    }
    std::cout << "Converged analytic for N = " << p.N
              << "\nD = " << std::setprecision(12) << D << '\n';
    return D;
}

#include <filesystem>
namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    size_t nThreads;
    char *penv;

    if ((penv = std::getenv("SLURM_JOB_CPUS_ON_NODE")))
        nThreads = std::stoul(penv);
    else
        nThreads = std::thread::hardware_concurrency();

    ThreadPool pool(nThreads);

    for (auto &entry: fs::directory_iterator(fs::current_path())) {
        if (not entry.is_regular_file() or entry.path().extension() != ".json")
            continue;

        std::cout << entry.path() << '\n';
        Params p = loadParams(entry.path().c_str());
        std::cout << p << '\n';
        std::string outFile = entry.path().stem().c_str() + std::string(".cao");
        p.sig = calcEffectiveSigma(p);
        p.N = 4900;
        pool.enqueue_work(
            [p, outFile]()
            {
                double D = calcCao(p, false);
                saveData(outFile, &D, 1, p);
            });
    }
}