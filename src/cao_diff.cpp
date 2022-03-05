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
    p.N = 1000;
    double D = calcCao(p, periodic);
    double prevD = -D;

    std::cout << "D = " << D << ", N = " << p.N << '\n';

    while (std::abs(D - prevD) / D > 1e-3) {
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

int main(int argc, char *argv[])
{
    CmdArgs cmdargs = processCmdArguments(argc, argv);
    Params p = loadParams(cmdargs.paramsFName);
    if (not cmdargs.quiet)
        std::cout << "gamma/J = " << p.sig * p.sig / (p.J * p.lam * hbar_cm1_fs)
                  << ", J = " << p.J << '\n';

    calcCaoIterative(p);
}