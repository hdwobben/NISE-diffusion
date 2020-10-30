#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <complex>
#include <NISE/random.hpp>
#include <cmath>
//#include <NISE/threadpool.hpp>
 
using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::VectorXcd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::RowVectorXcd;
using Eigen::ArrayXd;
using DiagonalMatrixXcd = 
    Eigen::DiagonalMatrix<Eigen::dcomplex, Eigen::Dynamic>;

using namespace std::complex_literals;

void updateHf(MatrixXd &Hf, RandomGenerator &rnd, double dt, double lam, double sig)
{
    long int N = Hf.cols();
    Eigen::ArrayXd r{N};
    double A = std::sqrt(1 - std::exp(-2 * dt * lam));

    for (int i = 0; i < N; ++i)
        r[i] = A * rnd.RandomGaussian(0, sig);

    Hf.diagonal() *= std::exp(-lam * dt);
    Hf.diagonal().array() += r;
}

int main()
{
    double R = 1; // Inter chain distance in nm
    int N = 7; // Chain length
    // Coupling constant in units of hbar = 1 (fs^-1)
    double J = 5.650954701926560e-03; // = 30 cm^-1 * hc / hbar
    double sig = 2 * J;
    double lam = 50; // 1/T in fs^-1
    double dt = 10; // time step in fs

    MatrixXd H0 = MatrixXd::Zero(N, N);
    H0.diagonal(1) = VectorXd::Constant(N - 1, J);
    H0.diagonal(-1) = VectorXd::Constant(N - 1, J);

    // Fluctuating part of the Hamiltonian (site basis)
    MatrixXd Hf = MatrixXd::Zero(N, N);
    
    RandomGenerator rnd;
    updateHf(Hf, rnd, dt, lam, sig);
    MatrixXd H = H0 + Hf;
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(H0.cols());
    solver.computeFromTridiagonal(H.diagonal(), H.diagonal(-1) );
    VectorXcd L = 
        (solver.eigenvalues().array() * -1i * dt).exp();
    MatrixXd U = solver.eigenvectors();
    std::cout << U * L.asDiagonal() * U.adjoint() << "\n\n\n";
    //std::cout << U * U.adjoint() << "\n\n\n";
    std::cout << (-1i * H * dt).exp() << '\n';
    // c = U * L * U.adjoint() * c;
}