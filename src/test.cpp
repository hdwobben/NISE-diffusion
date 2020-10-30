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

bool allClose(MatrixXcd const &A, MatrixXcd const &B, double eps = 1e-15)
{
    MatrixXd diff = (A - B).cwiseAbs();
    long size = A.size();
    double *data = diff.data();
    
    for (long i = 0; i < size; ++i)
    {
        if (data[i] > eps)
        {
            std::cerr << "Check failed for " << data[i] << '\n';
            return false;
        }
    }
    return true;
}

int main()
{
    double R = 1; // Inter chain distance in nm
    int N = 31; // Chain length
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
    
    auto s = seedsFromClock();
    int s1 = s.first;
    int s2 = s.second;
    for (int run = 0; run < 100; ++run)
    {   
        RandomGenerator rnd(s1, s2);
        Eigen::SelfAdjointEigenSolver<MatrixXd> solver(H0.cols());
        for (int i = 0; i < 500; ++i)
        {
            updateHf(Hf, rnd, dt, lam, sig);
            MatrixXd H = H0 + Hf;    
            solver.computeFromTridiagonal(H.diagonal(), H.diagonal(-1) );
            VectorXcd L = 
                (solver.eigenvalues().array() * -1i * dt).exp();
            MatrixXd const &U = solver.eigenvectors();
            MatrixXcd ex1 = U * L.asDiagonal() * U.transpose();
            // std::cout << U * U.transpose() << "\n\n\n";
            MatrixXcd ex2 = (-1i * H * dt).exp();
            if (not allClose(ex1, ex2, 1e-13) )
            {
                std::cout << s1 << ' ' << s2 << ' ' << run << ' ' << i << '\n' << (ex1 - ex2).cwiseAbs() << '\n';
                return 0;
            }
            // c = U * L * U.transpose() * c;
        }
        s2 = (s2 + 1) % 30081;
    }
}