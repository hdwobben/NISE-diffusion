#include <iostream>
#include <complex>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <NISE/random.hpp>
#include <NISE/utils.hpp>
#include <NISE/threadpool.hpp>
 
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
    long N = Hf.cols();
    ArrayXd r{N};
    double A = std::sqrt(1 - std::exp(-2 * dt * lam));

    for (int i = 0; i < N; ++i)
        r[i] = A * rnd.RandomGaussian(0, sig);

    Hf.diagonal() *= std::exp(-lam * dt);
    Hf.diagonal().array() += r;
}

ArrayXd evolve(RandomGenerator rnd, MatrixXd &H0, MatrixXd &Hf0,
               VectorXcd &c0, RowVectorXd &xxsq, 
               double dt, double lam, double sig, int nTimeSteps)
{
    MatrixXd H;
    MatrixXd Hf = Hf0;
    VectorXcd c = c0;
    ArrayXd msdi{nTimeSteps};

    for (int ti = 0; ti < nTimeSteps; ++ti)
    {
        // < (x(t) - x(0))^2 > (expectation value)
        msdi[ti] = xxsq * c.cwiseAbs2();
        updateHf(Hf, rnd, dt, lam, sig);
        H = H0 + Hf;
        c = (-1i * H * dt).exp() * c; // hbar = 1
    }
    return msdi;
}

int main()
{
    double R = 1; // Inter chain distance in nm
    int N = 91; // Chain length
    // Coupling constant in units of hbar = 1 (fs^-1)
    double J = 5.650954701926560e-03; // = 30 cm^-1 * hc / hbar
    double sig = 2 * J;
    double lam = 50; // 1/T in fs^-1
    double dt = 10; // time step in fs

    // Constant part of the Hamiltonian (site basis)
    MatrixXd H0 = MatrixXd::Zero(N, N);
    H0.diagonal(1) = VectorXd::Constant(N - 1, J);
    H0.diagonal(-1) = VectorXd::Constant(N - 1, J);

    // Fluctuating part of the Hamiltonian (site basis)
    MatrixXd Hf0 = MatrixXd::Zero(N, N);
    
    VectorXcd c0 = VectorXcd::Zero(N);
    c0[N / 2] = 1; // c(0), a single excitation in the middle of the chain

    double x0 = ((N + 1) / 2) * R; // middle of the chain
    
    // The diagonals of the matrix form of the operator (x - x0)^2
    RowVectorXd xxsq = 
        ArrayXd::LinSpaced(N, 1, N).square() * (R * R) - 
        ArrayXd::LinSpaced(N, 1, N) * 2 * R * x0 + (x0 * x0);
    
    int nTimeSteps = 600; // final time = nTimeSteps * dt
    int nRuns = 100;
    
    ThreadPool pool(std::thread::hardware_concurrency());
    
    std::vector<std::future<ArrayXd>> results;
    results.reserve(nRuns);

    auto s = seedsFromClock();
    int seed1 = s.first;
    int seed2 = s.second;

    for (int run = 0; run < nRuns; ++run)
    {
        RandomGenerator rnd(seed1, seed2);
        results.push_back(
            pool.enqueue(evolve, rnd, H0, Hf0, c0, xxsq, 
                         dt, lam, sig, nTimeSteps));
        seed2 = (seed2 + 1) % 30081;
    }

    // mean squared displacement
    ArrayXd msd = ArrayXd::Zero(nTimeSteps);

    for (int run = 0; run < nRuns; ++run)
    {
        msd += results[run].get();
        print_progress(std::cout, run + 1, nRuns, "", "", 1, 20);
    }
    msd /= nRuns;
}