#include <iostream>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <Eigen/Dense>
#include <NISE/random.hpp>
#include <NISE/utils.hpp>
#include <NISE/threading/threadpool.hpp>

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

// hbar / (hc * 100 cm/m) * 1e15 fs/s [cm^-1 fs]
const double hbar_cm1_fs = 1 / (2 * M_PI * 299792458.0 * 100) * 1e15;

void updateTrajectory(VectorXd &Hf, RandomGenerator &rnd, Params const &p)
{
    Hf *= std::exp(-p.lam * p.dt);

    double A = std::sqrt(1 - std::exp(-2 * p.dt * p.lam));

    for (int i = 0; i < p.N; ++i)
        Hf[i] += A * rnd.RandomGaussian(0, p.sig);
}

ArrayXd evolve(RandomGenerator rnd, MatrixXd const &H0, VectorXcd const &c0, 
               RowVectorXd const &xxsq, Params const &p)
{
    MatrixXd H = H0;
    VectorXcd c = c0;
    VectorXd Hf = VectorXd::Zero(p.N, p.N);
    for (int i = 0; i < p.N; ++i) // Prepare the starting disorder
        Hf[i] = rnd.RandomGaussian(0, p.sig);
    H.diagonal() = Hf;

    ArrayXd msdi{p.nTimeSteps}; // <(x(t) - x0)^2> in units of R^2
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(p.N);

    for (int ti = 0; ti < p.nTimeSteps; ++ti)
    {
        // < (x(t) - x(0))^2 > (expectation value)
        msdi[ti] = xxsq * c.cwiseAbs2();
        solver.computeFromTridiagonal(H.diagonal(), H.diagonal(-1) );
        VectorXcd L = 
            (solver.eigenvalues().array() * -1i * p.dt / hbar_cm1_fs).exp();
        MatrixXd const &V = solver.eigenvectors();
        c = V * L.asDiagonal() * V.adjoint() * c;
        updateTrajectory(Hf, rnd, p);
        H.diagonal() = Hf;
    }
    return msdi;
}

int main(int argc, char *argv[])
{
    CmdArgs cmdargs = processCmdArguments(argc, argv);
    Params p = loadParams(cmdargs.paramsFName);
    if (not cmdargs.quiet)
        std::cout << p << '\n';

    // Constant part of the Hamiltonian (site basis) in cm^-1
    MatrixXd H0 = MatrixXd::Zero(p.N, p.N);
    H0.diagonal(1) = VectorXd::Constant(p.N - 1, p.J);
    H0.diagonal(-1) = VectorXd::Constant(p.N - 1, p.J);

    VectorXcd c0 = VectorXcd::Zero(p.N);
    c0[p.N / 2] = 1; // single excitation in the middle of the chain

    double x0 = ((p.N + 1) / 2); // middle of the chain in units of R
    
    // Diagonals of the matrix form of the operator (x - x0)^2 in units of R^2
    RowVectorXd xxsq = 
        ArrayXd::LinSpaced(p.N, 1, p.N).square() - 
        ArrayXd::LinSpaced(p.N, 1, p.N) * 2 * x0 + (x0 * x0);
       
    ThreadPool pool(std::thread::hardware_concurrency());
    
    std::vector<std::future<ArrayXd>> results;
    results.reserve(p.nRuns);

    auto s = seedsFromClock(); // base seeds for random numbers
    int seed1 = s.first;
    int seed2 = s.second;

    for (int run = 0; run < p.nRuns; ++run)
    {
        RandomGenerator rnd(seed1, seed2); // every thread gets its own seed
        results.push_back(
            pool.enqueue_task(evolve, rnd, H0, c0, xxsq, p));
        seed2 = (seed2 + 1) % 30081;
    }

    // <<(x(t) - x0)^2>> in units of R^2
    ArrayXd msd = ArrayXd::Zero(p.nTimeSteps);

    if (not cmdargs.quiet)
        print_progress(std::cout, 0, p.nRuns, "", "", 1, 20);

    for (int run = 0; run < p.nRuns; ++run)
    {
        msd += results[run].get();
        if (not cmdargs.quiet)
            print_progress(std::cout, run + 1, p.nRuns, "", "", 1, 20);
    }
    msd /= p.nRuns;

    if (not cmdargs.outFile)
        return 0;

    if (cmdargs.outFName.empty() )
        cmdargs.outFName = nowStrLocal("%Y%m%d%H%M%S.msd");
    
    if (cmdargs.outFName != "out.tmp")
        std::cout << cmdargs.outFName << '\n';

    saveData(cmdargs.outFName, msd.data(), msd.size(), p);
}