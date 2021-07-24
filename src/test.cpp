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

ArrayXd evolve(RandomGenerator rnd, MatrixXd const &H0, Params const &p)
{
    MatrixXd H = H0;
    VectorXd Hf = VectorXd::Zero(p.N);
    for (int i = 0; i < p.N; ++i) // Prepare the starting disorder
        Hf[i] = rnd.RandomGaussian(0, p.sig);
    H.diagonal() = Hf;

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(p.N);

    ArrayXd eigt{p.N * p.nTimeSteps};

    for (int ti = 0; ti < p.nTimeSteps; ++ti)
    {
        solver.computeFromTridiagonal(H.diagonal(), H.diagonal(-1) );
        VectorXd const& w = solver.eigenvalues();

        for (int ni = 0; ni < p.N; ++ni) 
            eigt[ni * p.nTimeSteps + ti] = w[ni];

        updateTrajectory(Hf, rnd, p);
        H.diagonal() = Hf;
    }
    return eigt;
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

    std::vector<std::future<ArrayXd>> results;
    results.reserve(p.nRuns);

    auto s = seedsFromClock(); // base seeds for random numbers
    int seed1 = s.first;
    int seed2 = s.second;

    ThreadPool pool(std::thread::hardware_concurrency());
    
    for (int run = 0; run < p.nRuns; ++run)
    {
        RandomGenerator rnd(seed1, seed2); // every thread gets its own seed
        results.push_back(
            pool.enqueue_task(evolve, rnd, H0, p));
        seed2 = (seed2 + 1) % 30081;
    }

    // energy eigenvalues in time
    ArrayXd eigt = ArrayXd::Zero(p.N * p.nTimeSteps);

    if (not cmdargs.quiet)
        print_progress(std::cout, 0, p.nRuns, "", "", 1, 20);

    for (int run = 0; run < p.nRuns; ++run)
    {
        eigt += results[run].get();
        if (not cmdargs.quiet)
            print_progress(std::cout, run + 1, p.nRuns, "", "", 1, 20);
    }
    eigt /= p.nRuns;
    // if (not cmdargs.quiet)
    //     print_progress(std::cout, 0, p.nRuns, "", "", 1, 20);

    if (not cmdargs.outFile)
        return 0;

    if (cmdargs.outFName.empty() )
        cmdargs.outFName = nowStrLocal("%Y%m%d%H%M%S.eigt");
    
    if (cmdargs.outFName != "out.tmp")
        std::cout << cmdargs.outFName << '\n';

    saveData(cmdargs.outFName, eigt.data(), eigt.size(), p);
}