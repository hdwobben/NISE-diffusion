#include <Eigen/Dense>
#include <NISE/random.hpp>
#include <NISE/threading/threadpool.hpp>
#include <NISE/utils.hpp>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
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

ArrayXcd evolve(RandomGenerator rnd, MatrixXd const &H0, MatrixXcd const &j0,
                Params const &p)
{
    MatrixXd H = H0;
    MatrixXcd jt = j0;
    VectorXd Hf = VectorXd::Zero(p.N);
    for (int i = 0; i < p.N; ++i) // Prepare the starting disorder
        Hf[i] = rnd.RandomGaussian(0, p.sig);
    H.diagonal() = Hf;

    ArrayXcd integrand{p.nTimeSteps}; // Tr(j(u,t)j(u)) in units of R^2 [fs^-2]
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(p.N);

    for (int ti = 0; ti < p.nTimeSteps; ++ti) {
        integrand[ti] = (jt * j0).trace(); // jt.cwiseProduct(j0).sum();
        solver.computeFromTridiagonal(H.diagonal(), H.diagonal(-1));
        VectorXcd L =
            (solver.eigenvalues().array() * -1i * p.dt / hbar_cm1_fs).exp();
        MatrixXd const &V = solver.eigenvectors();
        MatrixXcd U = V * L.asDiagonal() * V.adjoint();
        jt = U.adjoint() * jt * U;
        updateTrajectory(Hf, rnd, p);
        H.diagonal() = Hf;
    }
    return integrand;
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

    // Flux operator j(u) (site basis) in units of R [fs^-1]
    MatrixXcd j0 = MatrixXcd::Zero(p.N, p.N);
    j0.diagonal(1) = VectorXcd::Constant(p.N - 1, 1i * p.J / hbar_cm1_fs);
    j0.diagonal(-1) = VectorXcd::Constant(p.N - 1, -1i * p.J / hbar_cm1_fs);

    unsigned long nThreads;
    char *penv;

    if ((penv = std::getenv("SLURM_JOB_CPUS_ON_NODE")))
        nThreads = std::stoul(penv);
    else
        nThreads = std::thread::hardware_concurrency();

    ThreadPool pool(nThreads);

    std::vector<std::future<ArrayXcd>> results;
    results.reserve(p.nRuns);

    auto s = seedsFromClock(); // base seeds for random numbers
    int seed1 = s.first;
    int seed2 = s.second;

    for (int run = 0; run < p.nRuns; ++run) {
        RandomGenerator rnd(seed1, seed2); // every thread gets its own seed
        results.push_back(pool.enqueue_task(evolve, rnd, H0, j0, p));
        seed2 = (seed2 + 1) % 30081;
    }

    // <Tr(j(u,t)j(u))> in units of R^2 [fs^-2]
    ArrayXcd integrand = ArrayXcd::Zero(p.nTimeSteps);

    if (not cmdargs.quiet)
        print_progress(std::cout, 0, p.nRuns, "", "", 1, 20);

    for (int run = 0; run < p.nRuns; ++run) {
        integrand += results[run].get();
        if (not cmdargs.quiet)
            print_progress(std::cout, run + 1, p.nRuns, "", "", 1, 20);
    }
    integrand /= p.nRuns;

    if (not cmdargs.outFile)
        return 0;

    if (cmdargs.outFName.empty())
        cmdargs.outFName = nowStrLocal("%Y%m%d%H%M%S.kuboint");

    if (cmdargs.outFName != "out.tmp")
        std::cout << cmdargs.outFName << '\n';

    saveData(cmdargs.outFName, integrand.data(), integrand.size(), p);
}