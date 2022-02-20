#include <Eigen/Dense>
#include <NISE/random.hpp>
#include <NISE/threading/threadpool.hpp>
#include <NISE/utils.hpp>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>

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

// class RunningStat
//     {
//     public:
//         RunningStat() : m_n(0) {}

//         void Clear()
//         {
//             m_n = 0;
//         }

//         void Push(double x)
//         {
//             m_n++;

//             // See Knuth TAOCP vol 2, 3rd edition, page 232
//             if (m_n == 1)
//             {
//                 m_oldM = m_newM = x;
//                 m_oldS = 0.0;
//             }
//             else
//             {
//                 m_newM = m_oldM + (x - m_oldM)/m_n;
//                 m_newS = m_oldS + (x - m_oldM)*(x - m_newM);

//                 // set up for next iteration
//                 m_oldM = m_newM;
//                 m_oldS = m_newS;
//             }
//         }

//         int NumDataValues() const
//         {
//             return m_n;
//         }

//         double Mean() const
//         {
//             return (m_n > 0) ? m_newM : 0.0;
//         }

//         double Variance() const
//         {
//             return ( (m_n > 1) ? m_newS/(m_n - 1) : 0.0 );
//         }

//         double StandardDeviation() const
//         {
//             return sqrt( Variance() );
//         }

//     private:
//         int m_n;
//         double m_oldM, m_newM, m_oldS, m_newS;
//     };

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

    for (int ti = 0; ti < p.nTimeSteps; ++ti) {
        // < (x(t) - x(0))^2 > (expectation value)
        msdi[ti] = xxsq * c.cwiseAbs2();
        solver.computeFromTridiagonal(H.diagonal(), H.diagonal(-1));
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

    auto s = seedsFromClock(); // base seeds for random numbers
    int seed1 = s.first;
    int seed2 = s.second;
    RandomGenerator rnd(seed1, seed2);

    // if (not cmdargs.quiet)
    //     print_progress(std::cout, 0, p.nRuns, "", "", 1, 20);

    MatrixXd H = H0;
    VectorXd Hf = VectorXd::Zero(p.N, p.N);
    for (int i = 0; i < p.N; ++i) // Prepare the starting disorder
        Hf[i] = rnd.RandomGaussian(0, p.sig);
    H.diagonal() = Hf;

    std::vector<double> eigStd;
    std::vector<double> eigMean;
    eigStd.reserve(p.N);
    eigMean.reserve(p.N);

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(p.N);

    std::vector<double> eigt(p.N * p.nTimeSteps);

    for (int ti = 0; ti < p.nTimeSteps; ++ti) {
        solver.computeFromTridiagonal(H.diagonal(), H.diagonal(-1));
        VectorXd const &w = solver.eigenvalues();

        for (int ni = 0; ni < p.N; ++ni)
            eigt[ni * p.nTimeSteps + ti] = w[ni];

        updateTrajectory(Hf, rnd, p);
        H.diagonal() = Hf;
    }

    if (not cmdargs.outFile)
        return 0;

    if (cmdargs.outFName.empty())
        cmdargs.outFName = nowStrLocal("%Y%m%d%H%M%S.eigt");

    if (cmdargs.outFName != "out.tmp")
        std::cout << cmdargs.outFName << '\n';

    saveData(cmdargs.outFName, eigt.data(), eigt.size(), p);
}