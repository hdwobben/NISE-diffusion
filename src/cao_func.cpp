#include <Eigen/Dense>
#include <NISE/random.hpp>
#include <NISE/threading/threadpool.hpp>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <tuple>

using json = nlohmann::json;

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

struct CmdArgs {
    double min;
    double max;
    size_t num;
    std::string outFName;
    bool quiet = false;
    bool outFile = true;
};

// Print a progress bar
void print_progress(std::ostream &os, int iter, int total,
                    std::string const &prefix, std::string const &suffix,
                    int decimals, int barLength)
{
    static constexpr char barChar[] = "█";

    double percents = 100 * (iter / static_cast<double>(total));
    int filledLength = static_cast<int>(
        std::round(barLength * iter / static_cast<double>(total)));

    std::string bar;
    bar.reserve(filledLength * (sizeof(barChar) - 1) + barLength);
    for (int i = 0; i < filledLength; ++i)
        bar.append("█");
    bar.append(barLength - filledLength, '-');

    std::ios_base::fmtflags flags = os.flags(); // save old stream settings
    std::streamsize ss = os.precision();

    os << "\33[2K\r" << prefix << "(" << iter << '/' << total << ") |" << bar
       << "| " << std::fixed << std::setprecision(decimals) << percents << '%'
       << suffix;

    os.flags(flags); // restore settings
    os.precision(ss);

    if (iter == total)
        os << '\n';

    os << std::flush;
}

CmdArgs processCmdArguments(int argc, char *argv[])
{
    bool hasMin = false;
    bool hasMax = false;
    bool hasNum = false;

    struct option const long_options[] = {
        {"no-outfile", 0, NULL, 'x'}, // to allow --no-outfile
        {"min", required_argument, NULL, 'i'},
        {"max", required_argument, NULL, 'a'},
        {"num", required_argument, NULL, 'n'},
        {NULL, 0, NULL, 0}};

    CmdArgs cargs;

    int ch;
    int option_index;
    while ((ch = getopt_long(argc, argv, "o:qx", long_options,
                             &option_index)) != -1) {
        switch (static_cast<char>(ch)) {
        case 'o':
            cargs.outFName = optarg;
            break;
        case 'q':
            cargs.quiet = true;
            break;
        case 'x':
            cargs.outFile = false;
            break;
        case 'i':
            cargs.min = std::stod(optarg);
            hasMin = true;
            break;
        case 'a':
            cargs.max = std::stod(optarg);
            hasMax = true;
            break;
        case 'n':
            cargs.num = std::stoul(optarg);
            hasNum = true;
            break;
        }
    }
    if (!hasMin or !hasMax or !hasNum)
        throw std::runtime_error("Need all of min, max and num");

    return cargs;
}

std::vector<double> linspace(double min, double max, size_t num,
                             bool endpoint = true)
{
    double div = static_cast<double>(endpoint ? (num - 1) : num);
    double step = (max - min) / div;
    std::vector<double> ret(num);

    for (size_t idx = 0; idx != num; ++idx) {
        ret[idx] = min + static_cast<double>(idx) * step;
    }

    return ret;
}

std::vector<double> logspace(double min, double max, size_t num,
                             bool endpoint = true)
{
    double emin = std::log(min);
    double emax = std::log(max);
    std::vector<double> ret = linspace(emin, emax, num, endpoint);
    for (size_t idx = 0; idx != num; ++idx) {
        ret[idx] = std::exp(ret[idx]);
    }
    return ret;
}

void saveData(std::string const &fname, std::vector<uint8_t> data,
              CmdArgs const &cargs)
{
    json::binary_t binData(std::move(data));

    json dataset;
    dataset["min"] = cargs.min;
    dataset["max"] = cargs.max;
    dataset["num"] = cargs.num;
    dataset["data"] = binData;

    std::vector<uint8_t> msg = json::to_msgpack(dataset);

    std::ofstream file(fname, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<char *>(msg.data()),
               static_cast<std::streamsize>(msg.size()));
}

template <class T>
void saveData(std::string const &fname, T *data, size_t size,
              CmdArgs const &cargs)
{
    // prepare input as binary data
    std::vector<uint8_t> binData(reinterpret_cast<uint8_t *>(data),
                                 reinterpret_cast<uint8_t *>(data) +
                                     size * sizeof(T));

    saveData(fname, std::move(binData), cargs);
}

std::pair<VectorXd, MatrixXcd> calcEigsJe(int N, double J, double sigma,
                                          RandomGenerator rnd)
{
    // Constant part of the Hamiltonian (site basis) in cm^-1
    MatrixXd H0 = MatrixXd::Zero(N, N);
    H0.diagonal(1) = VectorXd::Constant(N - 1, J);
    H0.diagonal(-1) = VectorXd::Constant(N - 1, J);

    for (int i = 0; i < N; ++i) { // Prepare the starting disorder
        H0(i, i) = rnd.RandomGaussian(0, sigma);
    }

    // Flux operator j(u) (site basis) in units of R [fs^-1]
    MatrixXcd js = MatrixXcd::Zero(N, N);
    js.diagonal(1) = VectorXcd::Constant(N - 1, 1i * J / hbar_cm1_fs);
    js.diagonal(-1) = VectorXcd::Constant(N - 1, -1i * J / hbar_cm1_fs);

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(N);
    solver.computeFromTridiagonal(H0.diagonal(), H0.diagonal(-1));

    MatrixXd const &v = solver.eigenvectors();
    VectorXd const &w = solver.eigenvalues();

    // j(u) in eigenbasis in units of R [fs^-1]
    MatrixXcd je = v.adjoint() * js * v;
    return {w, je};
}

// double calcCao(double gamma, double J = 300)
// {
//     const int N = 4000;
//     auto eig = calcEigsJe(N, J);
//     VectorXd &w = eig.first;
//     MatrixXcd &je = eig.second;

//     // Diffusion constant in units of R^2 [fs^-1]
//     double D = 0;

//     for (int mu = 0; mu < N; ++mu) {
//         for (int nu = 0; nu < N; ++nu) {
//             double omega = w[mu] - w[nu];
//             D += hbar_cm1_fs * std::norm(je(mu, nu)) * gamma /
//                  (std::pow(gamma, 2) + std::pow(omega, 2));
//         }
//     }
//     D /= N;

//     // std::cout << "gamma = " << gamma << ", D = " << D << '\n';
//     return D;
// }

double calcCaoPrecomputed(double gamma, VectorXd &w, MatrixXcd &je)
{
    if (w.rows() != je.rows() or je.rows() != je.cols())
        throw std::runtime_error("matrix dimensions must match!");

    long N = w.rows();

    // Diffusion constant in units of R^2 [fs^-1]
    double D = 0;

    for (int mu = 0; mu < N; ++mu) {
        for (int nu = 0; nu < N; ++nu) {
            double omega = w[mu] - w[nu];
            D += hbar_cm1_fs * std::norm(je(mu, nu)) * gamma /
                 (std::pow(gamma, 2) + std::pow(omega, 2));
        }
    }
    D /= static_cast<double>(N);

    // std::cout << "gamma = " << gamma << ", D = " << D << '\n';
    return D;
}

ArrayXd calcCaoFuncRealisation(std::vector<double> const &gamma, double J,
                               double sigma, int N, RandomGenerator rnd)
{
    VectorXd w;
    MatrixXcd je;
    ArrayXd result(gamma.size());

    std::tie(w, je) = calcEigsJe(N, J, sigma, rnd);

    for (size_t idx = 0, num = gamma.size(); idx != num; ++idx) {
        result[static_cast<long>(idx)] =
            calcCaoPrecomputed(gamma[idx] * J, w, je);
    }
    return result;
}

int main(int argc, char *argv[])
{
    CmdArgs cmdargs = processCmdArguments(argc, argv);
    std::vector<double> input = logspace(cmdargs.min, cmdargs.max, cmdargs.num);
    std::vector<double> result(cmdargs.num, 0);
    size_t realisations = 100;
    int N = 1000;
    double J = 300;
    double sigma = J;

    unsigned long nThreads;
    char *penv;

    if ((penv = std::getenv("SLURM_JOB_CPUS_ON_NODE")))
        nThreads = std::stoul(penv);
    else
        nThreads = std::thread::hardware_concurrency();

    ThreadPool pool(nThreads);

    auto s = seedsFromClock(); // base seeds for random numbers
    int seed1 = s.first;
    int seed2 = s.second;

    // for (int run = 0; run < p.nRuns; ++run) {
    //     RandomGenerator rnd(seed1, seed2); // every thread gets its own seed
    //     results.push_back(pool.enqueue_task(evolve, rnd, H0, c0, xxsq, p));
    //     seed2 = (seed2 + 1) % 30081;
    // }

    std::vector<std::future<ArrayXd>> results;
    results.reserve(realisations);
    ArrayXd D = ArrayXd::Zero(static_cast<long>(cmdargs.num));

    for (size_t i = 0; i < realisations; ++i) {
        RandomGenerator rnd(seed1, seed2);
        results.push_back(
            pool.enqueue_task(calcCaoFuncRealisation, input, J, sigma, N, rnd));
        seed2 = (seed2 + 1) % 30081;
    }
    print_progress(std::cout, 0, realisations, "", "", 1, 20);
    for (int r = 0; r < realisations; ++r) {
        D += results[r].get();
        print_progress(std::cout, r + 1, realisations, "", "", 1, 20);
    }
    D /= static_cast<double>(realisations);
    // size_t nThreads;
    // char *penv;

    // if ((penv = std::getenv("SLURM_JOB_CPUS_ON_NODE")))
    //     nThreads = std::stoul(penv);
    // else
    //     nThreads = std::thread::hardware_concurrency();

    // ThreadPool pool(nThreads);

    // std::vector<double> result(cmdargs.num);
    // std::vector<double> input = linspace(cmdargs.min, cmdargs.max,
    // cmdargs.num);

    // double *resultdata = result.data();
    // double *inputdata = input.data();
    // size_t pieceSize = cmdargs.num / nThreads;
    // size_t resid = cmdargs.num % nThreads;
    // if (resid) {
    //     ++pieceSize;
    // }
    // else {
    //     resid = nThreads;
    // }

    // for (size_t tid = 0; tid != nThreads; ++tid) {
    //     if (resid == 0)
    //         --pieceSize;

    //     std::cout << "Start thread " << tid + 1 << " with ps = " << pieceSize
    //     << '\n'; pool.enqueue_work(
    //         [inputdata, resultdata, pieceSize]()
    //         {
    //             for (size_t idx = 0; idx != pieceSize; ++idx) {
    //                 resultdata[idx] = calcCao(inputdata[idx]);
    //             }
    //         });
    //     inputdata += pieceSize;
    //     resultdata += pieceSize;
    //     --resid;
    // }
    saveData("multithread.cao", D.data(), D.size(), cmdargs);
}