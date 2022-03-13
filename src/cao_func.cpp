#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <tuple>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

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

std::pair<VectorXd, MatrixXcd> calcEigsJe(int N = 4000, double J = 300)
{
    // Constant part of the Hamiltonian (site basis) in cm^-1
    MatrixXd H0 = MatrixXd::Zero(N, N);
    H0.diagonal(1) = VectorXd::Constant(N - 1, J);
    H0.diagonal(-1) = VectorXd::Constant(N - 1, J);

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

double calcCao(double gamma, double J = 300)
{
    const int N = 4000;
    auto eig = calcEigsJe(N, J);
    VectorXd &w = eig.first;
    MatrixXcd &je = eig.second;

    // Diffusion constant in units of R^2 [fs^-1]
    double D = 0;

    for (int mu = 0; mu < N; ++mu) {
        for (int nu = 0; nu < N; ++nu) {
            double omega = w[mu] - w[nu];
            D += hbar_cm1_fs * std::norm(je(mu, nu)) * gamma /
                 (std::pow(gamma, 2) + std::pow(omega, 2));
        }
    }
    D /= N;

    // std::cout << "gamma = " << gamma << ", D = " << D << '\n';
    return D;
}

double calcCaoPrecomputed(double gamma, VectorXd &w, MatrixXcd &je)
{
    if (w.rows() != je.rows() or je.rows() != je.cols() )
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

int main(int argc, char *argv[])
{
    CmdArgs cmdargs = processCmdArguments(argc, argv);
    std::vector<double> input = linspace(cmdargs.min, cmdargs.max, cmdargs.num);
    std::vector<double> result(cmdargs.num);
    VectorXd w;
    MatrixXcd je;
    
    std::tie(w, je) = calcEigsJe(4000, 150);

    for (size_t idx = 0; idx != cmdargs.num; ++idx) {
        result[idx] = calcCaoPrecomputed(input[idx], w, je);
    }
    // size_t nThreads;
    // char *penv;

    // if ((penv = std::getenv("SLURM_JOB_CPUS_ON_NODE")))
    //     nThreads = std::stoul(penv);
    // else
    //     nThreads = std::thread::hardware_concurrency();

    // ThreadPool pool(nThreads);

    // std::vector<double> result(cmdargs.num);
    // std::vector<double> input = linspace(cmdargs.min, cmdargs.max, cmdargs.num);

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

    //     std::cout << "Start thread " << tid + 1 << " with ps = " << pieceSize << '\n';
    //     pool.enqueue_work(
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
    saveData("out.cao", result.data(), result.size(), cmdargs);
}