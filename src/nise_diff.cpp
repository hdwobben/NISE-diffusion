#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <getopt.h>
#include <nlohmann/json.hpp>
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

using json = nlohmann::json;
using namespace std::complex_literals;

// hbar / (hc * 100 cm/m) * 1e15 fs/s [cm^-1 fs]
const double hbar_cm1_fs = 1 / (2 * M_PI * 299792458.0 * 100) * 1e15;

struct Params
{
    int N = 51;                 // Chain length
    double J = 30;              // Coupling constant [cm^-1]
    double sig = 2 * J;         // Stddev of energy fluctuations [cm^-1]
    double lam = 0.002;         // Interaction rate 1/T [fs^-1]
    double dt = 1;              // Time step [fs]
    int nTimeSteps = 600;       // Final time = nTimeSteps * dt
    int nRuns = 100;            // Number of runs to average over
};

struct CmdArgs
{
    std::string paramsFName;
    std::string outFName;
    bool quiet = false;
    bool outFile = true;
};

CmdArgs processCmdArguments(int argc, char *argv[])
{
    struct option const long_options[] =
    {
        {"no-outfile", 0, NULL, 'x'}, // to allow --no-outfile 
        {NULL, 0, NULL, 0}
    };

    CmdArgs cargs;

    int ch;
    int option_index;
    while ((ch = getopt_long(argc, argv, "p:o:qx", 
                                long_options, &option_index)) != -1)
    {
        switch (static_cast<char>(ch))
        {
            case 'p':
                cargs.paramsFName = optarg;
                break;
            case 'o':
                cargs.outFName = optarg;
                break;
            case 'q':
                cargs.quiet = true;
                break;
            case 'x':
                cargs.outFile = false;
        }
    }

    if (cargs.paramsFName.empty())
    {
        if (cargs.quiet)
            cargs.paramsFName = "params.json";
        else
        {
            std::cout << "Enter a parameter file name: " << '\n';
            std::cin >> cargs.paramsFName;
        }
    }
    
    return cargs;
}

Params loadParams(std::string const &fname)
{
    std::ifstream in(fname, std::ios::in | std::ios::binary);
    if (in)
    {
        std::string contents;
        in.seekg(0, std::ios::end);
        contents.resize(in.tellg());
        in.seekg(0, std::ios::beg);
        in.read(&contents[0], contents.size());
        in.close();
        json pdata = json::parse(contents);
        Params prms;
        
        prms.N = pdata["N"];
        prms.J = pdata["J"];
        prms.lam = pdata["lam"];
        prms.sig = pdata["sig/J"].get<double>() * prms.J;
        prms.nRuns = pdata["nRuns"];
        prms.nTimeSteps = pdata["nTimeSteps"];
        prms.dt = pdata["dt"];
        
        return prms;
    }
    throw std::runtime_error(std::string("Error opening file: ") + 
                             std::strerror(errno));
}

std::ostream &operator<<(std::ostream &os, Params const &p)
{
    os << "N = " << p.N << '\n'
       << "J = " << p.J << " cm^-1\n"
       << "lam = " << p.lam << " fs^-1\n"
       << "sig = " << p.sig << " cm^-1\n"
       << "nRuns = " << p.nRuns << '\n'
       << "nTimeSteps = " << p.nTimeSteps << '\n'
       << "dt = " << p.dt << " cm^-1";
    return os;
}

void updateTrajectory(MatrixXd &Hf, RandomGenerator &rnd, Params const &p)
{
    ArrayXd r{p.N};
    double A = std::sqrt(1 - std::exp(-2 * p.dt * p.lam));

    for (int i = 0; i < p.N; ++i)
        r[i] = A * rnd.RandomGaussian(0, p.sig);

    Hf.diagonal() *= std::exp(-p.lam * p.dt);
    Hf.diagonal().array() += r;
}

ArrayXd evolve(RandomGenerator rnd, MatrixXd const &H0, MatrixXd const &Hf0,
               VectorXcd const &c0, RowVectorXd const &xxsq, Params const &p)
{
    MatrixXd H;
    MatrixXd Hf = Hf0;
    VectorXcd c = c0;
    ArrayXd msdi{p.nTimeSteps};
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(p.N);

    for (int ti = 0; ti < p.nTimeSteps; ++ti)
    {
        // < (x(t) - x(0))^2 > (expectation value)
        msdi[ti] = xxsq * c.cwiseAbs2();
        updateTrajectory(Hf, rnd, p);
        H = H0 + Hf;
        solver.computeFromTridiagonal(H.diagonal(), H.diagonal(-1) );
        VectorXcd L = 
            (solver.eigenvalues().array() * -1i * p.dt / hbar_cm1_fs).exp();
        MatrixXd const &U = solver.eigenvectors();
        c = U * L.asDiagonal() * U.transpose() * c;
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

    // Fluctuating part of the Hamiltonian (site basis) in cm^-1
    MatrixXd Hf0 = MatrixXd::Zero(p.N, p.N);
    
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
            pool.enqueue(evolve, rnd, H0, Hf0, c0, xxsq, p));
        seed2 = (seed2 + 1) % 30081;
    }

    // <<(x(t) - x0)^2>> in units of R^2
    ArrayXd msd = ArrayXd::Zero(p.nTimeSteps);

    for (int run = 0; run < p.nRuns; ++run)
    {
        if (not cmdargs.quiet)
            print_progress(std::cout, run + 1, p.nRuns, "", "", 1, 20);
        msd += results[run].get();
    }
    msd /= p.nRuns;

    if (not cmdargs.outFile)
        return 0;

    json::binary_t binMsd;
    {
        // prepare msd array as binary data
        std::vector<uint8_t> binData(
            reinterpret_cast<uint8_t *>(msd.data()), 
            reinterpret_cast<uint8_t *>(msd.data()) + 
                                        msd.size() * sizeof(double) );

        binMsd = json::binary_t(std::move(binData));
    }

    json dataset;    
    dataset["N"] = p.N;
    dataset["J"] = p.J;
    dataset["lam"] = p.lam;
    dataset["sig"] = p.sig;
    dataset["nRuns"] = p.nRuns;
    dataset["nTimeSteps"] = p.nTimeSteps;
    dataset["dt"] = p.dt;
    dataset["msd"] = binMsd;

    std::vector<uint8_t> msg = json::to_msgpack(dataset);

    if (cmdargs.outFName.empty() )
        cmdargs.outFName = nowStrLocal("%Y%m%d%H%M%S.msgpack");

    std::ofstream file(cmdargs.outFName, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<char *>(msg.data()), msg.size());
    
    if (cmdargs.outFName != "out.tmp")
        std::cout << cmdargs.outFName << '\n';
}