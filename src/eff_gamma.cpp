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
    VectorXd Hf = VectorXd::Zero(p.N, p.N);
    for (int i = 0; i < p.N; ++i) // Prepare the starting disorder
        Hf[i] = rnd.RandomGaussian(0, p.sig);
    H.diagonal() = Hf;

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(p.N);

    ArrayXd meanM2 = ArrayXd::Zero(2 * p.N);
    Eigen::Map<ArrayXd> mean(meanM2.data(), p.N);

    // sum of squared distance from the mean
    Eigen::Map<ArrayXd> m2(meanM2.data() + p.N, p.N);

    for (int ti = 0; ti < p.nTimeSteps; ++ti) {
        solver.computeFromTridiagonal(H.diagonal(), H.diagonal(-1));
        VectorXd const &w = solver.eigenvalues();
        for (int ni = 0; ni < p.N; ++ni) {
            // Welford's algorithm for mean and variance
            double d1 = w[ni] - mean[ni];
            mean[ni] += d1 / (ti + 1);
            double d2 = w[ni] - mean[ni];
            m2[ni] += d1 * d2;
        }

        updateTrajectory(Hf, rnd, p);
        H.diagonal() = Hf;
    }
    return meanM2;
}

void process(std::string const &fname)
{
    Params p = loadParams(fname);
    // p.nRuns = 100;
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

    for (int run = 0; run < p.nRuns; ++run) {
        RandomGenerator rnd(seed1, seed2); // every thread gets its own seed
        results.push_back(pool.enqueue_task(evolve, rnd, H0, p));
        seed2 = (seed2 + 1) % 30081;
    }

    // mean and summed squared distance to mean of eigenstates
    ArrayXd meanVar = ArrayXd::Zero(2 * p.N);
    Eigen::Map<ArrayXd> mean(meanVar.data(), p.N);
    Eigen::Map<ArrayXd> m2(meanVar.data() + p.N, p.N);

    int count = 0;
    print_progress(std::cout, 0, p.nRuns, "", "", 1, 20);
    for (int run = 0; run < p.nRuns; ++run) {
        ArrayXd imv = results[run].get();
        Eigen::Map<ArrayXd> imean(imv.data(), p.N);
        Eigen::Map<ArrayXd> im2(imv.data() + p.N, p.N);

        int newCount = count + p.nTimeSteps;

        for (int ni = 0; ni < p.N; ++ni) // Chan et al. parallel algorithm
        {
            double delta = imean[ni] - mean[ni];
            if (static_cast<double>(count) / p.nTimeSteps > 0.5 and
                p.nTimeSteps > 500000) {
                mean[ni] =
                    (count * mean[ni] + p.nTimeSteps * imean[ni]) / newCount;
            }
            else
                mean[ni] += delta * p.nTimeSteps / newCount;

            m2[ni] += im2[ni] + delta * delta * count * p.nTimeSteps / newCount;
        }
        count = newCount;
        print_progress(std::cout, run + 1, p.nRuns, "", "", 1, 20);
    }
    m2 /= count - 1; // m2 now holds the sample variance

    double avgVar = 0;
    for (int ni = 1; ni < p.N - 1; ++ni)
        avgVar += (m2[ni] - avgVar) / ni;

    std::cout << "Convolved mean = " << mean.sum() << '\n';
    std::cout << "Convolved var = " << m2.sum() << '\n';
    std::cout << "Avg var = " << avgVar << '\n';

    std::cout << "Gamma / J = " << p.sig * p.sig / (hbar_cm1_fs * p.lam * p.J)
              << '\n'
              << "Gamma_eff / J = " << avgVar / (hbar_cm1_fs * p.lam * p.J)
              << '\n';

    // std::string outFname = fname.substr(0, fname.find_last_of('.')) + ".mv";

    // saveData(outFname, meanVar.data(), meanVar.size(), p);
}

#include <filesystem>
namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    for (auto &entry: fs::directory_iterator(fs::current_path())) {
        if (not entry.is_regular_file() or entry.path().extension() != ".json")
            continue;

        process(entry.path().c_str());
        std::cout << entry.path() << '\n';
    }
}