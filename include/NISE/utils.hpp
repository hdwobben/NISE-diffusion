#ifndef NISE_UTILS_H
#define NISE_UTILS_H

#include <iostream>
#include <string>
#include <vector>

struct Params {
    int N = 51; // Chain length
    double J = 30; // Coupling constant [cm^-1]
    double sig = 2 * J; // Stddev of energy fluctuations [cm^-1]
    double lam = 0.002; // Interaction rate 1/T [fs^-1]
    double dt = 1; // Time step [fs]
    int nTimeSteps = 600; // Final time = nTimeSteps * dt
    int nRuns = 100; // Number of runs to average over
};

// Command line arguments for the programs
struct CmdArgs {
    std::string paramsFName;
    std::string outFName;
    bool quiet = false;
    bool outFile = true;
};

// Parse options from the command line
CmdArgs processCmdArguments(int argc, char *argv[]);

// Load parameters from a json file
Params loadParams(std::string const &fname);

// Convenience function for printing parameters
std::ostream &operator<<(std::ostream &os, Params const &p);

void saveData(std::string const &fname, std::vector<uint8_t> data,
              Params const &p);

// Save data to a msgpack file along with the parameters
template <class T>
void saveData(std::string const &fname, T *data, size_t size, Params const &p)
{
    // prepare input as binary data
    std::vector<uint8_t> binData(reinterpret_cast<uint8_t *>(data),
                                 reinterpret_cast<uint8_t *>(data) +
                                     size * sizeof(T));

    saveData(fname, std::move(binData), p);
}

// Call in a loop to create terminal progress bar
// @params:
//     os          - Required  : ostream to output to
//     iter        - Required  : current iteration
//     total       - Required  : total iterations
//     prefix      - Optional  : prefix string
//     suffix      - Optional  : suffix string
//     decimals    - Optional  : positive number of decimals in percent complete
//     barLength   - Optional  : character length of bar
void print_progress(std::ostream &os, int iter, int total,
                    std::string const &prefix = "",
                    std::string const &suffix = "", int decimals = 1,
                    int barLength = 40);

// get the current local timestamp as string
std::string nowStrLocal(std::string const &fmt = "%Y-%m-%d %H:%M:%S");

#endif // NISE_UTILS_H