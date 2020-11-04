#include <NISE/utils.hpp>
#include <nlohmann/json.hpp>
#include <getopt.h>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cerrno>
#include <fstream>

namespace chrono = std::chrono;
using json = nlohmann::json;

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
       << "dt = " << p.dt << " fs";
    return os;
}

void saveData(std::string const &fname, std::vector<uint8_t> data, 
              Params const &p)
{
    json::binary_t binData(std::move(data));

    json dataset;    
    dataset["N"] = p.N;
    dataset["J"] = p.J;
    dataset["lam"] = p.lam;
    dataset["sig"] = p.sig;
    dataset["nRuns"] = p.nRuns;
    dataset["nTimeSteps"] = p.nTimeSteps;
    dataset["dt"] = p.dt;
    dataset["data"] = binData;

    std::vector<uint8_t> msg = json::to_msgpack(dataset);

    std::ofstream file(fname, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<char *>(msg.data()), msg.size());
}

// Print a progress bar 
void print_progress(std::ostream &os, int iter, int total, 
                    std::string const &prefix, std::string const &suffix,
                    int decimals, int barLength)
{   
    static constexpr char barChar[] = "█";

    double percents = 100 * (iter / static_cast<double>(total));
    int filledLength = 
        static_cast<int>(
            std::round(barLength * iter / static_cast<double>(total)));

    std::string bar;
    bar.reserve(filledLength * (sizeof(barChar) - 1) + barLength);
    for (int i = 0; i < filledLength; ++i)
        bar.append("█");
    bar.append(barLength - filledLength, '-');

    os << "\33[2K\r" << prefix << "(" << iter << '/' << total << ") |" 
       << bar << "| "
       << std::fixed << std::setprecision(decimals)
       << percents << '%' << suffix;

    if (iter == total)
        os << '\n';
    
    os << std::flush;
}

// get the current local timestamp as string
std::string nowStrLocal(std::string const &fmt)
{
    chrono::time_point<chrono::system_clock> nowTp = 
        chrono::system_clock::now();

    std::ostringstream oss;
    std::time_t t = chrono::system_clock::to_time_t(nowTp);
    std::tm tmValue{*std::localtime(&t)};
    oss << std::put_time(&tmValue, fmt.c_str());
    return oss.str();
}