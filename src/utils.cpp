#include <NISE/utils.hpp>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <ctime>

namespace chrono = std::chrono;

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