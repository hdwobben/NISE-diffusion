#include <NISE/utils.hpp>
#include <iomanip>
#include <cmath>

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

    os << "\33[2K\r" << prefix << " (" << iter << '/' << total << ") |" 
       << bar << "| "
       << std::fixed << std::setprecision(decimals)
       << percents << '%' << ' ' << suffix;//comment

    if (iter == total)
        os << '\n';
    
    os << std::flush;
}