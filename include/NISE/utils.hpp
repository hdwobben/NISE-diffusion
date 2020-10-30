#ifndef NISE_UTILS_H
#define NISE_UTILS_H

#include <iostream>
#include <string>

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
                    std::string const &suffix = "",
                    int decimals = 1, int barLength = 40);

#endif // NISE_UTILS_H