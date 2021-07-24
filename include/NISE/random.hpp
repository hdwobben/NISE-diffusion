#ifndef NISE_RANDOM_H
#define NISE_RANDOM_H

#include <utility>

// Using the system clock, return two numbers with ranges 
// 0 <= num1 <= 31328,
// 0 <= num2 <= 30081
std::pair<int, int> seedsFromClock();

// This random number generator originally appeared in "Toward a Universal
// Random Number Generator" by George Marsaglia and Arif Zaman.
// Florida State University Report: FSU-SCRI-87-50 (1987)
// It was later modified by F. James and published in "A Review of Pseudo-
// random Number Generators"
// THIS IS THE BEST KNOWN RANDOM NUMBER GENERATOR AVAILABLE.
// (However, a newly discovered technique can yield
// a period of 10^600. But that is still in the development stage.)
// It passes ALL of the tests for random number generators and has a period
// of 2^144, is completely portable (gives bit identical results on all
// machines with at least 24-bit mantissas in the floating point
// representation).
// The algorithm is a combination of a Fibonacci sequence (with lags of 97
// and 33, and operation "subtraction plus one, modulo one") and an
// "arithmetic sequence" (using subtraction).

// Use IJ = 1802 & KL = 9373 to test the random number generator. The
// subroutine RANMAR should be used to generate 20000 random numbers.
// Then display the next six random numbers generated multiplied by 4096*4096
// If the random number generator is working properly, the random numbers
// should be:
// 	    6533892.0  14220222.0  7275067.0
// 	    6172232.0  8354498.0   10633180.0

class RandomGenerator
{
public:
    // This is the initialization routine for the random number generator.
    // NOTE: The seed variables can have values between:  0 <= IJ <= 31328
    //                                                    0 <= KL <= 30081
    // The random number sequences created by these two seeds are of sufficient
    // length to complete an entire calculation with. For example, if sveral
    // different groups are working on different parts of the same calculation,
    // each group could be assigned its own IJ seed. This would leave each group
    // with 30000 choices for the second seed. That is to say, this random
    // number generator can create 900 million different subsequences -- with
    // each subsequence having a length of approximately 10^30. 

    // This Random Number Generator is based on the algorithm in a FORTRAN
    // version published by George Marsaglia and Arif Zaman, Florida State
    // University; ref.: see original comments below.
    // At the fhw (Fachhochschule Wiesbaden, W.Germany), Dept. of Computer
    // Science, we have written sources in further languages (C, Modula-2
    // Turbo-Pascal(3.0, 5.0), Basic and Ada) to get exactly the same test
    // results compared with the original FORTRAN version.
    // April 1989
    // Karl-L. Noell <NOELL@DWIFH1.BITNET>
    //   and  Helmut  Weber <WEBER@DWIFH1.BITNET>
    RandomGenerator(int ij, int kl);

    // Calls RandomGenerator(p.first, p.second)
    RandomGenerator(std::pair<int, int> const &p)
    :
        RandomGenerator(p.first, p.second)
    {}

    // Generates two seeds from the clock time and then uses above constructor
    RandomGenerator()
    :
        RandomGenerator(seedsFromClock() )
    {}
     
    // This is the random number generator proposed by George Marsaglia in
    // Florida State University Report: FSU-SCRI-87-50
    double RandomUniform();    

    // ALGORITHM 712, COLLECTED ALGORITHMS FROM ACM.
    // THIS WORK PUBLISHED IN TRANSACTIONS ON MATHEMATICAL SOFTWARE,
    // VOL. 18, NO. 4, DECEMBER, 1992, PP. 434-435.
    // The function returns a normally distributed pseudo-random number
    // with a given mean and standard deviation.  Calls are made to a
    // function subprogram which must return independent random
    // numbers uniform in the interval (0,1).
    // The algorithm uses the ratio of uniforms method of A.J. Kinderman
    // and J.F. Monahan augmented with quadratic bounding curves.
    double RandomGaussian(double mean, double stddev);


    // Return random integer within a range, lower -> upper INCLUSIVE
    int RandomInt(int lower,int upper);

    // Return random float within a range, lower -> upper
    double RandomDouble(double lower, double upper);

private:
    double _u[97], _c, _cd, _cm;
    int _i97, _j97;
};

#endif // NISE_RANDOM_H