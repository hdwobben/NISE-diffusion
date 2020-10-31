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

#include <cmath>
#include <chrono>
#include <bitset>
#include <limits>

namespace chrono = std::chrono;

RandomGenerator::RandomGenerator(int ij, int kl)
{
    double s, t;
    int i, j, k, l, m;

	// Handle the seed range errors
	// First random number seed must be between 0 and 31328
	// Second seed must have a value between 0 and 30081   
    if (ij < 0 or ij > 31328) 
        ij = 1802;

    if (kl < 0 or kl > 30081)
        kl = 9373;

    i = (ij / 177) % 177 + 2;
    j = (ij % 177)       + 2;
    k = (kl / 169) % 178 + 1;
    l = (kl % 169);

    for (int ii = 0; ii < 97; ++ii) 
    {
        s = 0.0;
        t = 0.5;
        for (int jj = 0; jj < 24; ++jj)
        {
            m = (((i * j) % 179) * k) % 179;
            i = j;
            j = k;
            k = m;
            l = (53 * l + 1) % 169;
            if (((l * m % 64)) >= 32)
                s += t;
            t *= 0.5;
        }
        _u[ii] = s;
    }

    _c   = 362436.0 / 16777216.0;
    _cd  = 7654321.0 / 16777216.0;
    _cm  = 16777213.0 / 16777216.0;
    _i97 = 97;
    _j97 = 33;
}

double RandomGenerator::RandomUniform()
{
    double uni;

    uni = _u[_i97 - 1] - _u[_j97 - 1];
    if (uni <= 0.0)
        ++uni;
    _u[_i97 - 1] = uni;
    --_i97;
    if (_i97 == 0)
        _i97 = 97;
    --_j97;
    if (_j97 == 0)
        _j97 = 97;
    _c -= _cd;
    if (_c < 0.0)
        _c += _cm;
    uni -= _c;
    if (uni < 0.0)
        uni++;
 
    return uni;
}

double RandomGenerator::RandomGaussian(double mean, double stddev)
{
    double q, u, v, x, y;

	// Generate P = (u,v) uniform in rect. enclosing acceptance region 
	// Make sure that any random numbers <= 0 are rejected, since
	// gaussian() requires uniforms > 0, but RandomUniform() delivers >= 0.
    do 
    {
	    u = RandomUniform();
	    v = RandomUniform();
        if (u <= 0.0 or v <= 0.0) 
        {
            u = 1.0;
            v = 1.0;
        }
	    v = 1.7156 * (v - 0.5);

	    // Evaluate the quadratic form
	    x = u - 0.449871;
	    y = std::fabs(v) + 0.386595;
	    q = x * x + y * (0.19600 * y - 0.25472 * x);

	    // Accept P if inside inner ellipse
	    if (q < 0.27597)
			break;

	    // Reject P if outside outer ellipse, or outside acceptance region
	} while ((q > 0.27846) or (v * v > -4.0 * std::log(u) * u * u));

	// Return ratio of P's coordinates as the normal deviate
	return (mean + stddev * v / u);
}

int RandomGenerator::RandomInt(int lower, int upper)
{
   return static_cast<int>(RandomUniform() * (upper - lower + 1)) + lower;
}

double RandomGenerator::RandomDouble(double lower, double upper)
{
   return (upper - lower) * RandomUniform() + lower;
}

std::pair<int, int> seedsFromClock()
{
    chrono::time_point<chrono::system_clock> nowTp = 
        chrono::system_clock::now();
    
    unsigned long long now = 
        static_cast<unsigned long long>(
            chrono::duration_cast<chrono::microseconds>(
                nowTp.time_since_epoch()).count());
    
    std::bitset<64> nowBits{now};
    std::bitset<16> num1;
    std::bitset<16> num2;

    for (int pos = 0; pos < 16; pos += 2)
    {
        num1[pos] = nowBits[pos];
        num2[pos + 1] = nowBits[pos + 1];
    }

    uint16_t max_u16 = std::numeric_limits<uint16_t>::max();
    uint16_t res1 = static_cast<uint16_t>(num1.to_ulong());
    uint16_t res2 = static_cast<uint16_t>(num2.to_ulong());

    return { static_cast<double>(res1) / max_u16 * 31328,
             static_cast<double>(res2) / max_u16 * 30081 };
}

#include <vector>
#include <fstream>

int main()
{
    size_t size = 500 * 600 * 111;
    int seed1 = 1;
    int seed2 = 1;

    std::vector<double> rands;
    rands.reserve(size);

    for (int run = 0; run < 500; ++run)
    {
        RandomGenerator rnd(seed1, seed2);

        for (int ti = 0; ti < 600 * 111; ++ti)
            rands.push_back(rnd.RandomGaussian(0, 2 * 5.650954701926560e-03) );

        seed2 = (seed2 + 1) % 30081;
    }

    std::ofstream file("rands.dat", std::ios::out | std::ios::binary);
    file.write(
        reinterpret_cast<char *>(rands.data()), 
                                 rands.size() * sizeof(double) / sizeof(char));
}