#include <NISE/random.hpp>
#include <bitset>
#include <chrono>
#include <cmath>
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
    j = (ij % 177) + 2;
    k = (kl / 169) % 178 + 1;
    l = (kl % 169);

    for (int ii = 0; ii < 97; ++ii) {
        s = 0.0;
        t = 0.5;
        for (int jj = 0; jj < 24; ++jj) {
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

    _c = 362436.0 / 16777216.0;
    _cd = 7654321.0 / 16777216.0;
    _cm = 16777213.0 / 16777216.0;
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
    do {
        u = RandomUniform();
        v = RandomUniform();
        if (u <= 0.0 or v <= 0.0) {
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

    unsigned long long now = static_cast<unsigned long long>(
        chrono::duration_cast<chrono::microseconds>(nowTp.time_since_epoch())
            .count());

    std::bitset<64> nowBits{now};
    std::bitset<16> num1;
    std::bitset<16> num2;

    for (int pos = 0; pos < 16; pos += 2) {
        num1[pos] = nowBits[pos];
        num2[pos + 1] = nowBits[pos + 1];
    }

    uint16_t max_u16 = std::numeric_limits<uint16_t>::max();
    uint16_t res1 = static_cast<uint16_t>(num1.to_ulong());
    uint16_t res2 = static_cast<uint16_t>(num2.to_ulong());

    return {static_cast<double>(res1) / max_u16 * 31328,
            static_cast<double>(res2) / max_u16 * 30081};
}