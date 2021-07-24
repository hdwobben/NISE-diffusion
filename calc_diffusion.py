import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
import msgpack
import sys
import os

plt.style.use('science')

# hbar / (hc * 100 cm/m) * 1e15 fs/s [cm^-1 fs]
hbar_cm1_fs = 1 / (2 * np.pi * 299792458 * 100) * 1e15

def calcDiffAnalytic(N, J, sig, lam):
    gamma = sig**2 / (lam * hbar_cm1_fs) # Bath hom. line width [cm^-1]

    # Constant part of the Hamiltonian (site basis) in cm^-1
    H0 = np.diag([J] * (N - 1), -1) + np.diag([J] * (N - 1), 1) 

    # j in matrix form (site basis) in units of R [fs^-1]
    j_s = ( np.diag([-1j * J / hbar_cm1_fs] * (N - 1), -1) + 
            np.diag([1j * J / hbar_cm1_fs] * (N - 1), 1) )

    w, v = sp.linalg.eigh(H0)

    j_e = v.conj().T @ j_s @ v # j in eigenbasis in units of R [fs^-1]

    D = 0 # Diffusion constant in units of R^2 [fs^-1]
    for mu in range(N):
        for nu in range(N):
            omega = w[mu] - w[nu]
            D += ( gamma * hbar_cm1_fs / (gamma**2 + omega**2) * 
                   np.abs(j_e[mu, nu])**2 )

    D /= N
    return D

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = input("Enter a file to read: ")

if len(sys.argv) > 2:
    cutoff = float(sys.argv[2])
else:
    cutoff = 99999

with open(fname, "rb") as f:
    binData = f.read()

if fname == "out.tmp":
    os.remove(fname)

dataset = msgpack.unpackb(binData)

N = dataset['N']                     # Chain length
J = dataset['J']                     # Coupling constant [cm^-1]
sig = dataset['sig']                 # Stddev of energy fluctuations [cm^-1]
lam = dataset['lam']                 # Interaction rate 1/T [fs^-1]
dt = dataset['dt']                   # Time step [fs]
nTimeSteps = dataset['nTimeSteps']   # Final time = nTimeSteps * dt

print("N = {}".format(N))
print("J = {} cm^-1".format(J))
print("lam = {} fs^-1".format(lam))
print("sig = {} cm^-1".format(sig))
print("nRuns = {}".format(dataset['nRuns']))
print("nTimeSteps = {}".format(nTimeSteps))
print("dt = {} fs\n".format(dt))

# <<(x(t) - x0)^2>> in units of R^2
if 'msd' in dataset:
    msd = np.frombuffer(dataset["msd"])
else:
    msd = np.frombuffer(dataset["data"])

t = np.arange(dataset['nTimeSteps']) * dataset['dt'] # time [fs]

def msdf(t, a, b):
    """Model function for <(x(t)-x0)^2>.
    a = tau_b = m/gamma
    b = 2 * k_b * T / m
    """
    return b * a * (t - a * (1 - np.exp(-t / a)))

print("cutoff = {} fs".format(cutoff))
Da = calcDiffAnalytic(N, J, sig, lam)
popt, pcov = sp.optimize.curve_fit(msdf, t[t<=cutoff], msd[t<=cutoff], p0=(np.sqrt(Da), np.sqrt(Da)),
                                   maxfev=6000)
a, b = popt
print("D/R^2 =", a*b/2, "fs^-1")
# print("analytic: D/R^2 =", Da, "fs^-1")
fig, axs = plt.subplots(1,2, sharex = True, sharey = True, figsize=(12.75, 5.7))
plt.subplots_adjust(top=0.88, bottom=0.11, left=0.11, right=0.9, hspace=0.2,
                    wspace=0.245)
fig.suptitle(r'$J = {:g}\ cm^{{-1}},\ \sigma = {:g}J,\ '
             r'\Lambda = {:g}\ fs^{{-1}},\ N = {},\ dt = {}\ fs$'.format(J, sig/J, lam, N, dt))
ax1 = axs[0]
ax1.plot(t, msd, 'r', label='NISE')
ax1.grid()
ax1.legend()
ax1.set_ylabel(r'$\langle [ (x - x_0)^2 ] \rangle\ /\ R^2$')
ax1.set_xlabel('t (fs)')

ax2 = axs[1]
ax2.plot(t, msdf(t, *popt), label=r'$ab(t - a(1 - \exp(-\frac{t}{a})))$ fit')
ax2.plot(t, msd, 'r--', label='NISE')
ax2.autoscale(False)
ax2.plot(t, a*b*t - a*a*b , 'g-.', label = r'$abt - a^2b$')
ax2.set_xlabel('t (fs)')
ax2.set_ylabel(r'$\langle [ (x - x_0)^2 ] \rangle\ /\ R^2$')
ax2.legend()
ax2.grid()
plt.show()
