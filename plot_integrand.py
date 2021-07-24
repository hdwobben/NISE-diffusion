import numpy as np
import matplotlib.pyplot as plt
import msgpack
import sys
import os

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = input("Enter a file to read: ")

with open(fname, "rb") as f:
    bytes = f.read()

if fname == "out.tmp":
    os.remove(fname)
    
dataset = msgpack.unpackb(bytes)
dt = dataset['dt']
N = dataset['N']                     # Chain length
J = dataset['J']                     # Coupling constant [cm^-1]
sig = dataset['sig']                 # Stddev of energy fluctuations [cm^-1]
lam = dataset['lam']                 # Interaction rate 1/T [fs^-1]
dt = dataset['dt']                   # Time step [fs]
nTimeSteps = dataset['nTimeSteps']   # Final time = nTimeSteps * dt
# <Tr(j(u,t)j(u))> in units of R^2 [fs^-2]
if "integrand" in dataset:
    integrand = np.frombuffer(dataset["integrand"], dtype='complex128')
else:
    integrand = np.frombuffer(dataset["data"], dtype='complex128')

t = np.arange(nTimeSteps) * dt

fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.8))
plt.subplots_adjust(top=0.88, bottom=0.11, left=0.11, right=0.9, hspace=0.2, 
                    wspace=0.245)

fig.suptitle(r'$J = {:g}\ cm^{{-1}},\ \sigma = {:g}J,\ '
             r'\Lambda = {:g}\ fs^{{-1}},\ N = {},\ dt = {:g}\ fs$'.format(J, sig/J, lam, N, dt))
ax1 = axs[0]
ax1.plot(t, integrand.real, label=r'$\operatorname{Re}$')
ax1.grid()
ax1.set_ylabel(r'$\langle \operatorname{Tr}(j(u,t)j(u)) \rangle\ /\ R^2$ '
               r'(fs$^{-2}$)')
ax1.set_xlabel('t (fs)')
ax1.legend()

ax2 = axs[1]
ax2.plot(t, integrand.imag, label=r'$\operatorname{Im}$')
ax2.grid()
ax2.set_ylabel(r'$\langle \operatorname{Tr}(j(u,t)j(u)) \rangle\ /\ R^2$ '
               r'(fs$^{-2}$)')
ax2.set_xlabel('t (fs)')
ax2.legend()
plt.show()