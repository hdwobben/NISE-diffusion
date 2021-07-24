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
nTimeSteps = dataset['nTimeSteps']
dt = dataset['dt']

# <<(x(t) - x0)^2>> in units of R^2
if 'msd' in dataset:
    msd = np.frombuffer(dataset["msd"])
else:
    msd = np.frombuffer(dataset["data"])


t = np.arange(nTimeSteps) * dt
fig, ax = plt.subplots(1,1)
ax.set_xlabel('t (fs)')
ax.set_ylabel(r'$\langle [ (x - x_0)^2 ] \rangle$ in units of $R^2$')
ax.plot(t, msd)
# uncomment below to plot a line at the converging value of <<(x(t) - x0)^2>>
# ax.autoscale(False) 
# ax.plot(t, [(dataset['N']**2 - 1) / 12] * nTimeSteps, "--")
plt.show()