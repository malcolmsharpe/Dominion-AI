# vim: set filetype=python :

import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy
from numpy import array
import os
from scipy.linalg import lstsq
import sys

if len(sys.argv) != 2:
  print 'Usage: sage plot_money_density_experiment.sage money_density_experiment.dump'
  sys.exit(1)

print 'Reading data...'
xs = []
ys = []
zs1 = []
zs2 = []
zs3 = []
for line in file(sys.argv[1]):
  x,y,z1,z2,z3 = eval(line)
  xs.append(x)
  ys.append(y)
  zs1.append(z1)
  zs2.append(z2)
  zs3.append(z3)

print 'Plotting...'
plt.plot(xs,ys)
plt.plot(xs,zs1)
plt.plot(xs,zs2)
plt.plot(xs,zs3)

plt.savefig('plot_money_density_experiment.png')
os.system('open plot_money_density_experiment.png')
