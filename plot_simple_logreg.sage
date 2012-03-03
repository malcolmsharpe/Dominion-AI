# vim: set filetype=python :

print 'Importing...'

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
  print 'Usage: sage plot_simple_logreg.sage simple_logreg.dump'
  sys.exit(1)

print 'Reading data...'
ss = []
ys = []
zs = []
for line in file(sys.argv[1]):
  s,beta = eval(line)
  ss.append(s)
  ys.append(1.0/beta[0])
  zs.append(1.0/beta[1])

print 'Plotting...'
plt.plot(ss,ys)
plt.plot(ss,zs)

plt.savefig('plot_simple_logreg.png')
os.system('open plot_simple_logreg.png')
