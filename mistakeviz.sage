# vim: set filetype=python :

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy
from numpy import array
import os

print 'Reading rolling mistakes...'

xs = []
ys = []
ROLLING = 100

for i,line in enumerate(file('rolling_mistakes.txt')):
  if i%ROLLING == 0:
    xs.append(ROLLING+i)
    ys.append(eval(line))

print 'Plotting...'
plt.plot(xs,ys)
plt.savefig('mistakeviz.png')
os.system('open mistakeviz.png')
