# vim: set filetype=python :

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy
from numpy import array
import os
import sys

if len(sys.argv) != 2:
  print 'Usage: sage mistakeviz.sage mistake_rate.txt'
  sys.exit(1)

print 'Reading mistake rates...'
mistake_rates = []
for line in file(sys.argv[1]):
  mistake_rates.append(eval(line))

print 'Computing rolling averages...'
xs = []
ys = []
ROLLING = 1000
mistake_sum = 0.0
msamples = 0

for i in range(len(mistake_rates)):
  mistake_sum += mistake_rates[i]
  if i-ROLLING >= 0:
    mistake_sum -= mistake_rates[i-ROLLING]
  else:
    msamples += 1

  if (i+1)%ROLLING == 0:
    xs.append(i)
    ys.append(mistake_sum / msamples)

print 'Plotting...'
plt.plot(xs,ys)
plt.savefig('mistakeviz.png')
os.system('open mistakeviz.png')
