# vim: set filetype=python :

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy
from numpy import array
import os

w = eval(file('offline_weights.txt').read())

from model import *

xs = []
yss = [[] for i in range(PER_TURN_FEATURES)]
my_silver = []
my_gold = []
for t in range(FINE_GRAIN_TURNS+1):
  xs.append(t)
  for i in range(PER_TURN_FEATURES):
    yss[i].append(w[t*PER_TURN_FEATURES+i])

colours = [
  (192, 192, 192),
  (255, 215, 0),
  (0, 192, 0),
  (0, 128, 0),
  (0, 64, 0),

  (192, 192, 192),
  (255, 215, 0),
  (0, 192, 0),
  (0, 128, 0),
  (0, 64, 0),

  (0,0,128),
]

styles = ['-']*5 + ['--']*5 + ['-']

def proc_colour(c):
  r,g,b = c
  return (r/255.0,g/255.0,b/255.0)

for i in range(PER_TURN_FEATURES):
  plt.plot(xs,yss[i],styles[i],color=proc_colour(colours[i]))
plt.savefig('weightviz.png')
os.system('open weightviz.png')
