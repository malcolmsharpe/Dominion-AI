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
  print 'Usage: sage plot_simple.sage simple.dump'
  sys.exit(1)

print 'Reading data...'
xs = []
ys = []
for line in file(sys.argv[1]):
  x,y = eval(line)
  xs.append(x)
  ys.append(y)

print 'Fitting a linear least-squares model...'
A = array([[1.0,x] for x in xs])
b = array(ys)
cs, resid, rank, sigma = lstsq(A,b)

zs = A.dot(cs).tolist()

print 'Fitting a logistic model...'

def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))

def good_method():
  print '  using a good method...'
  from logistic_regression import simple_logistic_regression
  beta, J_bar, l = simple_logistic_regression(array(xs), array(ys))
  return beta

def bad_method():
  print '  using a bad method...'

  alpha = 0.2
  beta = array([0.0, 0.0])

  NITER = 200
  INCR = 10
  for i in range(NITER):
    if (i+1)%INCR == 0:
      print '    Step %d: beta = %s' % (i+1, beta)
    for x,y in zip(xs,ys):
      inputs = array([1.0, x])
      beta += alpha * (y - sigmoid(beta.dot(inputs))) * inputs

  return beta

def apply_logistic_model(beta):
  print '  beta = %s' % beta
  ws = []
  for x in xs:
    ws.append(sigmoid(beta[0] + beta[1]*x))
  return ws

ws = apply_logistic_model(bad_method())

print 'Plotting...'
plt.plot(xs,ys)
plt.plot(xs,zs)
plt.plot(xs,ws)
plt.savefig('plot_simple.png')
os.system('open plot_simple.png')
