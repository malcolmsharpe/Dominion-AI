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
  print 'Usage: sage plot_simple.sage simple_dp.dump'
  sys.exit(1)

print 'Reading data...'
ss = []
ds = []
vs = []
for line in file(sys.argv[1]):
  s,d,v = eval(line)
  ss.append(s)
  ds.append(d)
  vs.append(v)

logreg_dump = file('simple_logreg.dump', 'w')
def plot_data(s0):
  print '*** Iteration for s0=%d' % s0
  xs = []
  ys = []
  for s,d,v in zip(ss,ds,vs):
    if s == s0 and -10 <= d <= 10:
      xs.append(d)
      ys.append(v)

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

  beta = good_method()
  print >>logreg_dump, '%d,%r' % (s0,beta)
  ws = apply_logistic_model(beta)

  print 'Plotting...'
  #plt.plot(xs,ys)
  #plt.plot(xs,zs)
  #plt.plot(xs,ws)

for s0 in range(2,40):
  plot_data(s0)
#plot_data(20)

plt.savefig('plot_simple.png')
os.system('open plot_simple.png')
