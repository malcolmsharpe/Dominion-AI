# Neural network for predicting values in [0,1].
import math
import numpy
import random

def phi(x):
  return 1. / (1 + numpy.exp(-x))

class Network(object):
  def __init__(self, ninput, nhidden):
    self.m = ninput
    self.n = nhidden
    self.a = numpy.array([0.] * self.n)

    # It's necessary to break symmetry at some point. Otherwise, the update
    # rule will leave all coordinates of a, b, and w equal. So, introduce
    # a bit of randomness to the initial input weights. It turns out that this
    # was sufficient to get good performance on the XOR example.
    self.w = numpy.array([[random.gauss(0,1) for j in range(self.m)] for i in range(self.n)])

  def _get_hidden(self, x):
    # Element-wise application of phi.
    return phi(self.w.dot(x))

  def calc(self, x):
    return phi(self.a.dot(self._get_hidden(x)))

  def train(self, x, y, mu):
    Fx = self.calc(x)
    z = self._get_hidden(x)

    adj = mu * (y - Fx) * Fx * (1-Fx)
    
    u = numpy.matrix(self.a * z * (1.0-z)).transpose()
    v = numpy.matrix(x).transpose()
    self.w += adj * u * v.transpose()
    self.a += adj * z

  def prn(self):
    print 'a = %s' % self.a
    print 'w = %s' % self.w

  def dump(self, f):
    print >>f, 'Neural network dump.'
    print >>f, 'm = %s' % self.m
    print >>f, 'n = %s' % self.n
    print >>f, 'a = %r' % self.a.tolist()
    print >>f, 'w = %r' % self.w.tolist()

def xor_example():
  def xor(a,b):
    return a != b

  # This version of the neural network, where a sigmoid function is applied
  # to the output, does not do as well for XOR as the version that applies
  # a linear combination to the output of the hidden nodes.
  #
  # That is probably fine, as when we are predicting probabilities, they
  # will be quite a bit farther from 0 and 1.
  net = Network(3, 4)
  mu = 1.0

  verbose = 0
  semiverbose = 1

  errsum = 0.
  errN = 100

  N = 20000
  endview = 5
  for i in range(N):
    a,b = [random.randrange(2) for j in range(2)]
    x = numpy.array([1.0,a,b])
    y = xor(a,b)
    z = net.calc(x)
    err = abs(y-z)
    errsum += err
    if verbose or N-i <= endview:
      print '%d^%d: predicts %.2lf for %d; error %.2lf' % (a,b,z,y,err)
    net.train(x,y,mu)

    if semiverbose and (i+1)%errN == 0:
      print 'avg err = %.2lf' % (errsum / errN)
      errsum = 0.

  net.prn()

if __name__ == '__main__':
  xor_example()
