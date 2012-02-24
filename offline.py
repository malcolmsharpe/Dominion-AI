import math
import model
import numpy
from numpy import array
from scipy.linalg import solve,lstsq
import sys

if len(sys.argv) != 2:
  print 'Usage: python offline.py lsq_data.txt'
  print 'Find offline optimal weights.'
  sys.exit(1)

f = file(sys.argv[1])
A, b, soln = map(numpy.array, eval(f.read()))
iterative_solution = numpy.array(soln)

# There may be linear dependencies among features.
# This means A will be singular, which makes scipy.solve upset.
# Also, it's difficult to say which solution to the linear system we want.
# However, the lstsq routine seems to do okay.
# It uses the gelss routine from LAPACK, which finds a solution of
# minimum norm.
# See: http://www.netlib.org/lapack/lug/node27.html
w,residues,rank,sigma = lstsq(A, b)

print '# features = %d' % model.FEATURE_COUNT
print 'rank = %d' % rank
print 'total residue = %.8lf' % sum(residues)
print 'sigma = %s' % sigma
print 'Best: ', w
print 'Found: ', iterative_solution
print >>file('offline_weights.txt', 'w'), repr(w)
