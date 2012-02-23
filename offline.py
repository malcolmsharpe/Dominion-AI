import math
import numpy
from numpy import array
from scipy.linalg import solve
import sys

if len(sys.argv) != 2:
  print 'Usage: python offline.py lsq_data.txt'
  print 'Find offline optimal weights.'
  sys.exit(1)

f = file(sys.argv[1])
compressed_A, compressed_b, soln = eval(f.read())
iterative_solution = numpy.array(soln)

w = solve(compressed_A, compressed_b)

print 'Best: ', w
print 'Found: ', iterative_solution
