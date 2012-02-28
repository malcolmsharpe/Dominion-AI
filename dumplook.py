from model import *
import sys

if len(sys.argv) != 2:
  print 'Usage: python dumplook.py dumpfile'
  sys.exit(1)

f = file(sys.argv[1])
f.next()

for line in f:
  exec line

R = len(w)
C = len(w[0])
for j in range(C):
  mag = 0.0
  for i in range(R):
    mag += abs(w[i][j])
  print 'Feature %03d: %.5lf (%s)' % (j, mag, feature_name(j))
