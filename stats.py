import math
from scipy.stats import norm

def binomial_confidence_interval(t, f, conf):
  N = t+f
  p = t / float(N)
  return binomial_confidence_interval_p(p, N, conf)

def binomial_confidence_interval_p(p, N, conf):
  return norm.ppf(1 - conf/2) * math.sqrt(p*(1-p)/N)
