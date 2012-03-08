import math
import random
from scipy.stats import norm, poisson, nbinom

def make_nbinom(mu, sigmasq):
  p = 1.0 - mu/sigmasq
  r = mu * (1.0-p) / p
  return nbinom(r,1-p)

HAND_SIZE = 5
MAX_CASH = 5*3

# Starting deck.
#deck = 7*[1] + 3*[0]

# A deck representative of mid-game BMU.
deck = 7*[1] + 4*[2] + 4*[3] + 3*[0]

# A deck representative of end-of-game BMU.
#deck = 7*[1] + 4*[2] + 4*[3] + 11*[0]


TRIALS = 1000
stats = [0]*(MAX_CASH+1)
for i in range(TRIALS):
  deck2 = list(deck)

  res = 0
  for j in range(HAND_SIZE):
    c = random.choice(deck2)
    deck2.remove(c)
    res += c
  for k in range(res+1):
    stats[k] += 1
for i in range(len(stats)):
  stats[i] /= float(TRIALS)


money_density = sum(deck) / float(len(deck))
mu = 5.0 * money_density
sigma = math.sqrt(5.0 * (1 - 1/float(len(deck))) / float(len(deck))
                  * sum(c*c for c in deck))
model_norm = norm(mu, sigma)
model_poisson = poisson(mu)
model_nbinom = make_nbinom(mu, sigma**2)

f = file('money_density_experiment.dump', 'w')

print '    Actual Normal Poisson NBinom'
for C in range(MAX_CASH+1):
  norm_prediction = 1.0 - model_norm.cdf(C)
  poisson_prediction = 1.0 - model_poisson.cdf(C-1)
  nbinom_prediction = 1.0 - model_nbinom.cdf(C-1)
  print '%2d: %6.4f %6.4f %7.4f %6.4f' % (
    C, stats[C], norm_prediction, poisson_prediction, nbinom_prediction)
  print >>f, (C, stats[C], norm_prediction, poisson_prediction, nbinom_prediction)
