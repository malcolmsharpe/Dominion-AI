# Experiment with techniques on a (very) simple model of the Dominion endgame.
# The model is:
# - There are S=8 provinces in the supply.
# - Each turn, the current player has a fixed prob. p of buying a province.
# - A player wins when ending with > as many provinces as opponent _on his
#   turn_. (So tie points means a loss for the player that ends it! This means
#   that the first player to S/2=4 provinces wins.)
#
# This model admits an exact DP solution.

from logistic_regression import logistic_regression
import math
import numpy
from numpy import array
import random
from scipy.linalg import lstsq

def compute_dp(S,p):
  # dp[s,d]
  #   where
  #  s = # provinces in supply
  #  d = delta of my provinces vs opponent provinces
  dp = {}

  for d in range(-S,S+1):
    dp[0,d] = int(d >= 0)

  for s in range(1,S+1):
    taken = S-s
    for d in range(-taken,taken+1):
      dp[s,d] = (1/(2-p)) * (1 - dp[s-1,-(d+1)] + (1-p)*dp[s-1,d-1])

  return dp

assert compute_dp(8,1.0)[8,0] == 1.0

class AdaptiveAlpha(object):
  # This algorithm doesn't seem to converge well, but I think the idea makes
  # sense.
  #
  # The first idea is to adjust the learning rate up when we see two
  # consecutive differences with opposite sign, and down when they have the
  # same sign.
  #
  # The second idea is to do this in a way that (on average) does not change
  # the learning rate at the optimal solution. To be more precise, we want
  # the expectation of the adjustment exponent to be zero. Obviously we
  # actually want the learning rate to decrease at the optimal solution, so
  # a bias is added to the exponent to cause it to tend negative.
  #
  # So, how do we get E[|adjust|] = 0 at the optimal solution?
  # For simplicity, assume the optimal solution is fixed.
  # (This is approximately true if the learning rate is very small.)
  # 
  # As a first attempt, for consecutive observations a and b, let
  #   adjust = diff_a * diff_b.
  # Note that adjust > 0 when two consecutive differences have the same sign
  # and adjust < 0 when the sign differs, as we desired.
  # Since we assume a Markov process, diff_a and diff_b are independent, so
  #   E[adjust] = E[diff_a * diff_b]
  #             = E[diff_a] * E[diff_b]
  #             = E[diff]^2.
  # But since the optimal solution is an average, E[diff] = 0.
  #
  # Now the question is how to reasonably normalize "adjust".
  # We would like E[|normed adjust|] = 1.
  # To achieve this, observe that
  #   E[|adjust|] = E[|diff_a| |diff_b|] = E[|diff|]^2.
  # So we can normalize by dividing by E[|diff|]^2.
  # The difficulty is that we don't know what E[|diff|] is at the optimal
  # solution. So, estimate it based on recent observed absolute differences.
  #
  # Maybe an exponential scale-down is the wrong way to go.
  # Let's try a harmonic scale-down instead.
  # Indeed that gets OK results, although it might just be because alpha is
  # effectively just decreasing harmonically.

  DIFF_ALPHA = 0.01
  ADJUST_BIAS = -0.7
  ADJUST_COEFF = 1.0
  N0 = 1.0

  def __init__(self):
    self.est_abs_diff = 1.0
    self.prev_diff = None
    self.scale_down = 0

  def receive_diff(self, diff):
    self.est_abs_diff += self.DIFF_ALPHA * (abs(diff) - self.est_abs_diff)

    # If we don't do this, we get problems with constants.
    self.est_abs_diff = max(1e-2, self.est_abs_diff)
    
    if self.prev_diff is not None:
      diff_prod = diff * self.prev_diff
      norm_diff_prod = diff_prod / (self.est_abs_diff**2)
      adjust = self.ADJUST_COEFF * (self.ADJUST_BIAS + norm_diff_prod)
      self.scale_down -= adjust

      # Try to avoid crazy values.
      self.scale_down = max(0.0, self.scale_down)

    self.prev_diff = diff

  def get_alpha(self):
    return self.N0 / (self.N0 + self.scale_down)

def compute_td_table(S,p,ngames,incr=None,entry=None):
  print S,p,ngames,incr,entry
  states = []
  for s in range(S+1):
    for d in range(-S,S+1):
      states.append((s,d))
  rev_states = dict((st,i) for i,st in enumerate(states))

  seen = set()

  value = numpy.array([0.0] * len(states))

  for g in range(ngames):
    # lamb=0 seems best. i.e. Pure TD.
    lamb = 0.0
    # Tweaking a harmonic learning rate seems best.
    # It can get results comparable to the batch method. (Really?)
    # For lamb=0, setting N0 to roughly the number of states seems to work well
    # if large numbers of games are used.
    N0 = float(S)**2
    alpha = max(N0 / (N0 + g), 1e-4)
    if incr is not None and (g+1)%incr==0:
      assert entry is not None
      print ('  TD game %d: '
             'value[%s] = %.4lf, '
             'alpha = %.4lf') % (
        g,
        entry,value[rev_states[entry]],
        alpha)

    s = S
    d = 0

    elig = numpy.array([0.0] * len(states))

    while 1:
      prev = s,d
      won = None
      for j in range(2):
        if s and random.uniform(0,1) < p:
          d += (-1)**j
          s -= 1
        if s==0:
          if d>0: won = True
          elif d<0: won = False
          else: won = (j == 1)
          break

      # Train.
      if won is None:
        now = s,d
        target = value[rev_states[now]]
      else:
        target = int(won)

      diff = target-value[rev_states[prev]]

      elig = lamb*elig
      elig[rev_states[prev]] += 1.0
      value += alpha * diff * elig
      seen.add(prev)

      if won is not None:
        break

  ret = {}
  for i,st in enumerate(states):
    if st in seen:
      ret[st] = value[i]
  return ret

def gen_table_training_data(S,p,ngames):
  trans = {}
  rhs = {}

  for g in range(ngames):
    s = S
    d = 0

    while 1:
      prev = s,d
      won = None
      for j in range(2):
        if s and random.uniform(0,1) < p:
          d += (-1)**j
          s -= 1
        if s==0:
          if d>0: won = True
          elif d<0: won = False
          else: won = (j == 1)
          break

      # Store training data.
      if won is None:
        now = s,d
        trans[prev,prev] = trans.get((prev,prev),0.0) + 1.0
        trans[prev,now] = trans.get((prev,now),0.0) - 1.0
        rhs[prev] = rhs.get(prev,0.0) + 0.0
      else:
        trans[prev,prev] = trans.get((prev,prev),0.0) + 1.0
        rhs[prev] = rhs.get(prev,0.0) + float(won)

      if won is not None:
        break

  return trans, rhs

def batch_td_table(S,p,ngames):
  # Use least-squares TD algorithm to fill a TD table.
  # (In other words, solve exactly using observed probabilities.)

  trans,rhs = gen_table_training_data(S,p,ngames)

  states = list(rhs)
  A = numpy.array([[0.0]*len(states)]*len(states))
  b = numpy.array([0.0]*len(states))

  for i,r in enumerate(states):
    for j,c in enumerate(states):
      A[i,j] = trans.get((r,c), 0.0)
    b[i] = rhs.get(r, 0.0)

  x,resid,rank,sigma = lstsq(A,b)

  # value[s,d]
  value = {}

  for i,r in enumerate(states):
    value[r] = x[i]

  return value

def gen_training_data(S,p,ngames):
  records = []

  for g in range(ngames):
    s = S
    d = 0

    while 1:
      prev = s,d
      won = None
      for j in range(2):
        if s and random.uniform(0,1) < p:
          d += (-1)**j
          s -= 1
        if s==0:
          if d>0: won = True
          elif d<0: won = False
          else: won = (j == 1)
          break

      # Store training data.
      if won is None:
        now = s,d
        records.append((prev,now))
      else:
        records.append((prev,won))

      if won is not None:
        break

  return records

def extract_features(st):
  s,d = st
  return array([1.0/s, d, d/float(s)])

def sigmoid(z):
  return 1.0 / (1.0 + math.exp(-z))

def model_td_table(S,p,ngames):
  # Use batch TD algorithm with a logistic model.
  # Model terms: (1,) 1/s, d, d/s.
  # I guess we need to iterate the logistic regression a few times?

  records = gen_training_data(S,p,ngames)
  n = 3
  m = len(records)

  states = []
  X = array([[0.0]*m]*n)
  for i,(prev,_) in enumerate(records):
    X[:,i] = extract_features(prev)
    states.append(prev)

  ITERS = 20
  theta = array([0.0]*(n+1))
  for it in range(ITERS):
    y = array([0.0]*m)
    for i,(_,outcome) in enumerate(records):
      if isinstance(outcome,tuple):
        outcome = sigmoid(theta[0] + theta[1:].dot(extract_features(outcome)))
      y[i] = outcome

    theta,J_bar,l = logistic_regression(X,y,theta)

    print '  >>> theta = %s' % theta

  # value[s,d]
  value = {}

  for r in states:
    prediction = sigmoid(theta[0] + theta[1:].dot(extract_features(r)))
    value[r] = prediction

  return value

def incr_model_td_table(S,p,ngames,incr=None):
  # Use incremental TD algorithm, lambda=0, with a logistic model.
  # Model terms: (1,) 1/s, d, d/s.

  alpha = 0.1

  records = gen_training_data(S,p,ngames)
  n = 3
  m = len(records)

  theta = array([0.0]*(n+1))

  g = 0
  states = []
  x = array([1.0]*(n+1))
  for i,(prev,outcome) in enumerate(records):
    x[1:] = extract_features(prev)
    if isinstance(outcome,tuple):
      outcome = sigmoid(theta[0] + theta[1:].dot(extract_features(outcome)))
    else:
      g += 1
      if incr is not None and g%incr == 0:
        print '  >>> theta = %s' % theta

    theta = theta + alpha * (outcome - sigmoid(theta.dot(x))) * x
    states.append(prev)

  print '  >>> theta = %s' % theta

  # value[s,d]
  value = {}

  for r in states:
    prediction = sigmoid(theta[0] + theta[1:].dot(extract_features(r)))
    value[r] = prediction

  return value

def main():
  S = 8
  N = 10
  for i in range(N):
    p = (i+1)/float(N)
    dp = compute_dp(S,p)
    print 'p=%.2lf => %.4lf' % (p, dp[S,0])

  NGAMES = 1000
  INCR = 100
  S = 8
  p = 0.5
  entry = (S,0)
  dp = compute_dp(S,p)

  if 1:
    print '*** Compare DP and incremental TD table.'
    print 'dp[%s] = %.4lf' % (entry, dp[entry])
    td_table = compute_td_table(S,p,NGAMES,incr=INCR,entry=entry)
    print

  if 0:
    print '*** Compare DP and batch TD table.'
    # This shows that maybe the incremental procedure is partly hampered by
    # not enough trials to get the desired precision.
    # In fact, the current adaptive alpha seems to be doing about as well.
    print 'dp[%s] = %.4lf' % (entry, dp[entry])
    td_table = batch_td_table(S,p,NGAMES)
    print 'td_table[%s] = %.4lf' % (entry, td_table[entry])
    print

  if 1:
    print '*** Compare DP and batch model TD table'
    print 'dp[%s] = %.4lf' % (entry, dp[entry])
    td_table = model_td_table(S,p,NGAMES)
    entry = (S,0)
    print 'td_table[%s] = %.4lf (dp = %.4lf)' % (entry, td_table[entry], dp[entry])
    entry = (6,2)
    print 'td_table[%s] = %.4lf (dp = %.4lf)' % (entry, td_table[entry], dp[entry])
    entry = (5,1)
    print 'td_table[%s] = %.4lf (dp = %.4lf)' % (entry, td_table[entry], dp[entry])
    entry = (3,1)
    print 'td_table[%s] = %.4lf (dp = %.4lf)' % (entry, td_table[entry], dp[entry])
    entry = (1,1)
    print 'td_table[%s] = %.4lf (dp = %.4lf)' % (entry, td_table[entry], dp[entry])
    print

  if 1:
    print '*** Compare DP and incremental model TD table'
    print 'dp[%s] = %.4lf' % (entry, dp[entry])
    td_table = incr_model_td_table(S,p,NGAMES,incr=INCR)
    entry = (S,0)
    print 'td_table[%s] = %.4lf (dp = %.4lf)' % (entry, td_table[entry], dp[entry])
    entry = (6,2)
    print 'td_table[%s] = %.4lf (dp = %.4lf)' % (entry, td_table[entry], dp[entry])
    entry = (5,1)
    print 'td_table[%s] = %.4lf (dp = %.4lf)' % (entry, td_table[entry], dp[entry])
    entry = (3,1)
    print 'td_table[%s] = %.4lf (dp = %.4lf)' % (entry, td_table[entry], dp[entry])
    entry = (1,1)
    print 'td_table[%s] = %.4lf (dp = %.4lf)' % (entry, td_table[entry], dp[entry])
    print

  # Data for plotting.

  # Output exact data to plot.
  f = file('simple_dp.dump', 'w')
  S = 40
  p = 0.5
  dp = compute_dp(S,p)
  for s in range(S+1):
    for d in range(-(S-s), S-s+1):
      print >>f, '%d,%d,%r' % (s, d, dp[s,d])

main()
