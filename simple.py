# Experiment with techniques on a (very) simple model of the Dominion endgame.
# The model is:
# - There are S=8 provinces in the supply.
# - Each turn, the current player has a fixed prob. p of buying a province.
# - A player wins when ending with > as many provinces as opponent _on his
#   turn_. (So tie points means a loss for the player that ends it! This means
#   that the first player to S/2=4 provinces wins.)
#
# This model admits an exact DP solution.

import math
import random

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

  DIFF_ALPHA = 0.01
  ADJUST_BIAS = 0.0
  ADJUST_COEFF = 0.1

  def __init__(self):
    self.est_abs_diff = 1.0
    self.prev_diff = None
    self.alpha = 0.1

  def receive_diff(self, diff):
    self.est_abs_diff += self.DIFF_ALPHA * (abs(diff) - self.est_abs_diff)

    # If we don't do this, we get problems with constants.
    self.est_abs_diff = max(1e-2, self.est_abs_diff)
    
    if self.prev_diff is not None:
      diff_prod = diff * self.prev_diff
      norm_diff_prod = diff_prod / (self.est_abs_diff**2)
      adjust = self.ADJUST_COEFF * (self.ADJUST_BIAS + norm_diff_prod)
      self.alpha *= math.exp(adjust)

      # Try to avoid crazy values.
      self.alpha = min(1e-1, max(1e-8, self.alpha))

    self.prev_diff = diff

def compute_td_table(S,p,ngames,incr=None,entry=None):
  # value[s,d]
  value = {}

  # For S=2 and p=0.5, entry = (2,0):
  #   alpha = 0.1 is too high!
  #   alpha = 0.01 also too high.
  #   alpha = 0.001 gets ~1 decimal place accuracy after ~10,000 games.
  #   alpha = 0.0001 gets ~3 decimal places accuracy after 200,000 games.
  # Try varying learning rate.
  #   alpha = 100.0/(100+g) does not settle fast enough for 200,000 games.
  #   alpha = 10.0/(10+g) does better... about ~2dp consistently by 20k games.
  #   alpha = 1.0/(10+g) though does not approach fast enough.
  #   alpha = 1.0/(1+g) has the same problem.
  # Maybe choose something like 1/sqrt(g)?
  # That still satisfies the stochastic approx. conditions.
  #   alpha = 1.0 / math.sqrt(1+g) does not settle fast enough.
  # Conclusion so far:
  #   alpha = 10.0/(10+g) about the best we have tried.
  # Still the convergence is horribly horribly slow.
  #
  # Let's try an adaptive learning rate. The trick is to find a technique that
  # is guaranteed to likely decrease the learning rate over time once
  # convergence is obtained.

  alphas = {}

  for g in range(ngames):
    if incr is not None and (g+1)%incr==0:
      assert entry is not None
      if entry in value:
        print ('  TD game %d: '
               'value[%s] = %.4lf, '
               'alpha[%s] = %.4lf, '
               'est_abs_diff[%s] = %.4lf') % (
          g,
          entry,value.get(entry,0),
          entry,alphas[entry].alpha,
          entry,alphas[entry].est_abs_diff)

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

      # Train.
      if won is None:
        now = s,d
        target = value.get(now, 0)
      else:
        target = int(won)

      old = value.get(prev,0)
      diff = target-old

      if prev not in alphas:
        alphas[prev] = AdaptiveAlpha()
      alphas[prev].receive_diff(diff)

      value[prev] = old + alphas[prev].alpha * diff

      if won is not None:
        break

  return value

def main():
  S = 8
  N = 10
  for i in range(N):
    p = (i+1)/float(N)
    dp = compute_dp(S,p)
    print 'p=%.2lf => %.4lf' % (p, dp[S,0])

  # Compare DP and TD.
  S = 2
  p = 0.5
  entry = (2,0)
  dp = compute_dp(S,p)
  print 'dp[%s] = %.4lf' % (entry, dp[entry])
  compute_td_table(S,p,3000,incr=100,entry=entry)

  # Data for plotting.

  # Output exact data to plot.
  f = file('simple_dp.dump', 'w')
  S = 40
  p = 0.5
  dp = compute_dp(S,p)
  s = S/2
  for d in range(-s, s+1):
    print >>f, '%d,%r' % (d, dp[s,d])

main()
