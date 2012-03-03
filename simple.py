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
  for g in range(ngames):
    alpha = 10.0 / (10+g)

    if incr is not None and (g+1)%incr==0:
      assert entry is not None
      print '  TD game %d: value[%s] = %.4lf' % (g,entry,value.get(entry,0))

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

      if won is None:
        now = s,d
        target = value.get(now, 0)
      else:
        target = int(won)

      old = value.get(prev,0)

      value[prev] = old + alpha*(target-old)

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
  compute_td_table(S,p,20000,incr=2000,entry=entry)

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
