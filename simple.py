# Experiment with techniques on a (very) simple model of the Dominion endgame.
# The model is:
# - There are S=8 provinces in the supply.
# - Each turn, the current player has a fixed prob. p of buying a province.
# - A player wins when ending with > as many provinces as opponent _on his
#   turn_. (So tie points means a loss for the player that ends it! This means
#   that the first player to S/2=4 provinces wins.)
#
# This model admits an exact DP solution.

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

def main():
  S = 8
  N = 10
  for i in range(N):
    p = (i+1)/float(N)
    dp = compute_dp(S,p)
    print 'p=%.2lf => %.4lf' % (p, dp[S,0])

  # Output some data to plot.
  f = file('simple.dump', 'w')
  S = 40
  p = 0.5
  dp = compute_dp(S,p)
  s = S/2
  for d in range(-s, s+1):
    print >>f, '%d,%r' % (d, dp[s,d])

main()
