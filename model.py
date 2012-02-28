# Current model:
#   For each player and each type of card, a feature for "is there at least
#   this many" for each small number, and for larger numbers, a linear feature.
#
#   Also:
#   - a feature which is 1 for 2nd player and 0 for 1st player
#   - a bias feature which is always 1
#
#   The reason we are tracking copper in this version is that we are again
#   attempting to train by self-play.

INF = 100

PLAYER_FEATURES = []

class CountFeature(object):
  def player_extract(self, game, idx):
    ps = game.states[idx]
    return self.extract(ps.count_card(self.card))

  def supply_extract(self, game):
    return self.extract(game.supply[self.card])

class SmallCountFeature(CountFeature):
  def __init__(self, card, count):
    self.card = card
    self.count = count

  def extract(self, count):
    return float(count >= self.count)

class LargeCountFeature(CountFeature):
  def __init__(self, card, lo, hi, scale):
    self.card = card
    self.lo = lo
    self.hi = hi
    self.scale = scale

  def extract(self, count):
    n = max(self.lo, min(self.hi, count)) - self.lo
    return float(n * self.scale)

class SecondPlayerFeature(object):
  def player_extract(self, game, idx):
    return float(idx)

# For treasure, sometimes the AI's like to buy absurd amounts.
# So the features should handle this decently--if a player has
# half of any kind of treasure, the features should still max
# at around 1.
PLAYER_FEATURES.append(SmallCountFeature('g', 1))
PLAYER_FEATURES.append(SmallCountFeature('g', 2))
PLAYER_FEATURES.append(SmallCountFeature('g', 3))
PLAYER_FEATURES.append(SmallCountFeature('g', 4))
PLAYER_FEATURES.append(LargeCountFeature('g', 4, 6, 0.5))
PLAYER_FEATURES.append(LargeCountFeature('g', 6, INF, 0.1))

PLAYER_FEATURES.append(SmallCountFeature('s', 1))
PLAYER_FEATURES.append(SmallCountFeature('s', 2))
PLAYER_FEATURES.append(LargeCountFeature('s', 2, 4, 0.5))
PLAYER_FEATURES.append(LargeCountFeature('s', 4, 6, 0.5))
PLAYER_FEATURES.append(LargeCountFeature('s', 6, 8, 0.5))
PLAYER_FEATURES.append(LargeCountFeature('s', 8, INF, 0.1))

PLAYER_FEATURES.append(SmallCountFeature('c', 8))
PLAYER_FEATURES.append(LargeCountFeature('c', 8, 10, 0.5))
PLAYER_FEATURES.append(LargeCountFeature('c', 10, INF, 0.05))

PLAYER_FEATURES.append(SmallCountFeature('p', 1))
PLAYER_FEATURES.append(SmallCountFeature('p', 2))
PLAYER_FEATURES.append(SmallCountFeature('p', 3))
PLAYER_FEATURES.append(SmallCountFeature('p', 4))
PLAYER_FEATURES.append(LargeCountFeature('p', 4, INF, 0.5))

PLAYER_FEATURES.append(SmallCountFeature('d', 1))
PLAYER_FEATURES.append(SmallCountFeature('d', 2))
PLAYER_FEATURES.append(SmallCountFeature('d', 3))
PLAYER_FEATURES.append(SmallCountFeature('d', 4))
PLAYER_FEATURES.append(LargeCountFeature('d', 4, INF, 0.5))

PLAYER_FEATURES.append(SmallCountFeature('e', 4))
PLAYER_FEATURES.append(SmallCountFeature('e', 5))
PLAYER_FEATURES.append(SmallCountFeature('e', 6))
PLAYER_FEATURES.append(SmallCountFeature('e', 7))
PLAYER_FEATURES.append(LargeCountFeature('e', 7, INF, 0.5))

PLAYER_FEATURES.append(SecondPlayerFeature())


SUPPLY_FEATURES = []

SUPPLY_FEATURES.append(SmallCountFeature('p', 1))
SUPPLY_FEATURES.append(SmallCountFeature('p', 2))
SUPPLY_FEATURES.append(SmallCountFeature('p', 3))
SUPPLY_FEATURES.append(SmallCountFeature('p', 4))
SUPPLY_FEATURES.append(LargeCountFeature('p', 4, 6, 0.5))
SUPPLY_FEATURES.append(LargeCountFeature('p', 6, 8, 0.5))

SUPPLY_FEATURES.append(SmallCountFeature('d', 1))
SUPPLY_FEATURES.append(SmallCountFeature('d', 2))
SUPPLY_FEATURES.append(LargeCountFeature('d', 2, 4, 0.5))
SUPPLY_FEATURES.append(LargeCountFeature('d', 4, 8, 0.25))

SUPPLY_FEATURES.append(SmallCountFeature('e', 1))
SUPPLY_FEATURES.append(SmallCountFeature('e', 2))
SUPPLY_FEATURES.append(LargeCountFeature('e', 2, 4, 0.5))
SUPPLY_FEATURES.append(LargeCountFeature('e', 4, 8, 0.25))


FEATURE_COUNT = 2*len(PLAYER_FEATURES) + len(SUPPLY_FEATURES) + 1


def player_feature_name(i):
  return str(PLAYER_FEATURES[i])

def supply_feature_name(i):
  return str(SUPPLY_FEATURES[i])

def feature_name(i):
  if i < len(PLAYER_FEATURES):
    return 'My %s' % player_feature_name(i)
  i -= len(PLAYER_FEATURES)

  if i < len(PLAYER_FEATURES):
    return 'Opponent %s' % player_feature_name(i)
  i -= len(PLAYER_FEATURES)

  if i < len(SUPPLY_FEATURES):
    return 'Supply %s' % supply_feature_name(i)
  i -= len(SUPPLY_FEATURES)

  if i < 1:
    return 'Bias'
  i -= 1

  assert False
