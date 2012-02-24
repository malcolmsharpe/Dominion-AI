# Current model:
#   Keep different features for each turn of 1-25, and then for turns 26+.
#   The features are: # silver, # gold, # estates, # duchies, # provinces,
#                     am-i-second-player.
#   (Since BMU will not buy copper, don't include # copper.)
#   Except for the am-i-second-player bias parameter, each feature is present
#   for self and opponent.
FINE_GRAIN_TURNS = 25
PLAYER_FEATURES = 5
PER_TURN_FEATURES = 2*PLAYER_FEATURES+1
FEATURE_COUNT = (FINE_GRAIN_TURNS+1)*PER_TURN_FEATURES

def player_feature_name(i):
  return [
    '# silver',
    '# gold',
    '# estate',
    '# duchy',
    '# province',
    ][i]

def per_turn_feature_name(i):
  if i < PLAYER_FEATURES:
    return 'my %s' % player_feature_name(i)
  elif i < 2*PLAYER_FEATURES:
    return 'opponent %s' % player_feature_name(i-PLAYER_FEATURES)
  else:
    return 'second player bias'

def feature_name(i):
  turn_idx = i / PER_TURN_FEATURES
  if turn_idx < FINE_GRAIN_TURNS:
    turn_str = 'Turn %d' % (turn_idx+1)
  else:
    turn_str = 'Turn %d+' % (turn_idx+1)

  return '%s %s' % (turn_str, per_turn_feature_name(i % PER_TURN_FEATURES))
