import copy
import cProfile
import math
from net import Network
import numpy
from numpy import array
import random
import re
from stats import binomial_confidence_interval, binomial_confidence_interval_p

# I estimate roughly 800 games per minute when training with BMU
# and FINE_GRAIN_TURNS=25.
N = 10000
CHECKPOINT_N = 1000

verbose = 0

costs = {
  'c': 0,
  's': 3,
  'g': 6,
  'e': 2,
  'd': 5,
  'p': 8,
}

values = {
  'c': 1,
  's': 2,
  'g': 3,
}

points = {
  'e': 1,
  'd': 3,
  'p': 6,
}

ai_log_file = file('ai_log.txt', 'w')
def log_msg(msg):
  print >>ai_log_file, msg

def say(msg):
  log_msg(msg)
  if verbose:
    print msg

class PlayerState(object):
  def __init__(self, game, name):
    self.game = game
    self.name = name
    self.silent = False
    self.deck = []
    self.hand = []
    self.inplay = []
    self.buys = 1
    self.money = 0
    self.turns = 0
    self.discard = 7*['c'] + 3*['e']

    self.card_counts = {'c': 7, 'e': 3}

  def inc_card_count(self, card):
    self.card_counts[card] = self.count_card(card) + 1

  def copy(self, game, silent):
    dupe = PlayerState(game, self.name)
    dupe.silent = silent
    dupe.deck = list(self.deck)
    dupe.hand = list(self.hand)
    dupe.inplay = list(self.inplay)
    dupe.buys = self.buys
    dupe.money = self.money
    dupe.turns = self.turns
    dupe.discard = list(self.discard)
    return dupe

  def say(self, msg):
    if not self.silent:
      log_msg(msg)
      if verbose:
        print msg

  def draw(self, n):
    self.say('%s draws %d cards' % (self.name, n))
    drew = []
    for j in range(n):
      if len(self.deck) == 0:
        while self.discard:
          self.deck.append(self.discard.pop())
        if drew: self.say('... drew %s' % ''.join(drew))
        drew = []
        self.say('(%s reshuffles)' % self.name)
        random.shuffle(self.deck)

      drew.append(self.deck[-1])
      self.hand.append(self.deck.pop())
    if drew: self.say('... drew %s' % ''.join(drew))

  def play(self, card):
    if card not in self.hand: return False
    if card not in values: return False
    self.say('%s plays a %s' % (self.name, card))
    self.inplay.append(card)
    self.hand.remove(card)

    self.money += values[card]

    return True

  def gain(self, card):
    if not self.game.supply[card]:
      self.say('%s gains nothing' % self.name)
      return False

    self.say('%s gains a %s' % (self.name, card))
    self.discard.append(card)
    self.inc_card_count(card)
    self.game.supply[card] -= 1
    return True

  def buy(self, card):
    if self.buys == 0: return False
    if self.money < costs[card]: return False
    self.say('%s buys a %s' % (self.name, card))

    if not self.gain(card): return False
    self.buys -= 1
    self.money -= costs[card]
    return True

  def cleanup(self):
    self.say('%s cleans up' % self.name)
    self.buys = 1
    self.money = 0
    self.turns += 1
    while self.hand:
      self.discard.append(self.hand.pop())
    while self.inplay:
      self.discard.append(self.inplay.pop())

  def get_all_cards(self):
    return self.deck + self.hand + self.inplay + self.discard

  def get_vp(self):
    vp = 0
    for card in self.get_all_cards():
      if card in points:
        vp += points[card]
    return vp

  def count_card(self, card):
    return self.card_counts.get(card, 0)

  def card_density(self, card):
    return self.count_card(card) / float(self.total_cards_in_deck())

  def total_money_in_deck(self):
    return sum(values.get(card, 0) for card in self.get_all_cards())

  def total_cards_in_deck(self):
    return len(self.get_all_cards())

  def format_deck(self):
    counts = {}
    for c in self.get_all_cards():
      counts[c] = counts.get(c,0) + 1

    out = []
    for c in 'pdegsc':
      if c in counts:
        out.append('%d%s' % (counts[c], c))
    return ', '.join(out)

class Player(object):
  def __init__(self, idx, strategy):
    self.idx = idx
    self.strategy = strategy
    self.turn = 0

    self.experimented = True
    self.prev_features = None

  def update_model(self, game):
    self.strategy.update_model(self, game)
    self.turn += 1

  def train(self, features=None, outcome=None):
    self.strategy.train(self, features=features, outcome=outcome)

  def buy(self, game):
    self.strategy.buy(self, game)

class PlayerStrategy(object):
  def __init__(self):
    self.name = self.__class__.__name__

  def update_model(self, game):
    pass

  def train(self, features=None, outcome=None):
    pass

  def create_player(self, idx):
    return Player(idx, self)

  def buy(self, player, game):
    pass

from model import *

# We have 71 features. It seems reasonable to at least put the number of hidden
# nodes at a similar magnitude.
NHIDDEN = 60
class LearningPlayerStrategy(PlayerStrategy):
  def __init__(self):
    PlayerStrategy.__init__(self)
    self.net = Network(FEATURE_COUNT, NHIDDEN)

  def update_model(self, player, game):
    # Update model.
    features = self.extract_features(player, game)
    self.train(player, features=features)
    player.prev_features = features

  def evaluate_features(self, features):
    return self.net.calc(features)

  def evaluate(self, player, game):
    return self.evaluate_features(self.extract_features(player, game))

  def extract_features(self, player, game):
    ret = numpy.array([0.0]*FEATURE_COUNT)

    NPF = len(PLAYER_FEATURES)
    NSF = len(SUPPLY_FEATURES)
    for i,f in enumerate(PLAYER_FEATURES):
      ret[i] = f.player_extract(game, player.idx)
      ret[NPF+i] = f.player_extract(game, 1-player.idx)
    for i,f in enumerate(SUPPLY_FEATURES):
      ret[2*NPF+i] = f.supply_extract(game)
    ret[2*NPF+NSF] = 1.0

    return ret

  def train(self, player, features=None, outcome=None):
    global sumsqerr
    global nsamples

    if player.prev_features is None:
      say('  Player %d skipping model training on first turn.' % player.idx)
      return
    if player.experimented:
      say('  Player %d skipping model training due to experiment.' % player.idx)
      return

    if outcome is None:
      assert features is not None
      outcome = 2*self.net.calc(features) - 1

    if features is not None: mode = 'difference'
    else: mode = 'final'
    say('  Player %d training %s:' % (player.idx, mode))

    x_t = player.prev_features
    target = (outcome + 1) / 2.0
    prediction = self.net.calc(x_t)
    say('    target = %.5lf' % target)
    say('    prediction = %.5lf' % prediction)

    err = target - prediction
    sumsqerr += err**2
    nsamples += 1

    self.net.train(x_t, target, alpha)
    new_prediction = self.net.calc(x_t)
    say('    new prediction = %.5lf' % new_prediction)

class BasicBigMoney(PlayerStrategy):
  def buy(self, player, game):
    ps = game.states[player.idx]
    ps.buy('p')
    ps.buy('g')
    ps.buy('s')

class BigMoneyUltimate(LearningPlayerStrategy):
  def __init__(self):
    LearningPlayerStrategy.__init__(self)

  def buy(self, player, game):
    ps = game.states[player.idx]
    if ps.total_money_in_deck() > 18:
      ps.buy('p')
    if ps.game.supply['p'] <= 4:
      ps.buy('d')
    if ps.game.supply['p'] <= 2:
      ps.buy('e')
    ps.buy('g')
    if ps.game.supply['p'] <= 6:
      ps.buy('d')
    ps.buy('s')

    player.experimented = False

sumsqerr = 0.0
nsamples = 0
msqerrf = file('msqerr.txt', 'w')

msqerr_list = []
ROLLING = 100

def show_learn_data(strategy):
  global sumsqerr
  global nsamples
  msqerr = sumsqerr / nsamples
  msqerr_list.append(msqerr)
  sumsqerr = 0.0
  nsamples = 0

  if len(msqerr_list) >= ROLLING:
    rolling = sum(msqerr_list[-ROLLING:]) / ROLLING
  else:
    rolling = 0.0
  print '==> %s (msq err = %.5lf, rolling = %.5lf)' % (strategy.name, msqerr, rolling)
  print >>msqerrf, msqerr

class SimpleAI(LearningPlayerStrategy):
  def __init__(self):
    LearningPlayerStrategy.__init__(self)

  def buy(self, player, game):
    ps = game.states[player.idx]
    buy_options = ['']
    for c in costs:
      if costs[c] <= ps.money and game.supply[c] > 0:
        buy_options.append(c)

    if random.uniform(0,1) > experiment_p:
      best = -1e999
      bestc = 'x'
      for c in buy_options:
        future = game.copy(True)
        ps = future.states[player.idx]
        if c:
          ps.buy(c)
        ps.cleanup()
        now = self.evaluate(player, future)
        say('  Buying "%s" gives me value %.8lf' % (c,now))
        if now > best:
          best = now
          bestc = c
      assert bestc != 'x'
      player.experimented = False
    else:
      # Don't propagate results of experiments backwards!
      player.experimented = True
      bestc = random.choice(buy_options)
      say('  Experimenting.')

    if bestc:
      ps = game.states[player.idx]
      ps.buy(bestc)

TURN_LIMIT = 50
class Game(object):
  def __init__(self, parent=None, silent=False):
    self.silent = silent
    if parent:
      self.strategies = parent.strategies
      self.states = [ps.copy(self, silent) for ps in parent.states]
      self.players = [p for p in parent.players]
      self.supply = copy.copy(parent.supply)
    else:
      self.strategies = STRATEGIES
      random.shuffle(self.strategies)

      self.states = []
      self.players = []
      for i,s in enumerate(self.strategies):
        self.states.append(PlayerState(self, '%s (player %d)' % (s.name, i+1)))
        self.players.append(s.create_player(i))

      self.supply = {
        'c': 60,
        's': 40,
        'g': 30,
        'e': 12,
        'd': 8,
        'p': 8,
      }

  def copy(self, silent):
    return Game(self, silent)

  def say(self, msg):
    if not self.silent:
      log_msg(msg)
      if verbose:
        print msg

  def play(self):
    # Pick up initial hands.
    self.say('***** GAME START')
    for ps in self.states:
      ps.draw(5)

    done = False
    for turn in range(TURN_LIMIT):
      for ps,player in zip(self.states, self.players):
        self.say('')
        self.say('*** %s\'s turn %d' % (ps.name, turn))

        while ps.play('c'): pass
        while ps.play('s'): pass
        while ps.play('g'): pass
        player.buy(self)

        # Clean up hand and draw a new hand.
        ps.cleanup()
        player.update_model(self)
        ps.draw(5)

        self.say('Provinces remaining in supply: %d' % self.supply['p'])
        if self.supply['p'] == 0:
          self.say('*** All provinces gone!')
          done = True

        empty_piles = []
        for c,cnt in self.supply.items():
          if cnt == 0:
            empty_piles.append(c)
        if len(empty_piles) >= 3:
          self.say('*** %s piles gone!' % empty_piles)
          done = True

        if done: break
      if done: break

    self.say('')
    self.say('*** GAME OVER')
    for ps in self.states:
      self.say('%s had %d points: %s.' % (ps.name, ps.get_vp(), ps.format_deck()))
    st1,st2 = self.states
    p1,p2 = self.players
    if st1.get_vp() > st2.get_vp():
      self.say('%s wins!' % st1.name)
      p1.train(outcome=1)
      p2.train(outcome=-1)
      return [st1.name]
    elif st2.get_vp() > st1.get_vp() or (st2.get_vp() == st1.get_vp() and st2.turns < st1.turns):
      self.say('%s wins!' % st2.name)
      p1.train(outcome=-1)
      p2.train(outcome=1)
      return [st2.name]
    else:
      self.say('%s and %s rejoice in their shared victory!' % (st1.name, st2.name))
      p1.train(outcome=0)
      p2.train(outcome=0)
      return [st1.name, st2.name]

alpha = 0.0
experiment_p = 0.0

strategy = SimpleAI()

STRATEGIES = [strategy, strategy]

def main():
  global alpha
  global experiment_p

  wins = {}
  for i in range(N):
    log_msg('******* Game %d/%d' % (i,N))
    # In the old 4-feature model,
    # alpha = 0.5 approaches a reasonable solution quickly. Larger values
    # tend to diverge.
    alpha = 0.02
    experiment_p = 0.1
    game = Game()
    winners = game.play()
    print 'Game %d' % i,
    show_learn_data(strategy)
    print '  Winner %s' % winners
    winners = tuple(winners)
    wins[winners] = wins.get(winners,0) + 1

    if (i+1)%CHECKPOINT_N == 0:
      strategy.net.dump(file('net.%d.dump' % (i+1), 'w'))
  print

  for w,n in wins.items():
    print '%3d: %s' % (n,w)
  print
  
  basic_wins = {}
  for w,n in wins.items():
    if len(w) == 1:
      name = re.match('([^ ]*)', w[0]).group(1)
    else:
      name = 'Tie'
    basic_wins[name] = basic_wins.get(name, 0) + n
  assert sum(basic_wins.values()) == N

  conf = 0.01
  print 'All intervals with confidence %.1lf%%' % (100.0 * (1-conf))
  for w,n in basic_wins.items():
    p = n/float(N)
    ivl = binomial_confidence_interval_p(p, N, conf)
    print '  %s: %.1lf%% +/- %.1lf%%' % (w, 100.0 * p, 100.0 * ivl)

main()
#cProfile.run('main()')
