import copy
import math
import numpy
import random
from stats import binomial_confidence_interval

N = 1000
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

class PlayerStrategy(object):
  def __init__(self, idx):
    self.idx = idx

  def update_model(self, game):
    pass

  def train(self, features=None, outcome=None):
    pass

FEATURE_COUNT = 4

class LearningPlayerStrategy(PlayerStrategy):
  def __init__(self, idx):
    PlayerStrategy.__init__(self, idx)
    self.experimented = True
    self.prev_features = None

  def update_model(self, game):
    # Update model.
    features = self.extract_features(game)
    self.train(features=features)
    self.prev_features = features

  def evaluate_features(self, features):
    return self.__class__.weights.dot(features)

  def evaluate(self, game):
    return self.evaluate_features(self.extract_features(game))

  def player_features(self, game, idx):
    ret = []
    ps = game.states[idx]

    # Money density / 3.0.
    total_money = ps.total_money_in_deck()
    card_count = ps.total_cards_in_deck()
    ret.append((total_money / float(card_count)) / 3.0)

    # VP / 43.0.
    ret.append(ps.get_vp() / 43.0)

    return ret

  def extract_features(self, game):
    lst = self.player_features(game, self.idx) + self.player_features(game, 1-self.idx)
    return numpy.array(lst)

  def train(self, features=None, outcome=None):
    global sumsqerr
    global nsamples

    assert features is None or outcome is None
    assert features is not None or outcome is not None

    if self.prev_features is None:
      say('  Player %d skipping model training on first turn.' % self.idx)
      return
    if self.experimented:
      say('  Player %d skipping model training due to experiment.' % self.idx)
      return

    if features is not None: mode = 'difference'
    else: mode = 'final'
    say('  Player %d training %s:' % (self.idx, mode))

    x_t = self.prev_features
    w = self.__class__.weights
    w_x_t = w.dot(x_t)

    say('    w = %s' % w)
    say('    x_t = %s' % x_t)
    say('    w_x_t = %.8lf' % w_x_t)

    gradient = -x_t

    if features is not None:
      x_t1 = features
      w_x_t1 = w.dot(x_t1)
      say('    x_t1 = %s' % x_t1)
      say('    w_x_t1 = %.8lf' % w_x_t1)
      err = w_x_t1 - w_x_t

      x_t_m = numpy.matrix(x_t).transpose()
      x_t1_m = numpy.matrix(x_t1).transpose()

      self.__class__.compressed_A += x_t_m * (x_t_m - x_t1_m).transpose()
    else:
      say('    outcome = %.8lf' % outcome)
      err = outcome - w_x_t

      x_t_m = numpy.matrix(x_t).transpose()

      self.__class__.compressed_A += x_t_m * x_t_m.transpose()
      self.__class__.compressed_b += outcome * x_t

    say('    err = %.8lf' % err)

    sumsqerr += err**2
    nsamples += 1

    adjustment = -alpha * err * gradient
    say('    adjustment = %s' % adjustment)

    self.__class__.weights = w + adjustment

    say('    w\' = %s' % self.__class__.weights)
    say('')

class BasicBigMoney(PlayerStrategy):
  def buy(self, game):
    ps = game.states[self.idx]
    ps.buy('p')
    ps.buy('g')
    ps.buy('s')

class BigMoneyUltimate(LearningPlayerStrategy):
  compressed_A = numpy.array([[0.0]*FEATURE_COUNT for i in range(FEATURE_COUNT)])
  compressed_b = numpy.array([0.0]*FEATURE_COUNT)

  #weights = numpy.array([1.0,0.6,-1.0,-0.6])
  weights = numpy.array([0.0,1.0,-0.0,-1.0])

  def buy(self, game):
    ps = game.states[self.idx]
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

    self.experimented = False

sumsqerr = 0.0
nsamples = 0
msqerrf = file('msqerr.txt', 'w')

def show_learn_data(cls):
  global sumsqerr
  global nsamples
  msqerr = sumsqerr / nsamples
  sumsqerr = 0.0
  nsamples = 0
  print '==> %s weights: %s (msq err = %.8lf)' % (cls.__name__, cls.weights, msqerr)
  print >>msqerrf, msqerr

class SimpleAI(LearningPlayerStrategy):
  compressed_A = numpy.array([[0.0]*FEATURE_COUNT for i in range(FEATURE_COUNT)])
  compressed_b = numpy.array([0.0]*FEATURE_COUNT)

  #weights = numpy.array([random.gauss(0,1) for i in range(2)])
  #weights = numpy.array([1.0,0.6,-1.0,-0.6])
  #weights = numpy.array([1.0,1.0,-1.0,-1.0])

  # These weights were obtained from 1000 self-plays.
  #weights = numpy.array([ 0.95707939,  1.93092173, -1.02307045, -1.93986092])

  # These weights were obtained by playing BMU against itself for 10,000 plays,
  # then applying the LSTD algorithm to compute weights for which the sum of TD
  # updates is zero.
  # These weights make for a high-quality AI.
  weights = numpy.array([ 4.80675988, 3.00011302, -5.08273628, -2.92180695])

  # These weights were obtained by playing SimpleAI against itself for 1000 plays,
  # starting with the previous weights and updating incrementally,
  # then applying the LSTD algorithm to compute weights.
  #weights = numpy.array([ 6.00857446,  3.15251864, -6.38471951, -3.00783411])

  def __init__(self, idx):
    LearningPlayerStrategy.__init__(self, idx)

  def buy(self, game):
    ps = game.states[self.idx]
    buy_options = ['']
    for c in costs:
      if costs[c] <= ps.money and game.supply[c] > 0:
        buy_options.append(c)

    if random.uniform(0,1) > experiment_p:
      best = -1e999
      bestc = 'x'
      for c in buy_options:
        future = game.copy(True)
        ps = future.states[self.idx]
        if c:
          ps.buy(c)
        ps.cleanup()
        now = self.evaluate(future)
        say('  Buying "%s" gives me value %.8lf' % (c,now))
        if now > best:
          best = now
          bestc = c
      assert bestc != 'x'
      self.experimented = False
    else:
      # Don't propagate results of experiments backwards!
      self.experimented = True
      bestc = random.choice(buy_options)
      say('  Experimenting.')

    if bestc:
      ps = game.states[self.idx]
      ps.buy(bestc)

TURN_LIMIT = 50
class Game(object):
  def __init__(self, parent=None, silent=False):
    self.silent = silent
    if parent:
      self.strategy_types = parent.strategy_types
      self.states = [ps.copy(self, silent) for ps in parent.states]
      self.strategies = [s for s in parent.strategies]
      self.supply = copy.copy(parent.supply)
    else:
      self.strategy_types = STRATEGY_TYPES
      random.shuffle(self.strategy_types)

      self.states = []
      self.strategies = []
      for i,t in enumerate(self.strategy_types):
        self.states.append(PlayerState(self, '%s (player %d)' % (t.__name__, i+1)))
        self.strategies.append(t(i))

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
      for ps,strat in zip(self.states, self.strategies):
        self.say('')
        self.say('*** %s\'s turn %d' % (ps.name, turn))

        while ps.play('c'): pass
        while ps.play('s'): pass
        while ps.play('g'): pass
        strat.buy(self)

        # Clean up hand and draw a new hand.
        ps.cleanup()
        strat.update_model(self)
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
    p1,p2 = self.states
    s1,s2 = self.strategies
    if p1.get_vp() > p2.get_vp():
      self.say('%s wins!' % p1.name)
      s1.train(outcome=1)
      s2.train(outcome=-1)
      return [p1.name]
    elif p2.get_vp() > p1.get_vp() or (p2.get_vp() == p1.get_vp() and p2.turns < p1.turns):
      self.say('%s wins!' % p2.name)
      s1.train(outcome=-1)
      s2.train(outcome=1)
      return [p2.name]
    else:
      self.say('%s and %s rejoice in their shared victory!' % (p1.name, p2.name))
      s1.train(outcome=0)
      s2.train(outcome=0)
      return [p1.name, p2.name]

alpha = 0.0
experiment_p = 0.0

klass = SimpleAI
STRATEGY_TYPES = [SimpleAI, SimpleAI]

def main():
  global alpha
  global experiment_p

  wins = {}
  for i in range(N):
    log_msg('******* Game %d/%d' % (i,N))
    # alpha = 0.5 approaches a reasonable solution quickly. Larger values
    # tend to diverge.
    alpha = 0.0
    experiment_p = 0.0
    game = Game()
    winners = game.play()
    print 'Game %d' % i,
    show_learn_data(klass)
    print '  Winner %s' % winners
    winners = tuple(winners)
    wins[winners] = wins.get(winners,0) + 1
  print
  for w,n in wins.items():
    print '%3d: %s' % (n,w)

  f = file('lsq_data.txt', 'w')
  print >>f, repr((klass.compressed_A, klass.compressed_b, klass.weights))

main()
