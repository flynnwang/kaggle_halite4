# -*- coding: utf-8 -*-

import os

import json
import numpy as np
import tensorflow as tf
import keras
from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *

from matrix_v0 import (SHIP_ACTIONS, SHIPYARD_ACTIONS, HALITE_NORMALIZTION_VAL,
                       ModelInput, get_model, OFFSET)



SHIP_ACTION_TO_ACTION_IDX = {a: i for i, a in enumerate(SHIP_ACTIONS)}
SHIPYARD_ACTION_TO_ACTION_IDX = {a: i for i, a in enumerate(SHIPYARD_ACTIONS)}

EPS = np.finfo(np.float32).eps.item()
# MODEL_CHECKPOINT_DIR = "/home/wangfei/data/20200801_halite/model/unet_v2"


def is_current_player(func):

  def dec(self, unit, *args, **kwargs):
    if unit and unit.player_id != self.current_player_id:
      return

    # if self.step == self.configuration.episode_steps - 1:
      # return
    return func(self, unit, *args, **kwargs)

  return dec

def get_last_step_reward(last_board):
  c = last_board.configuration
  me = last_board.current_player

  def get_player_reward(player):
    if is_player_eliminated(player, c.spawn_cost):
      failed_steps = last_board.step - c.episode_steps - 1
      return failed_steps
    return player.halite

  # Use total deposite as inital optimization object.
  # return np.sqrt(last_board.total_deposite)
  # return np.sqrt(last_board.total_collect) + last_board.total_deposite
  total_reward = (last_board.total_collect * 0.5 + last_board.total_deposite
                  + me.halite)
  if total_reward == 0:
    return -10
  return total_reward
  # return 0

  # player_rewards = [get_player_reward(p) for p in last_board.players.values()]
  # sorted_rewards = sorted(player_rewards, reverse=True)
  # max_reward = max(player_rewards)

  # current_player_reward = player_rewards[last_board.current_player.id]

  # # If current player is the absolute 1st.
  # if current_player_reward == max_reward and max_reward > sorted_rewards[1]:
    # return np.sqrt(max_reward) + 500
  # return -np.sqrt(max_reward - current_player_reward) - 500

class EventBoard(Board):

  def __init__(self,
               raw_observation: Dict[str, Any],
               raw_configuration: Union[Configuration, Dict[str, Any]],
               next_actions: Optional[List[Dict[str, str]]] = None) -> None:
    super().__init__(raw_observation, raw_configuration, next_actions)
    self.step_reward = 0
    self.debug = True
    self.total_deposite = 0
    self.total_collect = 0

  def log_reward(self, name, unit, r):
    if not self.debug:
      return

    if unit:
      unit_type = 'ship' if isinstance(unit, Ship) else 'yard'
      print('  %s[%s] at %s %s: r=%s' %
            (unit_type, unit.id, unit.position, name, r))
    else:
      print("  %s, r=%s" % (name, r))

  def on_step_finished(self):
    # If having money but not convert into shipyard.
    me = self.current_player
    # if len(me.shipyard_ids) == 0 and me.halite > self.configuration.convert_cost:
      # self.step_reward -= 1
      # self.log_reward("not converting shipyard ", unit=None, r=-1)

    # if (len(me.shipyard_ids) > 0 and me.halite > self.configuration.spawn_cost
        # and len(me.ship_ids) == 0):
      # self.step_reward -= 1
      # self.log_reward("not spawn ship ", unit=None, r=-1)

    # if len(me.shipyard_ids) > 0:
      # self.step_reward += 1
      # self.log_reward("having at least one shipyard", unit=None, r=1)

    # if len(me.ship_ids) > 0:
      # self.step_reward += 1
      # self.log_reward("having at least one ship", unit=None, r=1)



  @is_current_player
  def on_ship_deposite(self, ship, shipyard):
    deposite = ship.halite
    self.step_reward += deposite
    self.total_deposite += deposite
    self.log_reward('on_ship_deposite', ship, ship.halite)

  @is_current_player
  def on_ship_collect(self, ship, delta_halite):
    self.step_reward += delta_halite
    self.total_collect += delta_halite
    self.log_reward('on_ship_collect', ship, delta_halite)

  def on_hand_left_over_halite(self, deposite_halite):
    r = 0
    self.total_deposite += deposite_halite
    self.log_reward('on_hand_left_over_halite', None, r)

  @is_current_player
  def on_invalid_convert(self, ship):
    assert ship.halite < self.configuration.convert_cost
    # r = -(self.configuration.convert_cost - ship.halite)
    # r = -1
    r = -50
    self.step_reward += r
    self.log_reward('on_invalid_convert', ship, r)

  @is_current_player
  def on_ship_move(self, ship):
    """Add some move cost."""
    MOVE_COST_RATE = 0.01
    # r = -max(ship.halite * MOVE_COST_RATE, 1)
    # r = -50
    r = 0
    self.step_reward += r
    self.log_reward('on_ship_move', ship, r)

  @is_current_player
  def on_ship_stay(self, ship):
    """Add some stay cost."""
    r = 0
    # if ship.cell.halite == 0:
      # r = -50
    self.step_reward += r
    self.log_reward('on_ship_stay', ship, r)

  @is_current_player
  def on_invalid_spawn(self, shipyard):
    assert shipyard.player.halite < self.configuration.spawn_cost
    # r = -(self.configuration.spawn_cost - shipyard.player.halite)
    # r = -50
    r = 0
    self.step_reward += r
    self.log_reward('on_invalid_spawn', shipyard, r)

  @is_current_player
  def on_shipyard_destroid_by_ship(self, shipyard, ship):
    r = -(self.configuration.spawn_cost + self.configuration.convert_cost)
    self.step_reward += r
    self.log_reward('on_shipyard_destroid_by_ship', shipyard, r)

  @is_current_player
  def on_ship_destroid_with_enemy_shipyard(self, ship, shipyard):
    # TODO(wangfei): add reward for nearby shipyard attack.
    r = 1
    self.step_reward += r
    self.log_reward('on_ship_destroid_with_enemy_shipyard', ship, r)

  @is_current_player
  def on_ship_destroid_in_ship_collison(self, ship):
    r = -self.configuration.spawn_cost
    self.step_reward += r
    self.log_reward('on_ship_destroid_in_ship_collison', ship, r)

  @is_current_player
  def on_ship_win_collision(self, ship, total_winning_halite,
                            total_destroied_ship):
    r = total_winning_halite + (self.configuration.spawn_cost *
                                total_destroied_ship)
    self.step_reward += r
    self.log_reward('on_ship_win_collision', ship, r)

  def next(self):
    """
    Returns a new board with the current board's next actions applied.
      The current board is unmodified.
      This can form a halite interpreter, e.g.
          next_observation = Board(current_observation, configuration, actions).next().observation
    """
    # Create a copy of the board to modify so we don't affect the current board
    board = deepcopy(self)
    configuration = board.configuration
    convert_cost = configuration.convert_cost
    spawn_cost = configuration.spawn_cost
    uid_counter = 0

    # This is a consistent way to generate unique strings to form ship and shipyard ids
    def create_uid():
      nonlocal uid_counter
      uid_counter += 1
      return f"{self.step + 1}-{uid_counter}"

    # Process actions and store the results in the ships and shipyards lists for collision checking
    for player in board.players.values():
      leftover_convert_halite = 0

      for shipyard in player.shipyards:
        if shipyard.next_action == ShipyardAction.SPAWN and player.halite < spawn_cost:
          self.on_invalid_spawn(shipyard)

        if shipyard.next_action == ShipyardAction.SPAWN and player.halite >= spawn_cost:
          # Handle SPAWN actions
          player._halite -= spawn_cost
          board._add_ship(
              Ship(ShipId(create_uid()), shipyard.position, 0, player.id,
                   board))
          # Clear the shipyard's action so it doesn't repeat the same action automatically
          # shipyard.next_action = None  # Do not clear action, will be use it for backprop

      for ship in player.ships:
        if ship.next_action == ShipAction.CONVERT:
          if ship.cell.shipyard_id or (ship.halite +
                                       player.halite) < convert_cost:
            self.on_invalid_convert(ship)

          # Can't convert on an existing shipyard but you can use halite in a ship to fund conversion
          if ship.cell.shipyard_id is None and (ship.halite +
                                                player.halite) >= convert_cost:
            # Handle CONVERT actions
            delta_halite = ship.halite - convert_cost
            # Excess halite leftover from conversion is added to the player's total only after all conversions have completed
            # This is to prevent the edge case of chaining halite from one convert to fund other converts
            leftover_convert_halite += max(delta_halite, 0)
            player._halite += min(delta_halite, 0)
            board._add_shipyard(
                Shipyard(ShipyardId(create_uid()), ship.position, player.id,
                         board))
            board._delete_ship(ship)
        elif ship.next_action is not None:
          self.on_ship_move(ship)

          # If the action is not None and is not CONVERT it must be NORTH, SOUTH, EAST, or WEST
          ship.cell._ship_id = None
          ship._position = ship.position.translate(ship.next_action.to_point(),
                                                   configuration.size)
          ship._halite *= (1 - board.configuration.move_cost)

          # We don't set the new cell's ship_id here as it would be overwritten by another ship in the case of collision.
          # Later we'll iterate through all ships and re-set the cell._ship_id as appropriate.

        if ship.next_action == None:
          self.on_ship_stay(ship)


        self.on_hand_left_over_halite(leftover_convert_halite)
        player._halite += leftover_convert_halite
        # Lets just check and make sure.
        assert player.halite >= 0

    def resolve_collision(
        ships: List[Ship]) -> Tuple[Optional[Ship], List[Ship]]:
      """
      Accepts the list of ships at a particular position (must not be empty).
        Returns the ship with the least halite or None in the case of a tie along with all other ships.
      """
      if len(ships) == 1:
        return ships[0], []
      ships_by_halite = group_by(ships, lambda ship: ship.halite)
      smallest_halite = min(ships_by_halite.keys())
      smallest_ships = ships_by_halite[smallest_halite]
      if len(smallest_ships) == 1:
        # There was a winner, return it
        winner = smallest_ships[0]
        return winner, [ship for ship in ships if ship != winner]
      # There was a tie for least halite, all are deleted
      return None, ships

    # Check for ship to ship collisions
    ship_collision_groups = group_by(board.ships.values(),
                                     lambda ship: ship.position)
    for position, collided_ships in ship_collision_groups.items():
      winner, deleted = resolve_collision(collided_ships)
      if winner is not None:
        winner.cell._ship_id = winner.id

      total_winning_halite = 0
      for ship in deleted:
        board._delete_ship(ship)
        total_winning_halite += ship.halite
        self.on_ship_destroid_in_ship_collison(ship)
        if winner is not None:
          # Winner takes deleted ships' halite
          winner._halite += ship.halite

      if winner is not None and deleted:
        self.on_ship_win_collision(winner, total_winning_halite, len(deleted))

    # Check for ship to shipyard collisions
    for shipyard in list(board.shipyards.values()):
      ship = shipyard.cell.ship
      if ship is not None and ship.player_id != shipyard.player_id:
        self.on_ship_destroid_with_enemy_shipyard(ship, shipyard)
        self.on_shipyard_destroid_by_ship(shipyard, ship)

        # Ship to shipyard collision
        board._delete_shipyard(shipyard)
        board._delete_ship(ship)

    # Deposit halite from ships into shipyards
    for shipyard in list(board.shipyards.values()):
      ship = shipyard.cell.ship
      if ship is not None and ship.player_id == shipyard.player_id:
        self.on_ship_deposite(ship, shipyard)
        shipyard.player._halite += ship.halite
        ship._halite = 0

    # Collect halite from cells into ships
    for ship in board.ships.values():
      cell = ship.cell
      delta_halite = int(cell.halite * configuration.collect_rate)
      if ship.next_action not in ShipAction.moves(
      ) and cell.shipyard_id is None and delta_halite > 0:
        self.on_ship_collect(ship, delta_halite)
        ship._halite += delta_halite
        cell._halite -= delta_halite
        # Clear the ship's action so it doesn't repeat the same action automatically
        # ship.next_action = None  # do not clear

    # Regenerate halite in cells
    for cell in board.cells.values():
      if cell.ship_id is None:
        next_halite = round(cell.halite * (1 + configuration.regen_rate), 3)
        cell._halite = min(next_halite, configuration.max_cell_halite)
        # Lets just check and make sure.
        assert cell.halite >= 0

    self.on_step_finished()
    board._step += 1
    # board.total_deposite = self.total_deposite
    # board.total_collect = self.total_collect
    return board


class Replayer:

  def __init__(self, strategy, replay_json, player_id=0):
    self.strategy = strategy
    self.replay_json = replay_json
    self.player_id = player_id
    self.env = make("halite",
                    configuration=replay_json['configuration'],
                    steps=replay_json['steps'])
    self.step = 0
    self.total_steps = len(replay_json['steps'])

  def get_board(self, step, board_cls=EventBoard):
    state = self.replay_json['steps'][step][0]
    obs = state['observation']
    obs['player'] = self.player_id

    actions = None
    if step + 1 < self.total_steps:
      actions = [
          self.replay_json['steps'][step + 1][p]['action'] for p in range(4)
      ]
    return board_cls(obs, self.env.configuration, actions)

  def check_board_valid(self):
    for i in range(self.total_steps):
      event_board = self.get_board(i, EventBoard)
      board = self.get_board(i, Board)
      assert event_board.next().observation == board.next(
      ).observation, 'Failed check on step %s' % i
      print("Step %s, step reward = %s" % (i, event_board.step_reward))

  def simulate(self, step=0):
    board = self.get_board(step)
    self.strategy.update(board)
    self.strategy.execute()
    self.step += 1

  def play(self, steps=1):
    for i in range(steps):
      self.simulate(i)


def is_player_eliminated(player, spawn_cost):
  """A player is eliminated when no ships left and no shipyard for building a
  ship, or not enough money to build one."""
  return (len(player.ship_ids) == 0 and
          (len(player.shipyard_ids) == 0 or player.halite < spawn_cost))


def gen_player_states(replay_json, player_id, steps, debug=False):
  replayer = Replayer(None, replay_json, player_id)
  # start from 1 because there ase only 399 actions.
  prev = None
  for i in range(0, steps):
    board = replayer.get_board(i)
    board.step_reward = 0
    if prev:
      board.total_deposite = prev.total_deposite
      board.total_collect = prev.total_collect

    board.debug = debug
    #         print('Step #', board.step)
    board.next()
    prev = board
    yield board

    if is_player_eliminated(board.current_player,
                            board.configuration.spawn_cost):
      break
  board.step_reward += get_last_step_reward(board)

def log_prob(g):
  g = list(g)
  if len(g) == 0:
    return tf.constant(0.0)
  if np.abs(np.sum(np.array(g))) < EPS:
    return tf.constant(0.0)
  return tf.math.log(tf.math.reduce_sum(g))


def gen_replays(path, replayer_id=None):
  for replay_name in os.listdir(path):
    replay_path = os.path.join(path, replay_name)
    with open(replay_path, 'r') as f:
      replay_json = json.loads(f.read())

    if replayer_id and replay_json['id'] != replayer_id:
      continue
    yield replay_json


class Trainer:

  def __init__(self, model, model_dir):
    self.model = model
    assert model

    self.optimizer = keras.optimizers.Adam(learning_rate=3e-5)
    # self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    self.huber_loss = tf.keras.losses.Huber()
    self.gamma = 0.99  # Discount factor for past rewards

    self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
    self.manager = tf.train.CheckpointManager(self.checkpoint,
                                              model_dir,
                                              max_to_keep=20)

    # Load existing model if it exists.
    self.checkpoint.restore(self.manager.latest_checkpoint)
    if self.manager.latest_checkpoint:
      print("Restored from {}".format(self.manager.latest_checkpoint))
    else:
      print("Initializing model from scratch.")

  def get_ship_action_log_probs(self, boards, ship_probs):

    def ship_action_probs(board, step_idx):
      for ship in board.current_player.ships:
        action_idx = SHIP_ACTION_TO_ACTION_IDX[ship.next_action]
        img_pos = ship.position + OFFSET
        yield ship_probs[step_idx, img_pos.x, img_pos.y, action_idx]

    action_log_probs = [
        log_prob(ship_action_probs(b, i)) for i, b in enumerate(boards)
    ]
    return tf.convert_to_tensor(action_log_probs)

  def get_shipyard_action_log_probs(self, boards, yard_probs):

    def shipyard_action_probs(board, step_idx):
      for yard in board.current_player.shipyards:
        action_idx = SHIPYARD_ACTION_TO_ACTION_IDX[yard.next_action]
        img_pos = yard.position + OFFSET
        prob = yard_probs[step_idx, img_pos.x, img_pos.y, 0]
        yield prob if action_idx == 0 else (1 - prob)

    action_log_probs = [
        log_prob(shipyard_action_probs(b, i)) for i, b in enumerate(boards)
    ]
    return tf.convert_to_tensor(action_log_probs)

  def get_returns(self, boards):
    step_rewards = np.array([b.step_reward for b in boards], dtype=np.float32)
    step_rewards /= HALITE_NORMALIZTION_VAL

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with gamma
    # - These are the labels for our critic
    returns = []
    discounted_sum = 0
    for r in step_rewards[::-1]:
      discounted_sum = r + self.gamma * discounted_sum
      returns.append(discounted_sum)
    returns = np.array(returns[::-1])

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    board = boards[-1]

    positive = len(returns[returns > 0])
    negative = len(returns[returns < 0])
    zero = len(returns) - positive - negative
    print('reward at last step=%s for player %s: return=%.3f(mean=%.3f, std=%.3f, +%s, -%s, zero=%s), total_deposite=%.0f' %
          (board.step, board.current_player.id, returns[-1], mean_return, std_return,
           positive, negative, zero, board.total_deposite))

    # Normalize
    # returns = (returns - mean_return) / (std_return + EPS)
    returns = returns / (std_return + EPS)
    return returns

  def train(self, player_boards, player_id):
    player_returns = [self.get_returns(boards) for boards in player_boards]

    def get_player_inputs(boards):
      return np.array([ModelInput(b).get_input() for b in boards],
                      dtype=np.float32)

    player_inputs = [get_player_inputs(boards) for boards in player_boards]

    with tf.GradientTape() as tape:
      player_preds = [self.model(X) for X in player_inputs]

      loss_values = None
      for i, ((ship_probs, yard_probs, critic_values), returns, boards) in enumerate(zip(player_preds, player_returns, player_boards)):
        ship_action_log_probs = self.get_ship_action_log_probs(boards, ship_probs)
        yard_action_log_probs = self.get_shipyard_action_log_probs(
            boards, yard_probs)

        # Calculating loss values to update our network.

        # At this point in history, the critic estimated that we would get a
        # total reward = `value` in the future. We took an action with log probability
        # of `log_prob` and ended up recieving a total reward = `ret`.
        # The actor must be updated so that it predicts an action that leads to
        # high rewards (compared to critic's estimate) with high probability.
        diff = returns - critic_values[:, 0]
        print('mean diff', np.mean(diff))
        ship_actor_losses = -ship_action_log_probs * diff
        yard_actor_losses = -yard_action_log_probs * diff

        print('p[%s] mean ship_actor_losses' % player_id, np.mean(ship_actor_losses))
        print('p[%s] mean yard_actor_losses' % player_id , np.mean(yard_actor_losses))

        # The critic must be updated so that it predicts a better estimate of
        # the future rewards.
        # critic_losses = self.huber_loss(critic_values[:, 0], returns)
        critic_losses = [self.huber_loss(tf.expand_dims(x, 0), tf.expand_dims(y, 0))
                         for x, y in zip(critic_values[:, 0], returns)]
        print('critic_values.shape', critic_values[:, 0].shape, 'return.shape',
              returns.shape, 'critic_losses.shape', np.array(critic_losses).shape)
        print('returns [:10]', returns[-10:])
        print('critic [:10]', critic_values[-10:, 0].numpy())

        cc = critic_values[:, 0].numpy()
        positive = len(cc[cc > 0])
        negative = len(cc[cc < 0])
        zero = len(cc) - positive - negative
        print('p[%s] mean critic_losses' % player_id, np.mean(critic_losses),
              self.huber_loss(critic_values[:, 0], returns).numpy(),
              'mean critic values:', np.mean(cc), ' +%s, -%s, zero=%s' % (positive, negative, zero))

        tmp = sum(ship_actor_losses) + sum(yard_actor_losses) + sum(critic_losses)
        # tmp = ship_actor_losses + yard_actor_losses + critic_losses
        assert len(player_boards) == 1
        loss_values = tmp

      grads = tape.gradient(loss_values, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    self.on_episode_training_finished()

  def on_episode_training_finished(self):
    self.checkpoint.step.assign_add(1)
    if int(self.checkpoint.step) % 50 == 0:
      save_path = self.manager.save()
      print("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path))


def train_on_replays(trainer, replay_jsons):
  for replay_json in replay_jsons:
    total_steps = len(replay_json['steps'])
    print('Start training on', replay_json['id'], replay_json['rewards'],
          'total_steps:', total_steps)

    num_players = len(replay_json['rewards'])
    player_id = np.random.choice(num_players)
    boards = list(gen_player_states(replay_json, player_id, total_steps))
    trainer.train([boards], player_id)

    # for player_id in range(num_players):
      # boards = list(gen_player_states(replay_json, player_id, total_steps))
      # assert boards
      # # player_boards[player_id] = boards
      # trainer.train([boards], player_id)
