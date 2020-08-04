# -*- coding: utf-8 -*-

import os
import random

import json
import numpy as np
import tensorflow as tf
import keras
from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *

from matrix_v0 import (SHIP_ACTIONS, SHIPYARD_ACTIONS, HALITE_NORMALIZTION_VAL,
                       ModelInput, get_model, BOARD_SIZE)



SHIP_ACTION_TO_ACTION_IDX = {a: i for i, a in enumerate(SHIP_ACTIONS)}
SHIPYARD_ACTION_TO_ACTION_IDX = {a: i for i, a in enumerate(SHIPYARD_ACTIONS)}

EPS = np.finfo(np.float32).eps.item()
# MODEL_CHECKPOINT_DIR = "/home/wangfei/data/20200801_halite/model/unet_v2"


def is_current_player(func):

  def dec(self, unit, *args, **kwargs):
    if unit:
      if isinstance(unit, Player):
        player_id = unit.id
      else:
        player_id = unit.player_id
      if player_id != self.current_player_id:
        return
    return func(self, unit, *args, **kwargs)

  return dec

def get_last_step_reward(last_board):
  c = last_board.configuration
  me = last_board.current_player
  return 0

  def get_elimination_penality(player):
    if is_player_eliminated(player, c.spawn_cost):
      return (c.episode_steps - 1 - last_board.step)
    return 0
    # return player.halite

  # halite = np.sqrt(me.halite) / 10
  halite = 0
  deposite = last_board.total_deposite
  collect = last_board.total_collect / 5
  penality = 0 #get_elimination_penality(me)
  total_reward = halite + collect + deposite - penality
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
    if not self.debug or r == 0:
      return

    if unit:
      unit_type = 'ship' if isinstance(unit, Ship) else 'yard'
      print('  S=%s: %s[%s] at %s %s: r=%s' %
            (self.step, unit_type, unit.id, unit.position, name, r))
    else:
      print("  S=%s: %s, r=%s" % (self.step, name, r))

  def on_step_finished(self):
    # If having money but not convert into shipyard.
    me = self.current_player
    if len(me.shipyard_ids) == 0:
      r = -500
      self.step_reward += r
      self.log_reward("no shipyard", unit=None, r=r)

    if len(me.ship_ids) == 0:
      r = -500
      self.step_reward += r
      self.log_reward("no ship", unit=None, r=r)

    # if (len(me.ship_ids) > 0 and len(me.shipyard_ids) == 0
        # and me.halite > self.configuration.convert_cost * 2):
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
    if ship.halite:
      print('deposite by ship %s from player %s h=%s' % (ship.id, ship.player_id, ship.halite))
    deposite = ship.halite
    self.step_reward += deposite * 2
    self.total_deposite += deposite
    self.log_reward('on_ship_deposite', ship, ship.halite)

  @is_current_player
  def on_ship_collect(self, ship, delta_halite):
    COLLECT_DISCOUNT = 0.1
    self.step_reward += delta_halite * COLLECT_DISCOUNT
    self.total_collect += delta_halite
    self.log_reward('on_ship_collect', ship, delta_halite)

  @is_current_player
  def on_hand_left_over_halite(self, player, deposite_halite):
    # r = 1 if deposite_halite > 0 else
    r = 0
    self.step_reward += r
    self.total_deposite += deposite_halite
    self.log_reward('on_hand_left_over_halite', None, r)
    self.log_reward('on_hand_left_over_halite.deposite', None, deposite_halite)


  @is_current_player
  def on_invalid_convert(self, ship):
    assert ship.halite < self.configuration.convert_cost
    # r = -(self.configuration.convert_cost - ship.halite)
    # r = -1
    r = 0
    self.step_reward += r
    self.log_reward('on_invalid_convert', ship, r)

  @is_current_player
  def on_ship_move(self, ship):
    """Add some move cost."""
    MOVE_COST_RATE = 0.01
    r = -max(ship.halite * MOVE_COST_RATE, 1)
    # r = -50
    # r = -1
    # r = 0
    self.step_reward += r
    self.log_reward('on_ship_move', ship, r)

  @is_current_player
  def on_ship_stay(self, ship, delta_halite):
    """Add some stay cost."""
    r = 0
    if delta_halite == 0:
      r = -1
    self.step_reward += r
    self.log_reward('on_ship_stay', ship, r)

  @is_current_player
  def on_invalid_spawn(self, shipyard):
    assert shipyard.player.halite < self.configuration.spawn_cost
    # r = -(self.configuration.spawn_cost - shipyard.player.halite)
    # r = -50
    # r = -1
    r = 0
    self.step_reward += r
    self.log_reward('on_invalid_spawn', shipyard, r)

  @is_current_player
  def on_shipyard_spawn(self, shipyard):
    # r = -self.configuration.spawn_cost * 0.1
    r = 0
    self.step_reward += r
    self.log_reward('on_shipyard_spawn', shipyard, r)

  @is_current_player
  def on_shipyard_destroid_by_ship(self, shipyard, ship):
    r = -(self.configuration.spawn_cost + self.configuration.convert_cost)
    self.step_reward += r
    self.log_reward('on_shipyard_destroid_by_ship', shipyard, r)

  @is_current_player
  def on_ship_destroid_with_enemy_shipyard(self, ship, shipyard):
    # TODO(wangfei): add reward for nearby shipyard attack.
    r = 0
    self.step_reward += r
    self.log_reward('on_ship_destroid_with_enemy_shipyard', ship, r)

  @is_current_player
  def on_ship_destroid_in_ship_collison(self, ship):
    COLLISION_DISCOUNT = 0.5
    r = -(self.configuration.spawn_cost + ship.halite)
    r *= COLLISION_DISCOUNT
    self.step_reward += r
    self.log_reward('on_ship_destroid_in_ship_collison', ship, r)

  @is_current_player
  def on_ship_win_collision(self, ship, total_winning_halite,
                            total_destroied_ship):
    COLLISION_DISCOUNT = 0.25
    r = total_winning_halite + (self.configuration.spawn_cost *
                                total_destroied_ship)
    r *= COLLISION_DISCOUNT
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
          self.on_shipyard_spawn(shipyard)
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

      if player.id == self.current_player.id:
        self.on_hand_left_over_halite(player, leftover_convert_halite)

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
      if (ship.next_action not in ShipAction.moves()
          and cell.shipyard_id is None and delta_halite > 0):
        self.on_ship_collect(ship, delta_halite)
        ship._halite += delta_halite
        cell._halite -= delta_halite
        # Clear the ship's action so it doesn't repeat the same action automatically
        # ship.next_action = None  # do not clear

      if ship.next_action not in ShipAction.moves():
        self.on_ship_stay(ship, delta_halite)


    # Regenerate halite in cells
    for cell in board.cells.values():
      if cell.ship_id is None:
        next_halite = round(cell.halite * (1 + configuration.regen_rate), 3)
        cell._halite = min(next_halite, configuration.max_cell_halite)
        # Lets just check and make sure.
        assert cell.halite >= 0

    self.on_step_finished()
    board._step += 1
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


def gen_player_states(replay_json, player_id, steps=None, debug=False):
  steps = len(replay_json['steps'])
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
    # print('S=%s, board.total_deposite=' % (i), board.total_deposite)
    yield board

    if is_player_eliminated(board.current_player,
                            board.configuration.spawn_cost):
      break
  board.step_reward += get_last_step_reward(board)


def gen_replays(path, replayer_id=None):
  for replay_name in os.listdir(path):
    replay_path = os.path.join(path, replay_name)
    with open(replay_path, 'r') as f:
      replay_json = json.loads(f.read())

    if replayer_id and replay_json['id'] != replayer_id:
      continue
    yield replay_json


def compute_returns(boards, gamma=0.995):
  step_rewards = np.array([b.step_reward for b in boards], dtype=np.float32)
  step_rewards /= HALITE_NORMALIZTION_VAL

  # Calculate expected value from rewards
  # - At each timestep what was the total reward received after that timestep
  # - Rewards in the past are discounted by multiplying them with gamma
  # - These are the labels for our critic
  returns = []
  discounted_sum = 0
  for r in step_rewards[::-1]:
    discounted_sum = r + gamma * discounted_sum
    returns.append(discounted_sum)
  returns = np.array(returns[::-1])
  return returns


class Trainer:

  BORRD_POSITIONS = [Point(i, j) for i in range(BOARD_SIZE)
                     for j in range(BOARD_SIZE)]

  def __init__(self, model, model_dir, return_params=None):
    self.model = model
    if self.model is None:
      self.model = get_model()
    assert self.model

    self.return_params = return_params
    self.optimizer = keras.optimizers.Adam(learning_rate=3e-5)
    # self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    self.huber_loss = tf.keras.losses.Huber()

    if model_dir:
      self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=self.model)
      self.manager = tf.train.CheckpointManager(self.checkpoint,
                                                model_dir,
                                                max_to_keep=20)

      # Load existing model if it exists.
      self.checkpoint.restore(self.manager.latest_checkpoint)
      if self.manager.latest_checkpoint:
        print("Restored from {}".format(self.manager.latest_checkpoint))
      else:
        print("Initializing model from scratch.")

  def get_ship_actor_loss(self, boards, ship_probs, critic_diffs):
    def get_ship_units(board):
      return board.current_player.ships

    loses, correct_rate = self.get_actor_loss(boards, ship_probs,
                                                        SHIP_ACTION_TO_ACTION_IDX,
                                                        get_ship_units,
                                                        critic_diffs)
    # print("Non action ship cells precision: ", correct_rate)
    return loses

  def get_shipyard_actor_loss(self, boards, yard_probs, critic_diffs):
    def get_shipyard_units(board):
      return board.current_player.shipyards

    loses, correct_rate =  self.get_actor_loss(boards, yard_probs,
                                     SHIPYARD_ACTION_TO_ACTION_IDX,
                                     get_shipyard_units, critic_diffs)
    # print("Non action shipyard cells precision: ", correct_rate)
    return loses

  def get_actor_loss(self, boards, unit_probs, action_to_idx, get_units, critic_diffs):
    # Sample non-actionable cells for fast training.
    N_SAMPLE = 50
    non_unit_cells = 0
    correct_non_unit_actions = 0

    def unit_action_probs(board, step_idx):
      nonlocal non_unit_cells, correct_non_unit_actions

      position_to_unit = {u.position : u for u in get_units(board)}

      # positions = (random.sample(self.BORRD_POSITIONS, N_SAMPLE)
                   # if len(boards) > 150
                   # else self.BORRD_POSITIONS)
      # for position in positions:
      # # for position in not_actionable_cells:
        # if position in position_to_unit:
          # continue

        # non_unit_cells += 1

        # # Because it won't affect the state of the board and action won't be
        # # recorded for non-unit cells, thus it's generated during training.
        # action_probs = unit_probs[step_idx, position.x, position.y, :]
        # action_idx = np.random.choice(len(action_to_idx), p=action_probs.numpy())
        # if action_idx == len(action_to_idx) -1:
          # correct_non_unit_actions += 1
        # yield action_probs[action_idx]

      for position, unit in position_to_unit.items():
        # if unit.next_action is None:
          # # Same here, since None or False will not affect the board,
          # # it's sampled again here.
          # action_probs = unit_probs[step_idx, position.x, position.y, -2:]

          # # adding EPS in case of zero
          # probs = np.array([action_probs[-2], action_probs[-1]]) + EPS
          # # TODO(wangfei): need damping factor here
          # probs = probs / np.sum(probs)
          # action_idx = np.random.choice(2, p=probs)
          # yield action_probs[action_idx]
        # else:
        action_idx = action_to_idx[unit.next_action]
        yield unit_probs[step_idx, position.x, position.y, action_idx]

    # def log_prob(g):
      # g = list(g)
      # if len(g) == 0:
        # return tf.constant(0.0)
      # if np.abs(np.sum(np.array(g))) < EPS:
        # return tf.constant(0.0)
      # This is bug!
      # return tf.math.log(tf.math.reduce_sum(g))


    def gen_action_loses():
      for i, (b, critic_diff) in enumerate(zip(boards, critic_diffs)):
        for prob in unit_action_probs(b, i):
          # Adding EPS in case of zero
          yield -tf.math.log(prob + EPS) * critic_diff

    action_losses = list(gen_action_loses())
    if len(action_losses) == 0:
      return 0.0, 0.0

    action_losses = tf.convert_to_tensor(action_losses)
    action_loss = tf.math.reduce_mean(action_losses)
    # non_action_cell_correct_rate = correct_non_unit_actions / non_unit_cells
    return action_loss, 0.0 # non_action_cell_correct_rate

  def get_returns(self, boards):
    board = boards[-1]
    print('\nPlayer[%s] finished at step=%s: total_deposite=%.0f, total_collect=%.0f' %
          (board.current_player.id, board.step, board.total_deposite, board.total_collect))

    returns = compute_returns(boards)

    # Normalize
    if self.return_params:
      mean_return, std_return = self.return_params

      positive = len(returns[returns > 0])
      negative = len(returns[returns < 0])
      returns = (returns - mean_return) / (std_return + EPS)

      p2 = len(returns[returns > 0])
      n2 = len(returns[returns < 0])
      print('Episode return(+%s, -%s), normalized(+%s, -%s)'
            % (positive, negative, p2, n2))
    return returns

  def train(self, player_boards, apply_grad=True):
    """Player boards is [player1_boards, player2_bords ...]"""

    def get_player_inputs(boards):
      return np.array([ModelInput(b).get_input() for b in boards],
                      dtype=np.float32)

    grad_values = []
    for rollout_idx, boards in enumerate(player_boards):
      returns = self.get_returns(boards)
      X = get_player_inputs(boards)

      with tf.GradientTape() as tape:
        (ship_probs, yard_probs, critic_values) = self.model(X)
        diffs = returns - critic_values[:, 0]

        ship_actor_loss = self.get_ship_actor_loss(boards, ship_probs, diffs)
        yard_actor_loss = self.get_shipyard_actor_loss(boards, yard_probs, diffs)
        critic_losses = self.huber_loss(critic_values[:, 0], returns)
        loss_regularization = tf.math.add_n(self.model.losses)
        loss = (ship_actor_loss + yard_actor_loss + 2*critic_losses + loss_regularization)
        gradients, global_norm = tf.clip_by_global_norm(tape.gradient(loss, self.model.trainable_variables), 2000)

        print('returns [-5:]', returns[-5:])
        print('critic [-5:]', critic_values[-5:, 0].numpy())
        cc = critic_values[:, 0].numpy()
        positive = len(cc[cc > 0])
        negative = len(cc[cc < 0])
        print('p[%s] critic_losses' % rollout_idx, np.sum(critic_losses),
              ', mean predicted critic:', np.mean(cc), ' +%s, -%s' % (positive, negative))

        print('mean diffs', np.mean(diffs), 'shape:', diffs.shape)
        print("Loss: ship=%.2f, yard=%.2f, critic=%.2f, regu=%.2f, gradient_norm=%.3f"
              % (ship_actor_loss, yard_actor_loss, 2 * critic_losses,
                 loss_regularization, global_norm))

        grad_values.append(gradients)

    if apply_grad:
      for i, grads in enumerate(grad_values):
        self.apply_grad(grads)

    return grad_values

  def apply_grad(self, grads):
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

  def on_batch_finished(self):
    self.checkpoint.step.assign_add(1)
    save_path = self.manager.save()
    print("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path))


def train_on_replays(model_dir, replay_jsons):

  def gen_player_boards():
    for replay_json in replay_jsons:
      total_steps = len(replay_json['steps'])
      print('Start training on', replay_json['id'], replay_json['rewards'],
            'total_steps:', total_steps)

      num_players = len(replay_json['rewards'])

      # player_id = np.random.choice(num_players)
      # boards = list(gen_player_states(replay_json, player_id, total_steps))
      # trainer.train([boards], player_id)

      player_ids = list(range(num_players))

      # np.random.shuffle(player_ids)
      # for player_id in player_ids:
        # boards = list(gen_player_states(replay_json, player_id, total_steps))
        # assert boards
        # trainer.train([boards], player_id)

      for player_id in player_ids:
        boards = list(gen_player_states(replay_json, player_id, total_steps))
        assert boards
        yield boards

  trainer = Trainer(None, model_dir)
  trainer.train(gen_player_boards())
  trainer.on_batch_finished()

