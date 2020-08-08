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
  return 0


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
    self.ship_rewards = {}  # unit_id -> (reward, position)
    self.shipyard_id_to_ship_id = {}  # record the convertion event
    self.new_ship_ids = set()

  def add_ship_reward(self, unit, reward):
    reward += self.ship_rewards.get(unit.id, 0)
    self.ship_rewards[unit.id] = reward

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
    pass

  @is_current_player
  def on_ship_deposite(self, ship, shipyard):
    # SHIP_REWARD_RATIO = 6 / 7
    deposite = ship.halite
    self.add_ship_reward(ship, deposite)

    self.step_reward += deposite
    self.total_deposite += deposite
    self.log_reward('on_ship_deposite', ship, ship.halite)


  @is_current_player
  def on_ship_collect(self, ship, delta_halite):
    # r = delta_halite * 0.03
    # self.add_ship_reward(ship, r)

    COLLECT_DISCOUNT = 0
    self.step_reward += delta_halite * COLLECT_DISCOUNT
    self.total_collect += delta_halite
    self.log_reward('on_ship_collect', ship, delta_halite)

  @is_current_player
  def on_hand_left_over_halite(self, player, deposite_halite):
    r = deposite_halite
    self.step_reward += r
    self.total_deposite += deposite_halite
    self.log_reward('on_hand_left_over_halite', None, r)

  @is_current_player
  def on_ship_move(self, ship):
    """Add some move cost."""
    # Do we need this?
    # MOVE_COST_RATE = 0.01
    # r = -max(ship.halite * MOVE_COST_RATE, 1)
    # self.add_ship_reward(ship, r)

    r = 0
    self.step_reward += r
    self.log_reward('on_ship_move', ship, r)

  @is_current_player
  def on_ship_stay(self, ship, delta_halite):
    """Add some stay cost."""
    r = 0
    self.step_reward += r
    self.log_reward('on_ship_stay', ship, r)

  @is_current_player
  def on_shipyard_spawn(self, shipyard):
    # TODO(wangfei): add it later, after agent learned how to collet halite.
    # r = -self.configuration.spawn_cost
    # self.add_shipyard_reward(shipyard, r)

    # r = -self.configuration.spawn_cost * 0.1
    r = 0
    self.step_reward += r
    self.log_reward('on_shipyard_spawn', shipyard, r)

  @is_current_player
  def on_shipyard_destroid_by_ship(self, shipyard, ship):
    # TODO(wangfei): add nearby ships for penalty.
    # r = -(self.configuration.spawn_cost + self.configuration.convert_cost)
    # r = 0
    # self.add_unit_reward(shipyard, r)

    r = -(self.configuration.spawn_cost + self.configuration.convert_cost)
    self.step_reward += r
    self.log_reward('on_shipyard_destroid_by_ship', shipyard, r)

  @is_current_player
  def on_ship_destroid_with_enemy_shipyard(self, ship, shipyard):
    if ship.id in self.new_ship_ids:
      # Do not give penality for new ship.
      return

    # TODO(wangfei): add reward for nearby shipyard attack.
    r = -(self.configuration.spawn_cost + ship.halite)
    self.step_reward += r
    self.log_reward('on_ship_destroid_with_enemy_shipyard', ship, r)

  @is_current_player
  def on_ship_destroid_in_ship_collison(self, ship):
    # Blame ship itself for the lose.
    r = -(self.configuration.spawn_cost + ship.halite)
    self.add_ship_reward(ship, r)

    self.step_reward += r
    self.log_reward('on_ship_destroid_in_ship_collison', ship, r)

  @is_current_player
  def on_ship_win_collision(self, ship, total_winning_halite,
                            total_destroied_ship):
    # COLLISION_DISCOUNT = 0.25
    # r = total_winning_halite + (self.configuration.spawn_cost *
                                # total_destroied_ship)
    # r *= COLLISION_DISCOUNT
    r = 0
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
        if shipyard.next_action == ShipyardAction.SPAWN and player.halite >= spawn_cost:
          # Handle SPAWN actions
          player._halite -= spawn_cost
          new_ship_id = ShipId(create_uid())
          self.new_ship_ids.add(new_ship_id)
          board._add_ship(
              Ship(new_ship_id, shipyard.position, 0, player.id,
                   board))
          self.on_shipyard_spawn(shipyard)
          # Clear the shipyard's action so it doesn't repeat the same action automatically
          # shipyard.next_action = None  # Do not clear action, will be use it for backprop

      for ship in player.ships:
        if ship.next_action == ShipAction.CONVERT:
          # Can't convert on an existing shipyard but you can use halite in a ship to fund conversion
          if ship.cell.shipyard_id is None and (ship.halite +
                                                player.halite) >= convert_cost:

            # TODO(wangfei): use a larger penalty after learning mining.
            # r = -self.configuration.convert_cost
            # r = 0
            # self.add_ship_reward(ship, r)
            # print("convert ship")

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


def gen_player_states(replay_json, player_id, debug=False):
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


def compute_returns(boards, as_list=False, gamma=0.99):
  ship_rewards = {}

  # Assign original unit reward.
  for b in boards:
    if as_list:
      for ship in b.current_player.ships:
        if ship.id not in ship_rewards:
          ship_rewards[ship.id] = []
        ship_rewards[ship.id].append(b.ship_rewards.get(ship.id, 0))
    else:
      for ship_id, r in b.ship_rewards.items():
        if ship_id not in ship_rewards:
          ship_rewards[ship_id] = np.zeros(len(boards), dtype=np.float32)
        ship_rewards[ship_id][b.step] += r

      # Add future events of a shipyard to the convert ship
      # source_uid = shipyard_id_to_ship_id.get(uid)
      # if source_uid:
        # ship_rewards[source_uid][b.step] += r

  for k in list(ship_rewards.keys()):
    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with gamma
    # - These are the labels for our critic
    returns = []
    discounted_sum = 0

    rewards = ship_rewards[k]
    # if as_list:
      # print("ship_id=%s, reward(n=%s, mx=%.2f, min=%.2f, mean=%.2f)" %
            # (k, len(rewards), max(rewards), min(rewards), np.mean(rewards)))
    for r in ship_rewards[k][::-1]:
      discounted_sum = r + gamma * discounted_sum
      returns.append(discounted_sum)

    returns = np.array(returns[::-1])
    ship_rewards[k] = returns / HALITE_NORMALIZTION_VAL
  return ship_rewards


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

  def get_ship_losses(self, boards, ship_probs, ship_returns, critic_values):

    def gen_action_probs(board, step_idx):
      for ship in board.current_player.ships:
        if ship.next_action == ShipAction.CONVERT:
          continue

        position = ship.position
        action_idx = SHIP_ACTION_TO_ACTION_IDX[ship.next_action]

        probs = ship_probs[step_idx, position.x, position.y, :]
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + EPS))

        prob = probs[action_idx]
        ret = ship_returns[ship.id][step_idx]
        critic = critic_values[step_idx, position.x, position.y, 0]
        yield prob, ret, critic, entropy

    def gen_action_loses():
      for i, b in enumerate(boards):
        for prob, ret, critic, entropy in gen_action_probs(b, i):
          # Adding EPS in case of zero
          # diff = ret - critic
          diff = tf.clip_by_value(ret - critic, -0.1, 0.1)
          # print('before clip: ', ret -critic)
          actor_loss = -tf.math.log(prob + EPS) * diff
          critic_loss = self.huber_loss(np.expand_dims(ret, 0),
                                        np.expand_dims(critic, 0))
          # critic_loss = tf.keras.losses.mean_absolute_error(
            # np.expand_dims(ret, 0),
            # np.expand_dims(critic, 0))
          # critic_loss = tf.nn.l2_loss(ret - critic)

          if random.random() < 0.002:
            print("prob=%.5f, ret=%.5f, critic=%.5f, critic_loss=%.5f, entropy=%.5f"
                  % (prob.numpy(), ret, critic.numpy(), critic_loss.numpy(),
                    entropy.numpy()))
          yield actor_loss, critic_loss, -entropy, critic, ret

    losses = list(gen_action_loses())
    if len(losses) == 0:
      return 0.0

    actor_losses, critic_losses, entropy_losses, critic_values, ret_values = list(zip(*losses))
    actor_losses = tf.convert_to_tensor(actor_losses)
    # actor_losses = tf.math.reduce_mean(actor_losses)
    actor_losses = tf.math.reduce_sum(actor_losses)


    critic_losses = tf.convert_to_tensor(critic_losses)
    # critic_losses = tf.math.reduce_mean(critic_losses)
    critic_losses = tf.math.reduce_sum(critic_losses)

    entropy_losses = tf.convert_to_tensor(entropy_losses)
    # entropy_losses = tf.math.reduce_mean(entropy_losses)
    entropy_losses = tf.math.reduce_sum(entropy_losses)


    cc = np.array(critic_values)
    positive = len(cc[cc > 0])
    negative = len(cc[cc < 0])
    mean_critic = np.mean(cc)

    rr = np.array(ret_values)
    rr_positive = len(rr[rr > 0])
    rr_negative = len(rr[rr < 0])
    mean_ret = np.mean(rr)
    print("critic(mean=%.5f, +%s, -%s), return(mean=%.5f, +%s, -%s)"
          % (mean_critic, positive, negative, mean_ret, rr_positive, rr_negative))

    return actor_losses, critic_losses, entropy_losses

  def get_returns(self, boards):
    board = boards[-1]
    print('\nPlayer[%s] finished at step=%s: total_deposite=%.0f, total_collect=%.0f' %
          (board.current_player.id, board.step, board.total_deposite, board.total_collect))

    ship_returns = compute_returns(boards)

    # Normalize
    mean, std = self.return_params
    def normalize(returns):
      # returns = (returns - mean) / (std + EPS)
      return returns


    ship_returns = {ship_id: normalize(returns)
                    for ship_id, returns in ship_returns.items()}
    return ship_returns

  def train(self, player_boards, apply_grad=True):
    """Player boards is [player1_boards, player2_bords ...]"""

    def get_player_inputs(boards):
      return np.array([ModelInput(b).get_input() for b in boards],
                      dtype=np.float32)

    grad_values = []
    for rollout_idx, boards in enumerate(player_boards):
      X = get_player_inputs(boards)
      ship_returns = self.get_returns(boards)

      with tf.GradientTape() as tape:
        ship_probs, critic_values = self.model(X)

        ship_actor_loss, ship_critic_loss, ship_entropy_loss = self.get_ship_losses(boards, ship_probs, ship_returns, critic_values)

        loss_regularization = tf.math.add_n(self.model.losses)
        loss = (ship_actor_loss * 1e-3 + ship_critic_loss + 1e-4 * ship_entropy_loss + loss_regularization)
        gradients, global_norm = tf.clip_by_global_norm(tape.gradient(loss, self.model.trainable_variables), 1000)

        print("Loss = %.5f: ship_actor=%.5f, ship_critic=%.5f, ship_entropy_loss=%.5f, regu=%.5f, gradient_norm=%.3f"
              % (loss, ship_actor_loss * 1e-3, ship_critic_loss,
                 1e-4*ship_entropy_loss, loss_regularization, global_norm))

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
      # boards = list(gen_player_states(replay_json, player_id))
      # trainer.train([boards], player_id)

      player_ids = list(range(num_players))

      # np.random.shuffle(player_ids)
      # for player_id in player_ids:
        # boards = list(gen_player_states(replay_json, player_id))
        # assert boards
        # trainer.train([boards], player_id)

      for player_id in player_ids:
        boards = list(gen_player_states(replay_json, player_id))
        assert boards
        yield boards

  trainer = Trainer(None, model_dir)
  trainer.train(gen_player_boards())
  trainer.on_batch_finished()

