from kaggle_environments import evaluate, make

from kaggle_environments.envs.halite.helpers import *


def is_current_player(func):
  def dec(self, unit, *args, **kwargs):
    if unit.player_id != self.current_player_id:
      return

    if self.step == self.configuration.episode_steps - 1:
      return
    return func(self, unit, *args, **kwargs)

  return dec

class EventBoard(Board):

  def __init__(
    self,
    raw_observation: Dict[str, Any],
    raw_configuration: Union[Configuration, Dict[str, Any]],
    next_actions: Optional[List[Dict[str, str]]] = None
  ) -> None:
    super().__init__(raw_observation, raw_configuration, next_actions)
    self.step_reward = 0
    self.debug = True

  def log_reward(self, name, unit, r):
    if not self.debug:
      return
    unit_type = 'ship' if isinstance(unit, Ship) else 'yard'
    print('  %s[%s] at %s %s: r=%s' % (unit_type, unit.id, unit.position,
                                         name, r))

  @is_current_player
  def on_ship_deposite(self, ship, shipyard):
    self.step_reward += ship.halite
    self.log_reward('on_ship_deposite', ship, ship.halite)

  @is_current_player
  def on_ship_collect(self, ship, delta_halite):
    self.step_reward += delta_halite
    self.log_reward('on_ship_collect', ship, delta_halite)

  @is_current_player
  def on_invalid_convert(self, ship):
    assert ship.halite < self.configuration.convert_cost
    r = -(self.configuration.convert_cost - ship.halite)
    self.step_reward += r
    self.log_reward('on_invalid_convert', ship, r)

  @is_current_player
  def on_ship_move(self, ship):
    """Add some move cost."""
    MOVE_COST_RATE = 0.01
    r = -max(ship.halite * MOVE_COST_RATE, 1)
    self.step_reward += r
    self.log_reward('on_ship_move', ship, r)

  @is_current_player
  def on_invalid_spawn(self, shipyard):
    assert shipyard.player.halite < self.configuration.spawn_cost
    r = -(self.configuration.spawn_cost - shipyard.player.halite)
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
    r = 50
    self.step_reward += r
    self.log_reward('on_ship_destroid_with_enemy_shipyard', ship, r)

  @is_current_player
  def on_ship_destroid_in_ship_collison(self, ship):
    r = -self.configuration.spawn_cost
    self.step_reward += r
    self.log_reward('on_ship_destroid_in_ship_collison', ship, r)

  @is_current_player
  def on_ship_win_collision(self, ship, total_winning_halite, total_destroied_ship):
    r = total_winning_halite + (self.configuration.spawn_cost * total_destroied_ship)
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
          board._add_ship(Ship(ShipId(create_uid()), shipyard.position, 0, player.id, board))
          # Clear the shipyard's action so it doesn't repeat the same action automatically
          # shipyard.next_action = None  # Do not clear action, will be use it for backprop

      for ship in player.ships:
        if ship.next_action == ShipAction.CONVERT:
          if ship.cell.shipyard_id or (ship.halite + player.halite) < convert_cost:
            self.on_invalid_convert(ship)

          # Can't convert on an existing shipyard but you can use halite in a ship to fund conversion
          if ship.cell.shipyard_id is None and (ship.halite + player.halite) >= convert_cost:
            # Handle CONVERT actions
            delta_halite = ship.halite - convert_cost
            # Excess halite leftover from conversion is added to the player's total only after all conversions have completed
            # This is to prevent the edge case of chaining halite from one convert to fund other converts
            leftover_convert_halite += max(delta_halite, 0)
            player._halite += min(delta_halite, 0)
            board._add_shipyard(Shipyard(ShipyardId(create_uid()), ship.position, player.id, board))
            board._delete_ship(ship)
        elif ship.next_action is not None:
          self.on_ship_move(ship)

          # If the action is not None and is not CONVERT it must be NORTH, SOUTH, EAST, or WEST
          ship.cell._ship_id = None
          ship._position = ship.position.translate(ship.next_action.to_point(), configuration.size)
          ship._halite *= (1 - board.configuration.move_cost)

          # We don't set the new cell's ship_id here as it would be overwritten by another ship in the case of collision.
          # Later we'll iterate through all ships and re-set the cell._ship_id as appropriate.

        player._halite += leftover_convert_halite
        # Lets just check and make sure.
        assert player.halite >= 0

    def resolve_collision(ships: List[Ship]) -> Tuple[Optional[Ship], List[Ship]]:
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
    ship_collision_groups = group_by(board.ships.values(), lambda ship: ship.position)
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
      if ship.next_action not in ShipAction.moves() and cell.shipyard_id is None and delta_halite > 0:
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

    board._step += 1
    return board


class Replayer:

  def __init__(self, strategy, replay_json, player_id=0):
    self.strategy = strategy
    self.replay_json = replay_json
    self.player_id = player_id
    self.env = make("halite", configuration=replay_json['configuration'],
                    steps=replay_json['steps'])
    self.step = 0
    self.total_steps = len(replay_json['steps'])

  def get_board(self, step, board_cls=EventBoard):
    state = self.replay_json['steps'][step][0]
    obs = state['observation']
    obs['player'] = self.player_id

    actions = None
    if step + 1 < self.total_steps:
      actions = [self.replay_json['steps'][step+1][p]['action']
                 for p in range(4)]
    return board_cls(obs, self.env.configuration, actions)

  def check_board_valid(self):
    for i in range(self.total_steps):
      event_board = self.get_board(i, EventBoard)
      board = self.get_board(i, Board)
      assert event_board.next().observation == board.next().observation, 'Failed check on step %s' % i
      print("Step %s, step reward = %s" % (i, event_board.step_reward))

  def simulate(self, step=0):
    board = self.get_board(step)
    self.strategy.update(board)
    self.strategy.execute()
    self.step += 1

  def play(self, steps=1):
    for i in range(steps):
      self.simulate(i)


class Trainer:

  def __init__(self, model):
    self.model = model

  def get_player_history(self, player_id):
    pass


  # def train(self, replay_json):




# replay_json = env.toJSON()
# with open("/ssd/wangfei/repo/flynn/kaggle_halite4/debug_replay/20200723_0.json", 'w') as f:
#     f.write(json.dumps(replay_json))
