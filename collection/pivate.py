import random


#FUNCTIONS###################################################
def get_map_and_average_rum(obs):
  """
        get average amount of rum per rum distillery
        and map as two dimensional array of dictionaries and set amounts of rum in each sector
    """
  game_map = []
  rum_distilleries_amount = 0
  rum_total_amount = 0
  for x in range(conf.size):
    game_map.append([])
    for y in range(conf.size):
      game_map[x].append({
          # value will be ID of tavern
          "tavern_name": None,
          # value will be ID of kaptin
          "tavern_kaptin": None,
          # value will be ID of ship
          "ship_name": None,
          # value will be ID of kaptin
          "ship_kaptin": None,
          # value will be amount of rum in ship's cargo hold
          "ship_cargo": None,
          # amount of rum in this sector
          "rum": obs["rum"][conf.size * y + x]
      })
      # count rum distilleries
      if game_map[x][y]["rum"] > 0:
        rum_total_amount += game_map[x][y]["rum"]
        rum_distilleries_amount += 1
  # average amount of rum on the map
  average_rum = rum_total_amount / rum_distilleries_amount
  return game_map, average_rum


def update_map(ph_env):
  """ update locations of ships and taverns on the map """
  # place on the map locations of ships and taverns of every kaptin
  for kaptin in range(len(ph_env["obs"]["kaptins"])):
    # place on the map locations of every tavern of the kaptin
    taverns = list(ph_env["obs"]["kaptins"][kaptin][1].values())
    for t in range(len(taverns)):
      x = taverns[t] % conf.size
      y = taverns[t] // conf.size
      # place tavern on the map
      ph_env["map"][x][y]["tavern_kaptin"] = kaptin
      # if it's your tavern, write its unique name on the map
      if kaptin == ph_env["obs"]["kaptin"]:
        ph_env["map"][x][y]["tavern_name"] = ph_env["taverns_names"][t]
    # place on the map locations of every ship of the kaptin
    ships = list(ph_env["obs"]["kaptins"][kaptin][2].values())
    for s in range(len(ships)):
      x = ships[s][0] % conf.size
      y = ships[s][0] // conf.size
      # place ship on the map
      ph_env["map"][x][y]["ship_kaptin"] = kaptin
      ph_env["map"][x][y]["ship_cargo"] = ships[s][1]
      # if it's your ship, write its unique name on the map
      if kaptin == ph_env["obs"]["kaptin"]:
        ph_env["map"][x][y]["ship_name"] = ph_env["ships_names"][s]


def get_c(c):
  """ get coordinate, considering donut type of the map """
  return c % conf.size


def good_place_for_tavern(x, y, ph_env):
  """ find good place to build a tavern """
  # if there is no taverns around
  if (ph_env["map"][x][y]["tavern_kaptin"] == None and
      ph_env["map"][get_c(x + 1)][y]["tavern_kaptin"] == None and
      ph_env["map"][get_c(x - 1)][y]["tavern_kaptin"] == None and
      ph_env["map"][x][get_c(y + 1)]["tavern_kaptin"] == None and
      ph_env["map"][x][get_c(y - 1)]["tavern_kaptin"] == None and
      ph_env["map"][get_c(x + 1)][get_c(y + 1)]["tavern_kaptin"] == None and
      ph_env["map"][get_c(x - 1)][get_c(y - 1)]["tavern_kaptin"] == None and
      ph_env["map"][get_c(x - 1)][get_c(y + 1)]["tavern_kaptin"] == None and
      ph_env["map"][get_c(x + 1)][get_c(y - 1)]["tavern_kaptin"] == None and
      ph_env["map"][get_c(x + 2)][y]["tavern_kaptin"] == None and
      ph_env["map"][get_c(x - 2)][y]["tavern_kaptin"] == None and
      ph_env["map"][x][get_c(y + 2)]["tavern_kaptin"] == None and
      ph_env["map"][x][get_c(y - 2)]["tavern_kaptin"] == None):
    rum_distilleries_amount = 0
    # count amount of distilleries around
    for d in directions:
      if ph_env["map"][directions[d]["x"](x)][directions[d]["y"](y)]["rum"] > 0:
        rum_distilleries_amount += 1
    if rum_distilleries_amount >= 3:
      return True
  return False


def drifting(actions, ship, ph_env):
  """ send ship drifting in search for a good place for a tavern """
  # if ship is already at that good place for a tavern
  if good_place_for_tavern(ship["x"], ship["y"], ph_env):
    ship["here_be_tavern"] = True
    return actions, ship
  to_x = None
  to_y = None
  dir_to_go = None
  keys = list(directions.keys())
  # go in random direction
  random.shuffle(keys)
  for key in keys:
    x = directions[key]["x"](ship["x"])
    y = directions[key]["y"](ship["y"])
    # if there is no ship or tavern
    if (ph_env["map"][x][y]["ship_kaptin"] == None and
        ph_env["map"][x][y]["tavern_kaptin"] == None):
      to_x = x
      to_y = y
      dir_to_go = key
      # if it is good place for a tavern
      if good_place_for_tavern(x, y, ph_env):
        ship["x"] = to_x
        ship["y"] = to_y
        actions[ship["name"]] = dir_to_go
        return actions, ship
  # if good place for a tavern has been found
  if dir_to_go != None:
    ship["x"] = to_x
    ship["y"] = to_y
    actions[ship["name"]] = dir_to_go
  return actions, ship


def define_some_globals(configuration):
  """ define some of the global variables """
  global conf
  global new_fleet_cost
  global steps_threshold
  global globals_not_defined
  conf = configuration
  new_fleet_cost = conf.spawnCost * 2 + conf.convertCost
  steps_threshold = conf.episodeSteps // 2
  globals_not_defined = False


def adapt_environment(observation, configuration):
  """ adapt environment for the pirate haven """
  ph_env = {}
  ph_env["obs"] = observation
  ph_env["taverns_names"] = list(
      ph_env["obs"]["kaptins"][ph_env["obs"]["kaptin"]][1].keys())
  ph_env["ships_names"] = list(
      ph_env["obs"]["kaptins"][ph_env["obs"]["kaptin"]][2].keys())
  ph_env["ships_logbooks"] = list(
      ph_env["obs"]["kaptins"][ph_env["obs"]["kaptin"]][2].values())
  if globals_not_defined:
    define_some_globals(configuration)
    # appoint first ship in a game
    rogue_ships.append({
        "name": ph_env["ships_names"][0],
        "here_be_tavern": False,
        "rum_reserved_by_this_ship": new_fleet_cost - conf.spawnCost,
        "x": ph_env["ships_logbooks"][0][0] % conf.size,
        "y": ph_env["ships_logbooks"][0][0] // conf.size
    })
  ph_env["map"], ph_env["average_rum"] = get_map_and_average_rum(ph_env["obs"])
  ph_env["stored_rum"] = ph_env["obs"]["kaptins"][ph_env["obs"]["kaptin"]][0]
  update_map(ph_env)
  if ph_env["obs"]["step"] == steps_threshold:
    global fleets_max_amount
    fleets_max_amount = 3
  return ph_env


def name_keys_properly(scurvy_observation):
  """ Argh! Name those scurvy observation keys properly! """
  scurvy_observation["rum"] = scurvy_observation["halite"]
  scurvy_observation["kaptin"] = scurvy_observation["player"]
  scurvy_observation["kaptins"] = scurvy_observation["players"]
  return scurvy_observation


def actions_of_fleets(ph_env):
  """ actions of every fleet of the kaptin """
  actions = {}
  for i in range(len(fleets))[::-1]:
    tactics_applied = False
    for tactics in fleets[i]["tactics"]:
      tactics_applied, fleets[i], actions = tactics(fleets[i], actions, ph_env)
      # if tactics successfully applied, don't try to apply any other tactics
      if tactics_applied:
        break
    if not tactics_applied:
      # disband fleet if no tactics could be applied
      disband_fleet(fleets[i], i)
  fleet_assembled = True
  # if possible, assemble new fleets from rogue ships
  while fleet_assembled and len(rogue_ships) > 0 and len(
      fleets) < fleets_max_amount:
    fleet_assembled, actions = assemble_new_fleet(actions, ph_env,
                                                  [one_guard_one_tavern])
  return actions


def assemble_new_fleet(actions, ph_env, tactics_list):
  """ assemble new fleet and apply first applicable tactics from tactics_list """
  fleet = {}
  fleet["tactics"] = tactics_list
  fleet["ships"] = []
  fleet["taverns"] = []
  tactics_applied = False
  for tactics in fleet["tactics"]:
    tactics_applied, fleet, actions = tactics(fleet, actions, ph_env)
    # if tactics successfully applied, don't try to apply other tactics from the tactics_list
    if tactics_applied:
      break
  # if no tactics from tactics_list were applied, do not assemble new fleet
  if not tactics_applied:
    return False, actions
  fleets.append(fleet)
  return True, actions


def disband_fleet(fleet, i):
  """ transfer all existing fleet's ships to rogues and disband fleet """
  rogue_ships.extend(fleet["ships"])
  fleets.pop(i)


def roll_call(ph_env):
  """ name all nameless and remove all non-existent ships and taverns """
  global reserved_rum
  for fleet in fleets:
    for i in range(len(fleet["ships"]))[::-1]:
      if fleet["ships"][i]["name"] == None:
        fleet["ships"][i]["name"] = ph_env["map"][fleet["ships"][i]["x"]][
            fleet["ships"][i]["y"]]["ship_name"]
      if fleet["ships"][i]["name"] not in ph_env["ships_names"]:
        reserved_rum -= fleet["ships"][i]["rum_reserved_by_this_ship"]
        fleet["ships"].pop()
    for i in range(len(fleet["taverns"]))[::-1]:
      if fleet["taverns"][i]["name"] == None:
        fleet["taverns"][i]["name"] = ph_env["map"][fleet["taverns"][i]["x"]][
            fleet["taverns"][i]["y"]]["tavern_name"]
      if fleet["taverns"][i]["name"] not in ph_env["taverns_names"]:
        reserved_rum -= fleet["taverns"][i]["rum_reserved_by_this_tavern"]
        fleet["taverns"].pop()
  for ship in rogue_ships[::-1]:
    if ship["name"] == None:
      ship["name"] = ph_env["map"][ship["x"]][ship["y"]]["ship_name"]
    if ship["name"] not in ph_env["ships_names"]:
      reserved_rum -= ship["rum_reserved_by_this_ship"]
      rogue_ships.pop()


#TACTICS################################################
def one_guard_one_tavern(fleet, actions, ph_env):
  """
        this tactics requires one tavern and one ship to guard that tavern
        and collect rum from neighbouring rum distilleries
    """
  global reserved_rum
  # amount of units required for this tactics
  ships_required = 1
  taverns_required = 1
  # if there is currently no ships or taverns in this fleet
  if len(fleet["ships"]) != ships_required and len(
      fleet["taverns"]) != taverns_required:
    # if there are some rogue ships available
    if len(rogue_ships) > 0:
      # transfer ship from rogues to this fleet
      fleet["ships"].append(rogue_ships.pop())
    # else tactics can't be applied
    else:
      return False, fleet, actions
  # if this fleet is currently full of ships and taverns
  # and there is enough rum to assemble new fleet
  # and this kaptin has less then max amount of fleets
  if (len(fleet["taverns"]) == taverns_required and
      len(fleet["ships"]) == ships_required and
      (ph_env["stored_rum"] - reserved_rum) >= new_fleet_cost and
      len(fleets) < fleets_max_amount):
    fleet["ships"][0]["here_be_tavern"] = False
    fleet["ships"][0][
        "rum_reserved_by_this_ship"] = new_fleet_cost - conf.spawnCost
    # transfer current ship to rogues
    rogue_ships.append(fleet["ships"].pop())
    reserved_rum += new_fleet_cost
  # if there is not enough ships in this fleet
  if len(fleet["ships"]) != ships_required:
    # if there is enough taverns in this fleet
    # and no ship of this kaptin is currently at this tavern
    # and there is enough rum to build a new ship
    if (len(fleet["taverns"]) == taverns_required and
        ph_env["map"][fleet["taverns"][0]["x"]][fleet["taverns"][0]["y"]]
        ["ship_kaptin"] != ph_env["obs"]["kaptin"] and
        ph_env["stored_rum"] >= conf.spawnCost):
      ph_env["stored_rum"] -= conf.spawnCost
      reserved_rum -= conf.spawnCost
      if reserved_rum < 0:
        reserved_rum = 0
      actions[fleet["taverns"][0]["name"]] = "SPAWN"
      # place ship of this kaptin at this tavern to avoid collisions
      ph_env["map"][fleet["taverns"][0]["x"]][
          fleet["taverns"][0]["y"]]["ship_kaptin"] = ph_env["obs"]["kaptin"]
      # if there is enough rum to assemble new fleet
      # and this kaptin has less then max amount of fleets
      # newly built ship will be appointed to rogues and new fleet assembled
      if ((ph_env["stored_rum"] - reserved_rum) >= new_fleet_cost and
          len(fleets) < fleets_max_amount):
        reserved_rum += new_fleet_cost
        fleet["taverns"][0]["rum_reserved_by_this_tavern"] = conf.spawnCost
        rogue_ships.append({
            "name": None,
            "here_be_tavern": False,
            "rum_reserved_by_this_ship": new_fleet_cost - conf.spawnCost,
            "x": fleet["taverns"][0]["x"],
            "y": fleet["taverns"][0]["y"]
        })
        assemble_new_fleet({}, ph_env, [one_guard_one_tavern])
        return True, fleet, actions
      # otherwise simply appoint newly built ship to current fleet
      fleet["taverns"][0]["rum_reserved_by_this_tavern"] = 0
      fleet["ships"].append({
          "name": None,
          "here_be_tavern": False,
          "rum_reserved_by_this_ship": 0,
          "x": fleet["taverns"][0]["x"],
          "y": fleet["taverns"][0]["y"]
      })
    return True, fleet, actions
  ship = fleet["ships"][0]
  # if there is currently not enough taverns
  if len(fleet["taverns"]) != taverns_required:
    # if the ship is named
    if ship["name"] != None:
      ph_env["map"][ship["x"]][ship["y"]]["ship_kaptin"] = None
      # send ship drifting
      actions, ship = drifting(actions, ship, ph_env)
      ph_env["map"][ship["x"]][
          ship["y"]]["ship_kaptin"] = ph_env["obs"]["kaptin"]
    # if ship is currently at a good place for tavern
    if ship["here_be_tavern"]:
      # if there is enough rum to build a tavern
      if ph_env["stored_rum"] >= conf.convertCost:
        actions[ship["name"]] = "CONVERT"
        ph_env["map"][ship["x"]][
            ship["y"]]["tavern_kaptin"] = ph_env["obs"]["kaptin"]
        x = ship["x"]
        y = ship["y"]
        # subtract cost of a tavern from ship's cargo and remaining cost, if any, from stored rum
        ph_env["map"][x][y]["ship_cargo"] -= conf.convertCost
        if ph_env["map"][x][y]["ship_cargo"] < 0:
          ph_env["stored_rum"] += ph_env["map"][x][y]["ship_cargo"]
          ph_env["map"][x][y]["ship_cargo"] = 0
        ship["rum_reserved_by_this_ship"] = 0
        reserved_rum -= conf.convertCost
        if reserved_rum < 0:
          reserved_rum = 0
        ph_env["map"][x][y]["ship_kaptin"] = None
        fleet["taverns"].append({
            "name": None,
            "rum_reserved_by_this_tavern": conf.spawnCost,
            "x": x,
            "y": y
        })
    return True, fleet, actions
  ship_x = ship["x"]
  ship_y = ship["y"]
  # if ship is now at the fleet's tavern and it is safe to set sail
  if (ship_x == fleet["taverns"][0]["x"] and
      ship_y == fleet["taverns"][0]["y"] and
      ((len(ph_env["taverns_names"]) >= fleets_max_amount and
        len(ph_env["ships_names"]) >= fleets_max_amount) or
       ph_env["obs"]["step"] < steps_threshold)):
    # Shiver me timbers! They're pillaging our rum distillery! Board 'em!!!
    for direction in directions:
      x = directions[direction]["x"](ship_x)
      y = directions[direction]["y"](ship_y)
      # if there is no tavern and hostile kaptin's ship has more rum
      if (ph_env["map"][x][y]["ship_kaptin"] != None and
          ph_env["map"][x][y]["tavern_kaptin"] == None and
          ph_env["map"][x][y]["ship_kaptin"] != ph_env["obs"]["kaptin"] and
          ph_env["map"][x][y]["ship_cargo"] >
          ph_env["map"][ship_x][ship_y]["ship_cargo"]):
        actions[ship["name"]] = direction
        ph_env["map"][x][y]["ship_kaptin"] = ph_env["obs"]["kaptin"]
        ph_env["map"][ship_x][ship_y]["ship_kaptin"] = None
        ship["x"] = x
        ship["y"] = y
        return True, fleet, actions
    # collect that rum before it's gone
    for direction in directions:
      x = directions[direction]["x"](ship_x)
      y = directions[direction]["y"](ship_y)
      # if there is a lot of rum and none of this kaptin's ships
      if ph_env["map"][x][y]["rum"] >= 450 and ph_env["map"][x][y][
          "ship_kaptin"] != ph_env["obs"]["kaptin"]:
        actions[ship["name"]] = direction
        ph_env["map"][x][y]["ship_kaptin"] = ph_env["obs"]["kaptin"]
        ph_env["map"][ship_x][ship_y]["ship_kaptin"] = None
        ship["x"] = x
        ship["y"] = y
        return True, fleet, actions
  # else, if there is low amount of rum in distillery, go back to the fleet's tavern for debauchery
  elif ph_env["map"][ship_x][ship_y]["rum"] < 376:
    for direction in directions:
      x = directions[direction]["x"](ship_x)
      y = directions[direction]["y"](ship_y)
      # if there is current fleet's tavern and none of this kaptin's ships
      if (x == fleet["taverns"][0]["x"] and y == fleet["taverns"][0]["y"] and
          ph_env["map"][x][y]["ship_kaptin"] != ph_env["obs"]["kaptin"]):
        actions[ship["name"]] = direction
        ph_env["map"][x][y]["ship_kaptin"] = ph_env["obs"]["kaptin"]
        ph_env["map"][ship_x][ship_y]["ship_kaptin"] = None
        ship["x"] = x
        ship["y"] = y
        return True, fleet, actions
  return True, fleet, actions


#GLOBAL_VARIABLES#############################################
conf = None
steps_threshold = None
# amount of rum needed to assemble new fleet
# one ship to convert to tavern, conversion cost, one ship to guard the tavern
new_fleet_cost = None
# list of currently existing fleets
fleets = []
# list of ships that doesn't belong to any fleet
rogue_ships = []
# rum reserved for future debauchery
reserved_rum = 0
# maximum amounts of fleets at any step
fleets_max_amount = 20
# not all global variables are defined
globals_not_defined = True

# dictionary of directions
directions = {
    "NORTH": {
        "x": lambda z: z,
        "y": lambda z: get_c(z - 1)
    },
    "EAST": {
        "x": lambda z: get_c(z + 1),
        "y": lambda z: z
    },
    "SOUTH": {
        "x": lambda z: z,
        "y": lambda z: get_c(z + 1)
    },
    "WEST": {
        "x": lambda z: get_c(z - 1),
        "y": lambda z: z
    }
}


#%Pirate_Haven%####################################################
def pirate_haven(scurvy_observation, configuration):
  """ Yo-ho-ho and a Bottle of Rum!!! """
  observation = name_keys_properly(scurvy_observation)
  ph_env = adapt_environment(observation, configuration)
  roll_call(ph_env)
  actions = actions_of_fleets(ph_env)
  return actions
