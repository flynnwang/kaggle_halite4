from kaggle_environments import make


class Replayer:

  def __init__(self, strategy, replay_json, player_id=0):
    self.strategy = strategy
    self.replay_json = replay_json
    self.player_id = player_id
    self.env = make("halite",
                    configuration=replay_json['configuration'],
                    steps=replay_json['steps'])
    self.step = 0

    def play(self, steps=1):
      for i in range(steps):
        self.simulate(i)
        self.step += 1

    def simulate(self, step=0):
      # list of length 1 for each step
      state = self.replay_json['steps'][step][0]
      obs = state['observation']
      obs['player'] = self.player_id
      board = Board(obs, self.env.configuration)
      self.strategy.update(board)
      self.strategy.execute()
