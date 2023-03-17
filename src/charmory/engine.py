from armory.scenarios.main import main as scenario_main


class Engine:
  def __init__(self, evaluation):
    self.evaluation = evaluation

  def run(self):
    return scenario_main(self.evaluation)
