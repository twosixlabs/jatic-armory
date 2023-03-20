from charmory.scenario import ScenarioRunner


class Engine:
    def __init__(self, evaluation):
        self.evaluation = evaluation

    def run(self):
        return ScenarioRunner(self.evaluation).run()
