from importlib import import_module
import time

from tqdm import tqdm


class Engine:
    def __init__(self, evaluation):
        if not hasattr(evaluation, "eval_id"):
            evaluation.eval_id = str(time.time())

        # TODO: Refactor the dynamic import mechanism. -CW
        scenario_module, scenario_method = evaluation.scenario.function.split(":")
        ScenarioClass = getattr(import_module(scenario_module), scenario_method)

        self.evaluation = evaluation
        self.model = self.evaluation.model
        self.dataset = self.evaluation.dataset
        self.attack = self.evaluation.attack
        self.scenario = ScenarioClass(self.evaluation)

    def run(self, export=False):
        results = self.scenario.evaluate() if not export else self.export_run()

        self.dataset = self.scenario.dataset
        self.model = self.scenario._loaded_model

        return results

    def export_run(self):
        scenario = self.scenario
        # TODO: Refactor this to use the new exporter API. -CW
        for _ in tqdm(range(len(scenario.test_dataset)), desc="Evaluation"):
            scenario.next()
            if not scenario.skip_benign:
                scenario.run_benign()
                try:
                    scenario.sample_exporter.export(
                        scenario.x[0], f"benign_batch_{scenario.i}", with_boxes=False
                    )
                    if scenario.y[0] != None and scenario.y_pred[0] != None:
                        scenario.sample_exporter.export(
                            scenario.x[0],
                            f"benign_batch_{scenario.i}_bbox",
                            y=scenario.y[0],
                            y_pred=scenario.y_pred[0],
                            with_boxes=True,
                        )
                except:
                    pass
            if not scenario.skip_attack:
                scenario.run_attack()
                try:
                    scenario.sample_exporter.export(
                        scenario.x_adv[0],
                        f"adversarial_batch_{scenario.i}",
                        with_boxes=False,
                    )
                    if scenario.y[0] != None or scenario.y_pred_adv[0] != None:
                        scenario.sample_exporter.export(
                            scenario.x_adv[0],
                            f"adversarial_batch_{scenario.i}_bbox",
                            y=scenario.y[0],
                            y_pred=scenario.y_pred_adv[0],
                            with_boxes=True,
                        )
                except:
                    pass

            scenario.hub.set_context(stage="finished")
            scenario.finalize_results()
            results_dict = scenario.prepare_results()
            scenario.save()
            return results_dict["results"]
