# Generated by CodiumAI

import pytest

from charmory.evaluation import Attack

"""
Code Analysis:
- The class 'Attack' is a dataclass used to store information about an attack in the Armory Experiment Configuration.
- It has five fields: 'function', 'kwargs', 'knowledge', 'use_label', and 'type'.
- 'function' is a reference to a python method that represents the attack.
- 'kwargs' is a dictionary of string keys and any values that represent the arguments to the attack method.
- 'knowledge' is a string literal that represents the type of knowledge the attacker has about the target model, either "white" or "black".
- 'use_label' is a boolean that indicates whether the attack should use the true label of the input during the attack.
- 'type' is an optional string that represents the type of attack being performed.
- The class is used to create instances of attacks that can be executed on a target model.
- The 'function' field is used to specify the attack method to be executed.
- The 'kwargs' field is used to specify the arguments to the attack method.
- The 'knowledge' field is used to specify the type of knowledge the attacker has about the target model.
- The 'use_label' field is used to specify whether the attack should use the true label of the input during the attack.
- The 'type' field is used to specify the type of attack being performed, if applicable.
- The class is used in conjunction with other classes and methods in the Armory Experiment Configuration to configure and execute attacks on target models.
"""


class TestAttack:
    # Tests that an instance of attack can be created with all required fields specified. tags: [happy path]
    def test_create_attack_with_required_fields(self):
        attack = Attack(
            function="armory.attacks.fgsm", kwargs={"eps": 0.3}, knowledge="white"
        )
        assert isinstance(attack, Attack)
        assert attack.function == "armory.attacks.fgsm"
        assert attack.kwargs == {"eps": 0.3}
        assert attack.knowledge == "white"
        assert not attack.use_label
        assert attack.type is None

    # Tests that an instance of attack can be created with all fields specified, including the optional 'type' field. tags: [happy path]
    def test_create_attack_with_all_fields_specified(self):
        attack = Attack(
            function="armory.attacks.pgd",
            kwargs={"eps": 0.3, "alpha": 0.1},
            knowledge="black",
            use_label=True,
            type="untargeted",
        )
        assert isinstance(attack, Attack)
        assert attack.function == "armory.attacks.pgd"
        assert attack.kwargs == {"eps": 0.3, "alpha": 0.1}
        assert attack.knowledge == "black"
        assert attack.use_label
        assert attack.type == "untargeted"

    # Tests that an instance of attack cannot be created with invalid fields. tags: [edge case]
    def test_create_attack_with_invalid_fields(self):
        with pytest.raises(TypeError):
            Attack(function=10)  # type: ignore

    # Tests that an attack can be executed on a target model using an instance of attack with valid fields. tags: [happy path]
    def test_execute_attack_on_target_model(self):
        # TODO: Implement this test
        # attack = Attack(
        #     function="armory.attacks.fgsm", kwargs={"eps": 0.3}, knowledge="white"
        # )
        pass

    # Tests that an attack can be executed on a target model using an instance of attack with valid fields and the 'use_label' field set to true. tags: [happy path]
    def test_execute_attack_on_target_model_with_use_label(self):
        # TODO: Implement this test
        # attack = Attack(
        #     function="armory.attacks.fgsm",
        #     kwargs={"eps": 0.3},
        #     knowledge="white",
        #     use_label=True,
        # )
        pass

    # Tests that an attack can be executed on a target model using an instance of attack with valid fields and the 'type' field set to a valid value. tags: [happy path]
    def test_execute_attack_on_target_model_with_valid_type(self):
        # TODO: Implement this test
        pass
