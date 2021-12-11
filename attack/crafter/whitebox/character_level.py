"""
Whitebox Character Level Attack attention scores
=================================================================

"""
from textattack import Attack
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    MinWordLength,
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from utils.greedy_word_swap_wir import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

from textattack.attack_recipes.attack_recipe import AttackRecipe


class WhiteboxCharacterLevel(AttackRecipe):
    """An extension of the attack used in "Combating Adversarial
    Misspellings with Robust Word Recognition", Pruthi et al., 2019. Where we compute word importances using attentions
    scores for character level perturbations

    This attack focuses on a small number of character-level changes that simulate common typos. It combines:
        - Swapping neighboring characters
        - Deleting characters
        - Inserting characters
        - Swapping characters for adjacent keys on a QWERTY keyboard.

    https://arxiv.org/abs/1905.11268

    :param model: Model to attack.
    :param max_num_word_swaps: Maximum number of modifications to allow.
    """

    @staticmethod
    def build(model_wrapper, model_name=None, max_percent=0.5):
        # a combination of 4 different character-based transforms
        # ignore the first and last letter of each word, as in the paper
        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapQWERTY(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
            ]
        )
        # only edit words of length >= 4, edit max_num_word_swaps words.
        # note that we also are not editing the same word twice, so
        # max_num_word_swaps is really the max number of character
        # changes that can be made. The paper looks at 1 and 2 char attacks.
        constraints = [
            MinWordLength(min_length=4),
            StopwordModification(),
            MaxWordsPerturbed(max_percent=max_percent),
            RepeatModification(),
        ]
        # untargeted attack
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="attention", model_name=model_name)
        return Attack(goal_function, constraints, transformation, search_method)
