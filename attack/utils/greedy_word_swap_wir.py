"""
Greedy Word Swap with Word Importance Ranking
===================================================


When WIR method is set to ``unk``, this is a reimplementation of the search
method from the paper: Is BERT Really Robust?

A Strong Baseline for Natural Language Attack on Text Classification and
Entailment by Jin et. al, 2019. See https://arxiv.org/abs/1907.11932 and
https://github.com/jind11/TextFooler.
"""

import numpy as np
import torch
import operator
import copy
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)


class GreedyWordSwapWIR(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(self, wir_method="unk", model_name=None):
        self.wir_method = wir_method
        if wir_method == "attention":
            attention_model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            attention_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.cls_explainer = SequenceClassificationExplainer(
                attention_model, attention_tokenizer
            )

    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""
        len_text = len(initial_text.words)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "random":
            index_order = np.arange(len_text)
            np.random.shuffle(index_order)
            search_over = False

        elif self.wir_method == "attention":
            index_order = np.arange(len_text)
            clean_text = " ".join(initial_text.words)
            index_scores = self.get_attention_scores(text=clean_text)
            search_over = False

        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = (-index_scores).argsort()

        return index_order, search_over

    def get_attention_scores(self, text):
        token_scores = []
        token_importance = []
        try:
            word_attributions = self.cls_explainer(text)
            for token, score in word_attributions:
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                if "##" in token:
                    token_scores[-1] = (
                        token_scores[-1][0] + token.replace("##", ""),
                        token_scores[-1][1] + score,
                    )
                else:
                    token_scores.append((token, score))

            token_scores_sorted = copy.deepcopy(token_scores)
            token_scores_sorted.sort(key=operator.itemgetter(1))
            for token, score in token_scores_sorted:
                index = token_scores.index((token, score))
                token_importance.append(index)
        except Exception as e:
            print(f"Error getting attention scores {e}")

        return np.array(token_importance)

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text

        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)

        i = 0
        cur_result = initial_result
        results = None
        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result

        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "attention":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]
