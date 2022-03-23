import os.path as osp


class DataPaths:
    def __init__(self,  root_path=None):
        if root_path is not None:
            self._add_root_path(root_path, self.HASOC)
            self._add_root_path(root_path, self.GERMEVAL)

    HASOC = {
        "train": "bert_finetuning/datasets/hasoc_dataset/hasoc_german_train.csv",
        "dev": "bert_finetuning/datasets/hasoc_dataset/hasoc_german_validation.csv",
        "test": "bert_finetuning/datasets/hasoc_dataset/hasoc_german_test.csv",

        "adversarial": {
            "train": "defense/abstain_training/datasets/results-hasoc_whitebox_charlevel_attack_train.csv",
            "test": "defense/explicit_character_level/datasets/perturbed_sets/results-hasoc_whitebox_charlevel_attack.csv",
        },
    }

    GERMEVAL = {
        "train": "bert_finetuning/datasets/germeval_dataset/germ_eval_train.csv",
        "dev": "bert_finetuning/datasets/germeval_dataset/germ_eval_validation.csv",
        "test": "bert_finetuning/datasets/germeval_dataset/germ_eval_test.csv",

        "adversarial": {
            "train": "defense/abstain_training/datasets/results-germeval_whitebox_charlevel_attack.csv",
            "test": "defense/explicit_character_level/datasets/perturbed_sets/results-germeval_whitebox_charlevel_attack.csv",
        },
    }

    def _add_root_path(self, root_path, to_dict):
        """
        Recursively adds root_path to each leaf to_dict using os.path.join.
        """
        for k, v in to_dict:
            if isinstance(v, dict):
                self._add_root_path(root_path, v)
            else:
                to_dict[k] = osp.join(root_path, v)
