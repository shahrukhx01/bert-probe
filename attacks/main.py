from executor.execute import ExecuteAttack
from model.model import GermanHateSpeechModelWrapper
from dataset.dataset import GermanDataset
from crafter import (
    BlackboxWordLevelAttack,
    BlackboxCharacterLevel,
    WhiteboxWordLevelAttack,
    WhiteboxCharacterLevel,
    BaselineWhiteboxWordLevelAttack,
)


def main():
    """
    Crafting an attack on German based trained BERT models for GermEval2021 and HASOC FIRE 2019 datasets
    """
    logs_path = "./attack_logs"
    attack_config = [
        (
            "hasoc",
            "shahrukhx01/gbert-hasoc-german-2019",
            "../data/hasoc_german_2019/hasoc_german_probing_set.csv",
        ),
        # (
        #     "germeval",
        #     # "shahrukhx01/gbert-germeval-2021",
        #     # "../data/hasoc_german_2019/germeval_probing_set.csv",
        # ),
    ]
    for attack_name, model_name_path, dataset_path in attack_config:
        ## load dataset
        dataset = GermanDataset(
            filepath=dataset_path
        ).load_dataset()  ### sampling = False
        ## load model
        model_wrapper = GermanHateSpeechModelWrapper(model_name_path=model_name_path)
        ## define and build attacks
        blackbox_wordlevel_attack = BlackboxWordLevelAttack.build(model_wrapper)
        blackbox_charlevel_attack = BlackboxCharacterLevel.build(model_wrapper)
        whitebox_wordlevel_attack = WhiteboxWordLevelAttack.build(
            model_wrapper=model_wrapper, model_name=model_name_path
        )
        whitebox_charlevel_attack = WhiteboxCharacterLevel.build(
            model_wrapper=model_wrapper, model_name=model_name_path
        )
        baseline_whitebox_wordlevel_attack = BaselineWhiteboxWordLevelAttack.build(
            model_wrapper=model_wrapper, model_name=model_name_path
        )

        attacks = [
            (
                f"{attack_name}_blackbox_wordlevel_attack",
                blackbox_wordlevel_attack,
            ),
            (
                f"{attack_name}_blackbox_charlevel_attack",
                blackbox_charlevel_attack,
            ),
            (
                f"{attack_name}_whitebox_wordlevel_attack",
                whitebox_wordlevel_attack,
            ),
            (
                f"{attack_name}_baseline_whitebox_wordlevel_attack",
                baseline_whitebox_wordlevel_attack,
            ),
            (
                f"{attack_name}_whitebox_charlevel_attack",
                whitebox_charlevel_attack,
            ),
        ]

        ## execute the attack
        ExecuteAttack.execute(dataset, attacks=attacks, logs_path=logs_path)


if __name__ == "__main__":
    main()
