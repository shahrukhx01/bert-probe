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
    Crafting an attack on HASOC based trained BERT model
    """
    attack_config = [
        (
            "hasoc",
            "/Users/shahrukh/Desktop/victim_models/hasoc_model",
            "/Users/shahrukh/Desktop/adversarial-bert-german-attacks-defense/attack/dataset/hasoc_german_test_subset.csv",
        ),
        (
            "germeval",
            "/Users/shahrukh/Desktop/victim_models/germeval_model",
            "/Users/shahrukh/Desktop/adversarial-bert-german-attacks-defense/attack/dataset/germ_eval_test_subset.csv",
        ),
    ]
    for attack_name, model_name_path, dataset_path in attack_config:
        ## load dataset
        hasoc_dataset = GermanDataset(filepath=dataset_path).load_dataset()
        ## load model
        hasoc_model_wrapper = GermanHateSpeechModelWrapper(
            model_name_path=model_name_path
        )
        ## define and build attacks
        blackbox_wordlevel_attack = BlackboxWordLevelAttack.build(hasoc_model_wrapper)
        blackbox_charlevel_attack = BlackboxCharacterLevel.build(hasoc_model_wrapper)
        whitebox_wordlevel_attack = WhiteboxWordLevelAttack.build(
            model_wrapper=hasoc_model_wrapper, model_name=model_name_path
        )
        whitebox_charlevel_attack = WhiteboxCharacterLevel.build(
            model_wrapper=hasoc_model_wrapper, model_name=model_name_path
        )
        baseline_whitebox_wordlevel_attack = BaselineWhiteboxWordLevelAttack.build(
            model_wrapper=hasoc_model_wrapper, model_name=model_name_path
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
        ExecuteAttack.execute(hasoc_dataset, attacks=attacks)


if __name__ == "__main__":
    main()
