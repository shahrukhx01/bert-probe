from executor.execute import ExecuteAttack
from model.model import GermanHateSpeechModelWrapper
from dataset.dataset import GermanDataset
from crafter import (
    BlackboxWordLevelAttack,
    BlackboxCharacterLevel,
    WhiteboxWordLevelAttack,
    WhiteboxCharacterLevel,
)


def main():
    """
    Crafting an attack on HASOC based trained BERT model
    """
    ## load dataset
    hasoc_model_name_path = "/Users/shahrukh/Desktop/victim_models/hasoc_model"
    hasoc_dataset = GermanDataset(
        filepath="/Users/shahrukh/Desktop/adversarial-bert-german-attacks-defense/attack/dataset/hasoc_german_test_subset.csv"
    ).load_dataset()
    ## load model
    hasoc_model_wrapper = GermanHateSpeechModelWrapper(
        model_name_path=hasoc_model_name_path
    )
    ## define and build attacks
    """
    blackbox_wordlevel_attack = BlackboxWordLevelAttack.build(hasoc_model_wrapper)
    blackbox_charlevel_attack = BlackboxCharacterLevel.build(hasoc_model_wrapper)
    whitebox_wordlevel_attack = WhiteboxWordLevelAttack.build(
        model_wrapper=hasoc_model_wrapper, model_name=hasoc_model_name_path
    )
    """
    whitebox_charlevel_attack = WhiteboxCharacterLevel.build(
        model_wrapper=hasoc_model_wrapper, model_name=hasoc_model_name_path
    )

    attacks = [
        whitebox_charlevel_attack
    ]  # , blackbox_wordlevel_attack, blackbox_charlevel_attack]

    ## execute the attack
    for attack in attacks:
        ExecuteAttack.execute(hasoc_dataset, attacks=[attack])


if __name__ == "__main__":
    main()
