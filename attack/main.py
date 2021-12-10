from executor.execute import ExecuteAttack
from model.model import GermanHateSpeechModelWrapper
from dataset.dataset import GermanDataset
from crafter import BlackboxWordLevelAttack, BlackboxCharacterLevel


def main():
    """
    Crafting an attack on HASOC based trained BERT model
    """
    ## load dataset
    hasoc_dataset = GermanDataset(
        filepath="/Users/shahrukh/Desktop/adversarial-bert-german-attacks-defense/attack/dataset/hasoc_german_test_subset.csv"
    ).load_dataset()
    ## load model
    hasoc_model_wrapper = GermanHateSpeechModelWrapper(
        model_name_path="/Users/shahrukh/Desktop/victim_models/hasoc_model"
    )
    ## define and build attacks
    blackbox_wordlevel_attack = BlackboxWordLevelAttack.build(hasoc_model_wrapper)
    blackbox_charlevel_attack = BlackboxCharacterLevel.build(hasoc_model_wrapper)
    attacks = [blackbox_charlevel_attack, blackbox_wordlevel_attack]
    ## execute the attack
    for attack in attacks:
        ExecuteAttack.execute(hasoc_dataset, attacks=[attack])


if __name__ == "__main__":
    main()
