from executor.execute import ExecuteAttack
from model.model import GermanHateSpeechModelWrapper
from dataset.dataset import GermanDataset
from crafter.blackbox_word_level import BlackboxWordLevelAttack


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
    ## define executor and execute the attack
    executor = ExecuteAttack()
    executor.execute(hasoc_dataset, attacks=[blackbox_wordlevel_attack])


if __name__ == "__main__":
    main()
