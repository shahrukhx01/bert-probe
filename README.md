# BERT Probe
A python package for probing attention based robustness to character and word based adversarial evaluation. Also, with recipes of implicit and explicit defenses against character-level attacks.

**Links to code sections**
- [BERT Finetunig](https://github.com/shahrukhx01/adversarial-bert-german-attacks-defense/tree/main/bert_finetuning)
- [Attacks: Whitebox Baseline, Character-level and Word-level](https://github.com/shahrukhx01/adversarial-bert-german-attacks-defense/tree/main/attack/crafter/whitebox)
```python
  logs_path = "./attack_logs"
  attack_config = [
        (
            "hasoc",
            "shahrukhx01/gbert-hasoc-german-2019",
            "data/hasoc_german_2019/hasoc_german_probing_set.csv",
        ),
        (
             "germeval",
             "shahrukhx01/gbert-germeval-2021",
             "data/hasoc_german_2019/germeval_probing_set.csv",
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
    attacks = [
            (
                f"{attack_name}_blackbox_wordlevel_attack",
                blackbox_wordlevel_attack,
            ),
            (
                f"{attack_name}_blackbox_charlevel_attack",
                blackbox_charlevel_attack,
            ),
        ]
     ## execute the attack
    ExecuteAttack.execute(dataset, attacks=attacks, logs_path=logs_path)
```
- [Defenses: Explicit Character-level and Abstain label training](https://github.com/shahrukhx01/adversarial-bert-german-attacks-defense/tree/main/defense) <br/>
**Datasets**:
- Germeval 2021 Task 1: Toxic Comment Classification
- HASOC (2019) German Language: Sub Task 1, Hate Speech Classification


## Citing & Authors

If you find this repository helpful, feel free to cite our publication:


```bibtex
@inproceedings{bertprobe,
  author    = {Shahrukh Khan and
               Mahnoor Shahid and
               Navdeeppal Singh},
  title     = {White-Box Attacks on Hate-speech BERT Classifiers in German with Explicit and Implicit Character Level Defense},
  booktitle = {BOHR International Journal of Intelligent Instrumentation and Computing, 2022},
  publisher = {BOHR Publishers},
  year      = {2022},
  url       = {https://bohrpub.com/journals/BIJIIAC/Vol1N1/BIJIIAC_20221104.html}
}
```
