# BERT Probe: A python package for probing attention based robustness evaluation of BERT models
Evaluates BERT models on character and word based adversarial attacks. Also, presents recipes of implicit and explicit defenses against character-level attacks.

## Usage
```bash
# install dependencies
pip install -r requirements.txt
```

```python
import stanza

# download stanza german model
stanza.download('de')
```
- [BERT Finetunig](https://github.com/shahrukhx01/bert-probe/tree/main/training_bert)

```python
  epochs = 10
  num_labels = 2
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  data_path = {
      "train": "./datasets/hasoc_dataset/hasoc_german_train.csv",
      "dev": "./datasets/hasoc_dataset/hasoc_german_validation.csv",
      "test": "./datasets/hasoc_dataset/hasoc_german_test.csv",
  }
  model_name = "deepset/gbert-base"
  data_loaders = GermanDataLoader(
      data_path, model_name, do_cleansing=False, max_sequence_length=128, batch_size=8
  )
  model = BERTClassifier(num_labels=num_labels).get_model()
  optim_config = BertOptimConfig(
      model=model, train_dataloader=data_loaders.train_dataloader, epochs=epochs
  )

  ## execute the training routine
  model = train_model(
      model=model,
      optimizer=optim_config.optimizer,
      scheduler=optim_config.scheduler,
      train_dataloader=data_loaders.train_dataloader,
      validation_dataloader=data_loaders.validation_dataloader,
      epochs=epochs,
      device=device,
      model_name=model_name,
  )

  ## test model performance on unseen test set
  eval_model(model=model, test_dataloader=data_loaders.test_dataloader, device=device)
```
- [Attacks: Whitebox Baseline, Character-level and Word-level](https://github.com/shahrukhx01/bert-probe/blob/main/attacks/main.py)
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
        ),
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
- [Defenses: Explicit Character-level and Abstain label training](https://github.com/shahrukhx01/bert-probe/tree/main/defenses) <br/>
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
