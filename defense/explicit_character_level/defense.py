from character_embeddings import CharacterEmbeddings
from model import GermanHateSpeechModel
import pandas as pd
import os
from tqdm import tqdm

tqdm.pandas(desc="my bar!")


def remove_tags(sequence):
    colors = ["red", "green"]
    sequence = sequence.replace("</font>", "")
    for color in colors:
        sequence = sequence.replace(f"<font color = {color}>", "")
    return sequence


def main():
    char_embedder = CharacterEmbeddings()
    basepath = "/Users/shahrukh/Desktop/adversarial-bert-german-attacks-defense/defense/datasets"
    models = {
        "hasoc": GermanHateSpeechModel(
            "/Users/shahrukh/Desktop/victim_models/hasoc_model"
        ),
        "germeval": GermanHateSpeechModel(
            "/Users/shahrukh/Desktop/victim_models/germeval_model"
        ),
    }
    for filename in os.listdir(basepath):
        if "defense" in filename:
            continue
        successful, failed = 0, 0
        result_name = f"{filename}_defense".replace(".csv", "")
        dataset = pd.read_csv(f"{basepath}/{filename}")
        dataset = dataset[
            dataset["result_type"] == "Successful"
        ].head()  ## TODO: Revert head()
        model = models["hasoc"]
        if "germeval" in result_name:
            model = models["germeval"]
        for idx, row in tqdm(dataset.iterrows()):
            perturbed_sequence = remove_tags(row["perturbed_text"])
            original_sequence = remove_tags(row["original_text"])

            perturbed_tokens = perturbed_sequence.split(" ")
            original_tokens = original_sequence.split(" ")

            perturbed_embeddings = char_embedder.get_sequence_embeddings(
                perturbed_sequence
            )
            ground_truth_embeddings = char_embedder.get_sequence_embeddings(
                original_sequence
            )
            perturbed_update = perturbed_tokens
            for index, _ in enumerate(perturbed_tokens):
                best_candidate_idx = char_embedder.cos_sim(
                    perturbed_embeddings[index, :], ground_truth_embeddings
                ).argmax()
                perturbed_update[index] = original_tokens[best_candidate_idx]
            defended_sequence = " ".join(perturbed_update)
            gt = int(row["ground_truth_output"])
            label = f"LABEL_{gt}"
            prediction = model([defended_sequence])[0]["label"]
            if prediction == label:
                successful += 1
            else:
                failed += 1
        pd.DataFrame([{"Successful ": successful, "Failed ": failed}]).to_csv(
            f"{basepath}/{result_name}.csv", index=False
        )


if __name__ == "__main__":
    main()
