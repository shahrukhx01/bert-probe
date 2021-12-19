from character_embeddings import CharacterEmbeddings
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
    for filename in os.listdir(basepath)[:1]:  ## TODO: Revert subscript [:1]
        result_name = f"{filename}_defense".replace(".csv", "")
        dataset = pd.read_csv(f"{basepath}/{filename}")
        dataset = dataset[
            dataset["result_type"] == "Successful"
        ].head()  ## TODO: Revert head()
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
            for index, _ in enumerate(perturbed_tokens):
                best_candidate_idx = char_embedder.cos_sim(
                    perturbed_embeddings[index, :], ground_truth_embeddings
                ).argmax()
                perturbed_update = original_tokens[best_candidate_idx]
                print(
                    f"original word: {original_tokens[index]} perturbed word: {perturbed_tokens[index]} updated: {perturbed_update}"
                )


if __name__ == "__main__":
    main()
