from .character_embeddings import CharacterEmbeddings
from .model import GermanHateSpeechModel
import pandas as pd
import os
from tqdm import tqdm
import torch
import numpy as np

tqdm.pandas(desc="my bar!")


def remove_tags(sequence):
    colors = ["red", "green"]
    sequence = sequence.replace("</font>", "")
    for color in colors:
        sequence = sequence.replace(f"<font color = {color}>", "")
    return sequence


def main():
    char_embedder = CharacterEmbeddings()
    basepath = "./datasets"
    models = {
        "hasoc": GermanHateSpeechModel(
            "/Users/shahrukh/Desktop/victim_models/hasoc_model"
        ),
        "germeval": GermanHateSpeechModel(
            "/Users/shahrukh/Desktop/victim_models/germeval_model"
        ),
    }
    embeddings = {
        "hasoc": (
            pd.read_csv("./embeddings/hasoc_vocab_embeddings.csv"),
            np.load("./embeddings/hasoc_embeddings.npy"),
        ),
        "germeval": (
            pd.read_csv("./embeddings/germeval_vocab_embeddings.csv"),
            np.load("./embeddings/germeval_embeddings.npy"),
        ),
    }
    for filename in os.listdir(f"{basepath}/perturbed_sets"):
        if "defense" in filename:
            continue
        successful, failed = 0, 0
        result_name = f"{filename}_defense".replace(".csv", "")
        dataset = pd.read_csv(f"{basepath}/perturbed_sets/{filename}")
        dataset = dataset[dataset["result_type"] == "Successful"]
        model = models["hasoc"]
        embedding = embeddings["hasoc"]
        if "germeval" in result_name:
            model = models["germeval"]
            embedding = embeddings["germeval"]
        vocab_words = embedding[0].word.values.tolist()
        vocab_words = [str(word) for word in vocab_words]
        vocab_words.sort()
        embedding_matrix = torch.Tensor(embedding[1])
        defense_results = []
        for idx, row in tqdm(dataset.iterrows()):
            perturbed_sequence = remove_tags(row["perturbed_text"])
            original_sequence = remove_tags(row["original_text"])

            perturbed_tokens = perturbed_sequence.split(" ")
            original_tokens = original_sequence.split(" ")

            perturbed_embeddings = char_embedder.get_sequence_embeddings(
                perturbed_sequence
            )

            perturbed_update = perturbed_tokens
            for index, _ in enumerate(perturbed_tokens):
                sim = char_embedder.cos_sim(
                    perturbed_embeddings[index, :], embedding_matrix
                )
                if sim.max().item() > 0.7 and sim.max().item() < 1.0:
                    best_candidate_idx = sim.argmax()
                    perturbed_update[index] = vocab_words[best_candidate_idx]

            defended_sequence = " ".join(perturbed_update)
            gt = int(row["ground_truth_output"])
            label = f"LABEL_{gt}"
            prediction = model([defended_sequence])[0]["label"]
            if prediction == label:
                successful += 1
                defense_results.append(
                    {
                        "original_text": original_sequence,
                        "perturbed_text": perturbed_sequence,
                        "defended_text": defended_sequence,
                        "defended_outcome": "SUCCESSFUL",
                        "original_label": label,
                        "defended_prediction": prediction,
                    }
                )
            else:
                failed += 1
                defense_results.append(
                    {
                        "original_text": original_sequence,
                        "perturbed_text": perturbed_sequence,
                        "defended_text": defended_sequence,
                        "defended_outcome": "FAILED",
                        "original_label": label,
                        "defended_prediction": prediction,
                    }
                )
        pd.DataFrame([{"Successful ": successful, "Failed ": failed}]).to_csv(
            f"{basepath}/summary-{result_name}.csv", index=False
        )
        pd.DataFrame(defense_results).to_csv(
            f"{basepath}/{result_name}.csv", index=False
        )


if __name__ == "__main__":
    main()
