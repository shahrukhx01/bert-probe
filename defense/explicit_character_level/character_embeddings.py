import torch
from transformers import AutoTokenizer, AutoModel
from torch import Tensor


class CharacterEmbeddings:
    def __init__(
        self, model_name_or_path="shahrukhx01/paraphrase-mpnet-base-v2-fuzzy-matcher"
    ):
        # Load model from HuggingFace Hub
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)

    def cos_sim(self, a: Tensor, b: Tensor):
        """
        borrowed from sentence transformers repo
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_word_embedding(self, word):
        spaced_word = " ".join(list(word))
        # Tokenize sentences
        encoded_input = self.tokenizer(
            [spaced_word], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        return self.mean_pooling(model_output, encoded_input["attention_mask"])

    def get_sequence_embeddings(self, sequence):
        embeddings_list = []
        for token in sequence.split(" "):
            embeddings_list.append(self.get_word_embedding(token))
        embeddings = torch.Tensor(len(embeddings_list), 768)
        torch.cat(embeddings_list, out=embeddings)
        return embeddings
