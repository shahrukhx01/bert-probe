import logging

import pandas as pd

from bert_finetuning.data import GermanData
from defense.explicit_character_level.defense import remove_tags

logger = logging.getLogger(__name__)


class GermanAdversarialData(GermanData):

    def __init__(
            self,
            data_path,
            model_name,
            separator=",",
            max_sequence_length=512,
            do_cleansing=True,
    ):
        super().__init__(
            data_path, model_name, separator, max_sequence_length, do_cleansing
        )
        self.adv_train_df = pd.read_csv(data_path["adv"]["train"], sep=separator)

        # merge with normal ones
        self.merge_adv_dfs()

    def merge_adv_dfs(self) -> None:
        # calc abstain label
        ABSTAIN = len(pd.Categorical(self.train_df.label).categories)

        # select only successful adv examples
        df = self.adv_train_df
        text = df.perturbed_text[df.result_type == "Successful"]
        label = [ABSTAIN] * len(text)

        # put into other
        df = pd.DataFrame({"text": text, "label": label})
        self.train_df.append(df, ignore_index=True)

        # NOTE: No need to shuffle since a RandomSampler is used to sample
        # from the training data.

        # clear memory
        del self.adv_train_df

    def clean_text(self, text: str) -> str:
        # remove html tags inserted by attack
        text = remove_tags(text)
        text = super().clean_text(text)
        return text
