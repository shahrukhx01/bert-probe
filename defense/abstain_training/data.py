import logging

import pandas as pd

from bert_finetuning.data import GermanData
from defense.explicit_character_level.defense import remove_tags

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


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

        logger.info("Read in normal dataset. Proceeding to adversarial examples...")
        logger.debug(f"self.train_df.shape={self.train_df.shape}")
        logger.debug(f"self.dev_df.shape={self.dev_df.shape}")
        logger.debug(f"self.test_df.shape={self.test_df.shape}")

        self.adv_train_df = pd.read_csv(data_path["adversarial"]["train"], sep=separator)
        self.adv_test_df = pd.read_csv(data_path["adversarial"]["test"], sep=separator)

        logger.info("Read in adversarial examples. Now merging normal examples with adversarial ones...")
        logger.debug(f"self.adv_train_df.shape={self.adv_train_df.shape}")
        logger.debug(f"self.adv_test_df.shape={self.adv_test_df.shape}")

        # merge with normal ones
        self.merge_adv_dfs()

        logger.info("Merging complete.")
        logger.debug(f"self.train_df.shape={self.train_df.shape}")
        logger.debug(f"self.dev_df.shape={self.dev_df.shape}")
        logger.debug(f"self.test_df.shape={self.test_df.shape}")

    def merge_adv_dfs(self) -> None:
        # calc abstain label
        ABSTAIN = len(pd.Categorical(self.train_df.label).categories)
        logger.debug("Using `%d` as abstain label.", ABSTAIN)

        # select only successful adv examples
        train_text = self.adv_train_df.perturbed_text[self.adv_train_df.result_type == "Successful"]
        test_text = self.adv_test_df.perturbed_text[self.adv_test_df.result_type == "Successful"]
        train_text = train_text.apply(remove_tags)  # remove inserted tags
        test_text = test_text.apply(remove_tags)  # remove inserted tags
        train_label = [ABSTAIN] * len(train_text)
        test_label = [ABSTAIN] * len(test_text)

        # put into other
        self.adv_train_df = pd.DataFrame({"text": train_text, "label": train_label})
        self.adv_test_df = pd.DataFrame({"text": test_text, "label": test_label})
        self.train_df.append(self.adv_train_df, ignore_index=True)
        self.test_df.append(self.adv_test_df, ignore_index=True)

        # NOTE: No need to shuffle train_df since a RandomSampler is used to
        # sample from the training data.

        # clear memory
        del self.adv_train_df
        del self.adv_test_df
