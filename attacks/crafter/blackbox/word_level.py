"""
Blackbox Word Level Attack using MLMSwap
============================================

"""
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from utils.universal_sentence_encoder_de import UniversalSentenceEncoderDE
from textattack.goal_functions import UntargetedClassification
from utils.greedy_word_swap_wir import GreedyWordSwapWIR
from textattack.transformations import WordSwapMaskedLM

from textattack.attack_recipes.attack_recipe import AttackRecipe


class BlackboxWordLevelAttack(AttackRecipe):
    """
    The blackbox word level attack is based on the following study:
    BAE: BERT-based Adversarial Examples for Text Classification.

    https://arxiv.org/pdf/2004.01970

    This is "attack mode" 1 from the paper, BAE-R, word replacement.

    We present 4 attack modes for BAE based on the
        R and I operations, where for each token t in S:
        • BAE-R: Replace token t (See Algorithm 1)
        • BAE-I: Insert a token to the left or right of t
        • BAE-R/I: Either replace token t or insert a
        token to the left or right of t
        • BAE-R+I: First replace token t, then insert a
        token to the left or right of t

    We have made further changes to this attack to adapt this attack for German language
    """

    @staticmethod
    def build(model_wrapper):
        # "In this paper, we present a simple yet novel technique: BAE (BERT-based
        # Adversarial Examples), which uses a language model (LM) for token
        # replacement to best fit the overall context. We perturb an input sentence
        # by either replacing a token or inserting a new token in the sentence, by
        # means of masking a part of the input and using a LM to fill in the mask."
        #
        # We only consider the top K=50 synonyms from the MLM predictions.
        #
        # "When choosing the top-K candidates from the BERT masked LM, we filter out
        # the sub-words and only retain the whole words (by checking if they are
        # present in the GloVE vocabulary)"
        #
        transformation = WordSwapMaskedLM(
            masked_language_model="deepset/gbert-base",
            method="bae",
            max_candidates=50,
            min_confidence=0.0,
        )
        #
        # Don't modify the same word twice or stopwords.
        #
        constraints = [RepeatModification(), StopwordModification()]

        # For the R operations we add an additional check for
        # grammatical correctness of the generated adversarial example by filtering
        # out predicted tokens that do not form the same part of speech (POS) as the
        # original token t_i in the sentence.
        constraints.append(
            PartOfSpeech(
                tagger_type="stanza", allow_verb_noun_swap=True, language_stanza="de"
            )
        )

        # "To ensure semantic similarity on introducing perturbations in the input
        # text, we filter the set of top-K masked tokens (K is a pre-defined
        # constant) predicted by BERT-MLM using a Universal Sentence Encoder (USE)
        # (Cer et al., 2018)-based sentence similarity scorer."
        #
        # "[We] set a threshold of 0.8 for the cosine similarity between USE-based
        # embeddings of the adversarial and input text."
        #
        # "For a fair comparison of the benefits of using a BERT-MLM in our paper,
        # we retained the majority of TextFooler's specifications. Thus we:
        # 1. Use the USE for comparison within a window of size 15 around the word
        # being replaced/inserted.
        # 2. Set the similarity score threshold to 0.1 for inputs shorter than the
        # window size (this translates roughly to almost always accepting the new text).
        # 3. Perform the USE similarity thresholding of 0.8 with respect to the text
        # just before the replacement/insertion and not the original text (For
        # example: at the 3rd R/I operation, we compute the USE score on a window
        # of size 15 of the text obtained after the first 2 R/I operations and not
        # the original text).
        # ...
        # To address point (3) from above, compare the USE with the original text
        # at each iteration instead of the current one (While doing this change
        # for the R-operation is trivial, doing it for the I-operation with the
        # window based USE comparison might be more involved)."
        #
        # Finally, since the BAE code is based on the TextFooler code, we need to
        # adjust the threshold to account for the missing / pi in the cosine
        # similarity comparison. So the final threshold is 1 - (1 - 0.8) / pi
        # = 1 - (0.2 / pi) = 0.936338023.
        use_constraint = UniversalSentenceEncoderDE(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=False,
        )
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification.
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # "We estimate the token importance Ii of each token
        # t_i ∈ S = [t1, . . . , tn], by deleting ti from S and computing the
        # decrease in probability of predicting the correct label y, similar
        # to (Jin et al., 2019).
        #
        # • "If there are multiple tokens can cause C to misclassify S when they
        # replace the mask, we choose the token which makes Sadv most similar to
        # the original S based on the USE score."
        # • "If no token causes misclassification, we choose the perturbation that
        # decreases the prediction probability P(C(Sadv)=y) the most."
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return BlackboxWordLevelAttack(
            goal_function, constraints, transformation, search_method
        )
