import pandas as pd
import re
import os
import random
import gdown
from textattack.attack_recipes import (
    DeepWordBugGao2018,
    Pruthi2019,
    TextBuggerLi2018,
    PSOZang2020,
    PWWSRen2019,
    TextFoolerJin2019,
    IGAWang2019,
    BAEGarg2019,
    CheckList2020,
)
from textattack.loggers import CSVLogger
from textattack.datasets import Dataset
from flair.models import TextClassifier
import time
from pandas import DataFrame
from textattack.models.wrappers import ModelWrapper


recipes = [
    "DeepWordBugGao2018",
    "Pruthi2019",
    "TextBuggerLi2018",
    "PSOZang2020",
    "PWWSRen2019",
    "TextFoolerJin2019",
    "IGAWang2019",
    "BAEGarg2019",
    "CheckList2020",
]
models = ["flair", "berttweet", "roberta"]


for model_name in models:
    for recipe in recipes:
        print(recipe)
        print(model_name)

        recept_name = recipe
        timestr = time.strftime("%Y%m%d-%H%M%S")
        result_dir = (
            "/content/drive/MyDrive/text-attack-results-false/results-"
            + recept_name
            + "-"
            + timestr
        )
        os.mkdir(result_dir)

        if model_name == "flair":
            url = "https://drive.google.com/uc?id=1xh7O-Wl4Rwr-eau1OrBQRCRXvZ6Z31J4"
        elif model_name == "berttweet":
            url = "https://drive.google.com/uc?id=1m1Zqif7HH4alCoEPBKwXk-9gXXuP0mDh"
        else:
            url = "https://drive.google.com/uc?id=1-ouVsPHfIM7pscfguU_upcjZao55A02S"
        output = "best-model.pt"
        gdown.download(url, output, quiet=True)

        model = TextClassifier.load("./best-model.pt")
        model_wrapper = CustomModelWrapper(model)

        if recipe == "DeepWordBugGao2018":
            attack = DeepWordBugGao2018.build(model_wrapper)
        elif recipe == "Pruthi2019":
            attack = Pruthi2019.build(model_wrapper)
        elif recipe == "TextBuggerLi2018":
            attack = TextBuggerLi2018.build(model_wrapper)
        elif recipe == "PSOZang2020":
            attack = PSOZang2020.build(model_wrapper)
        elif recipe == "PWWSRen2019":
            attack = PWWSRen2019.build(model_wrapper)
        elif recipe == "TextFoolerJin2019":
            attack = TextFoolerJin2019.build(model_wrapper)
        elif recipe == "IGAWang2019":
            attack = IGAWang2019.build(model_wrapper)
        elif recipe == "BAEGarg2019":
            attack = BAEGarg2019.build(model_wrapper)
        elif recipe == "CheckList2020":
            attack = CheckList2020.build(model_wrapper)

        dataset = Dataset(custom_dataset)

        # Add timestamp to result file
        logger = CSVLogger(
            color_method="html", filename=result_dir + "/results-" + model_name + ".csv"
        )

        for example, label in custom_dataset:
            result = attack.attack(example, label)
            logger.log_attack_result(result)
            print(result.__str__(color_method="ansi"))

        # Write the result csv to google drive
        logger.flush()
        summary = logger.df["result_type"].value_counts()
        summary.to_csv(result_dir + "/summary-" + model_name + ".csv")
