from tqdm import tqdm
from textattack.loggers import CSVLogger


class ExecuteAttack:
    """
    Executor class has the executable for carrying out an attack on a
    victim model
    """

    @staticmethod
    def execute(dataset, attacks):
        """
        Execute the given attacks by perturbing the dataset on
        victim model

        Args:
            dataset (List): Dataset to be perturbed.
            attacks (List[Attack]): List of attacks to be carried out.
        """

        print(f"length of attack dataset: {len(dataset)}")
        for attack_name, attack in tqdm(attacks):
            logger = CSVLogger(
                color_method="html",
                filename=f"./attack_logs/results-{attack_name}.csv",
            )
            for text, label in dataset:
                result = attack.attack(text, label)
                logger.log_attack_result(result)
                print(result.__str__(color_method="ansi"))
            # write the results and attack summary to csv
            logger.flush()
            summary = logger.df["result_type"].value_counts()
            summary.to_csv(f"./attack_logs/summary-{attack_name}.csv")
