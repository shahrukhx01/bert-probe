from tqdm import tqdm


class ExecuteAttack:
    def execute(self, dataset, attacks):
        print(f"length of attack dataset: {len(dataset)}")
        for text, label in tqdm(dataset):
            for attack in attacks:
                result = attack.attack(text, label)
                print(result)
