class ExecuteAttack:
    def execute(self, dataset, attacks):
        for text, label in dataset:
            for attack in attacks:
                result = attack.attack(text, label)
                print(result)
        ##attack = BlackboxWordLevelAttack.build(model_wrapper)
