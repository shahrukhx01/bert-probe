#This code is taken from:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L102
from transformers import AdamW, get_linear_schedule_with_warmup

class BertOptimConfig:
    def __init__(self, model, train_dataloader, epochs=2):
        # Don't apply weight decay to any parameters whose names include these tokens.
        # (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        # Separate the `weight` parameters from the `bias` parameters. 
        # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01. 
        # - For the `bias` parameters, the 'weight_decay_rate' is 0.0. 
        optimizer_grouped_parameters = [
            # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.1},
            
            # Filter for parameters which *do* include those.
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
        # Note - `optimizer_grouped_parameters` only includes the parameter values, not 
        # the names.

        # Number of training epochs (authors recommend between 2 and 4)
        self.epochs = epochs

        self.optimizer = AdamW(optimizer_grouped_parameters,
                        lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )
        # Total number of training steps is number of batches * number of epochs.
        # `train_dataloader` contains batched data so `len(train_dataloader)` gives 
        # us the number of batches.
        total_steps = len(train_dataloader) * self.epochs

        ## create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)