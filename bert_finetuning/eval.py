import torch
from utils import flat_accuracy


def eval_model(model, test_dataloader, device):
    model = model.to(device)
    ## tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    ## evaluate data for one epoch
    for batch in test_dataloader:
        ## add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        ## unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        ## avoiding model's computation and storage of gradients -> saving memory and speeding up validation
        with torch.no_grad():
            # forward pass, calculate logit predictions
            logits = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        ## move logits and labels to CPU
        logits = logits[0].detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Test Accuracy: {}".format(eval_accuracy / nb_eval_steps))


def eval_model_classification_report(model, test_dataloader, device):
    import numpy as np
    from sklearn.metrics import classification_report

    model = model.to(device)
    ## tracking variables
    preds = []
    ground_truth = []

    ## evaluate data for one epoch
    for batch in test_dataloader:
        ## add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        ## unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        ## avoiding model's computation and storage of gradients -> saving memory and speeding up validation
        with torch.no_grad():
            # forward pass, calculate logit predictions
            logits = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        ## move logits and labels to CPU
        logits = logits[0].detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        preds.extend(np.argmax(logits, axis=1).flatten())
        ground_truth.extend(label_ids.flatten())

    print(classification_report(ground_truth, preds))
