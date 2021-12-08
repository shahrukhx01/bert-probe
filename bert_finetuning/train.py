from utils import flat_accuracy
from tqdm import tqdm, trange
import torch


def train_model(
    model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs, device
):
    t = []

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):

        ## set our model to training mode
        model.train()

        ## tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # train the model for one epoch
        for step, batch in enumerate(train_dataloader):
            ## move batch to GPU
            batch = tuple(t.to(device) for t in batch)
            ## unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            ## reset the gradients
            optimizer.zero_grad()
            ## forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            loss, logits = outputs[:2]
            train_loss_set.append(loss.item())
            ## backward pass
            loss.backward()
            ## update parameters and take a step using the computed gradient
            optimizer.step()

            ## update the learning rate.
            scheduler.step()

            ## update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # avoiding model's computation and storage of gradients -> saving memory and speeding up validation
            with torch.no_grad():
                # forward pass, calculate logit predictions
                logits = model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )

            # Move logits and labels to CPU
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        model.save_pretrained("./model")
        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    return model
