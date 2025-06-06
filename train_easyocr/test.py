import sys
from pathlib import Path
# Add project root directory (EasyOCR_finetune) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from time import time
import torch
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

from train_easyocr.utils import (
    Averager
)


def validation(model, criterion, val_loader, converter, config, device):
    """ Validation or evaluation """
    n_correct = 0
    norm_ed = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for image_tensors, labels in val_loader:
        batch_size = image_tensors.size(0)
        length_of_data += batch_size

        # Move input images to GPU (or whatever `device` is)
        image = image_tensors.to(device)

        # Prepare placeholders on the same device
        length_for_pred = torch.IntTensor([config.batch_max_length] * batch_size).to(device)
        text_for_pred   = torch.LongTensor(batch_size, config.batch_max_length + 1).fill_(0).to(device)

        # Encode labels → get (text_for_loss, length_for_loss) on CPU, then move to device
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=config.batch_max_length)
        text_for_loss   = text_for_loss.to(device)
        length_for_loss = length_for_loss.to(device)

        start_time = time()
        if 'CTC' in config.Prediction:
            # Forward pass
            preds = model(image, text_for_pred)
            forward_time = time() - start_time

            # preds has shape (batch_size, seq_len, n_classes)
            # We want log_probs in shape (seq_len, batch_size, n_classes)
            log_probs = preds.log_softmax(2).permute(1, 0, 2)  # on `device`

            # Build preds_size *directly on the same device* as log_probs / length_for_loss
            seq_len = preds.size(1)
            preds_size = torch.IntTensor([seq_len] * batch_size).to(device)

            # Now all four arguments (log_probs, text_for_loss, preds_size, length_for_loss)
            # live on `device`, so CTC-loss won’t complain.
            loss = criterion(
                log_probs=log_probs,
                targets=text_for_loss,
                input_lengths=preds_size,
                target_lengths=length_for_loss
            )

            # Greedy or beamsearch decoding
            if config.decode == 'greedy':
                # preds: (batch_size, seq_len, n_classes)
                # argmax over dim=2 → (batch_size, seq_len)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)  # flatten for converter
                preds_str = converter.decode_greedy(preds_index.data, preds_size.data)
            elif config.decode == 'beamsearch':
                preds_str = converter.decode_beamsearch(preds, beamWidth=2)

        else:
            # Attention‐based prediction branch
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time() - start_time

            # preds: (batch_size, seq_len, n_classes)
            # We trim to match target length (excluding [GO] and [s] tokens)
            preds = preds[:, : text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # Remove [GO] token from ground truth
            loss = criterion(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1)
            )

            # Greedy decoding for Attn
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(loss)

        # Compute confidence scores & normalized edit distance
        preds_prob = F.softmax(preds, dim=2)  # (batch_size, seq_len, n_classes)
        preds_max_prob, _ = preds_prob.max(dim=2)  # (batch_size, seq_len)
        confidence_score_list = []

        # Note: `labels` here is a list of decoded ground‐truth strings
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in config.Prediction:
                # strip off anything after the “[s]” token in prediction
                gt = gt[: gt.find('[s]')] if '[s]' in gt else gt
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS] if pred_EOS != -1 else pred
                pred_max_prob = pred_max_prob[:pred_EOS] if pred_EOS != -1 else pred_max_prob

            # Exact‐match accuracy
            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ed += 0
            elif len(gt) > len(pred):
                norm_ed += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ed += 1 - edit_distance(pred, gt) / len(pred)

            # Confidence = product of max‐prob over the predicted character sequence
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # e.g., empty pred after pruning
            confidence_score_list.append(confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ed  = norm_ed / float(length_of_data)

    return (
        valid_loss_avg.val(),
        accuracy,
        norm_ed,
        preds_str,
        confidence_score_list,
        labels,
        infer_time,
        length_of_data
    )
