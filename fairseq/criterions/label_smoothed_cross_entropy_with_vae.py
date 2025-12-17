# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
from .label_smoothed_cross_entropy import label_smoothed_nll_loss, LabelSmoothedCrossEntropyCriterion

def vae_kld_loss(mean, log_std):
    if mean == None or log_std == None:
        return torch.tensor(0)
    kld = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(2 * log_std), dim=-1)
    kld = kld.mean()
    return kld


@register_criterion('label_smoothed_cross_entropy_with_vae')
class LabelSmoothedCrossEntropyCriterionWithVAE(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.beta = args.vae_beta

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--vae-beta', default=1., type=float, metavar='D',
                            help='weight of kld loss')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, kld_loss, metrics = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'kld_loss': utils.item(kld_loss.data) if reduce else kld_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        logging_output.update(metrics)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # -------- LM loss --------
        lprobs = model.get_normalized_probs(net_output[0], log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output[0]).view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target.unsqueeze(-1), self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        # -------- VAE KLD --------
        kld_loss = vae_kld_loss(net_output[1], net_output[2])
        loss = loss + self.beta * kld_loss

        # -------- Metrics --------
        with torch.no_grad():
            preds = lprobs.argmax(dim=-1)

            mask = target.ne(self.padding_idx)
            preds = preds[mask]
            target = target[mask]

            num_classes = lprobs.size(-1)
            eps = 1e-9

            tp = torch.zeros(num_classes, device=preds.device)
            fp = torch.zeros(num_classes, device=preds.device)
            fn = torch.zeros(num_classes, device=preds.device)
            tn = torch.zeros(num_classes, device=preds.device)

            for c in range(num_classes):
                tp[c] = ((preds == c) & (target == c)).sum()
                fp[c] = ((preds == c) & (target != c)).sum()
                fn[c] = ((preds != c) & (target == c)).sum()
                tn[c] = ((preds != c) & (target != c)).sum()

        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        return loss, nll_loss, kld_loss, metrics

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        eps = 1e-9
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        tp = sum(log['tp'] for log in logging_outputs)
        fp = sum(log['fp'] for log in logging_outputs)
        fn = sum(log['fn'] for log in logging_outputs)
        tn = sum(log['tn'] for log in logging_outputs)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        support = tp + fn

        accuracy = tp.sum() / (support.sum() + eps)

        precision_macro = precision.mean()
        recall_macro = recall.mean()
        f1_macro = f1.mean()

        precision_weighted = (precision * support).sum() / (support.sum() + eps)
        recall_weighted = (recall * support).sum() / (support.sum() + eps)
        f1_weighted = (f1 * support).sum() / (support.sum() + eps)

        mcc_num = (tp * tn - fp * fn).sum()
        mcc_den = torch.sqrt(
            (tp + fp).sum() *
            (tp + fn).sum() *
            (tn + fp).sum() *
            (tn + fn).sum() + eps
        )
        mcc = mcc_num / mcc_den

        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'kld_loss': sum(log.get('kld_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            
            'accuracy': accuracy.item(),
            'precision_macro': precision_macro.item(),
            'recall_macro': recall_macro.item(),
            'f1_macro': f1_macro.item(),
            'precision_weighted': precision_weighted.item(),
            'recall_weighted': recall_weighted.item(),
            'f1_weighted': f1_weighted.item(),
            'mcc': mcc.item(),

            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
