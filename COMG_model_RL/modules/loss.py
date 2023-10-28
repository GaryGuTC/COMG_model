import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class LossWrapper(nn.Module):
    def __init__(self):
        super(LossWrapper, self).__init__()
        self.criterion = LanguageModelCriterion()
        self.criterion_mlc = nn.BCELoss()

    def forward(self, output, output_mlc, reports_ids, reports_masks, label):
        loss = self.criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        loss_mlc = self.criterion_mlc(output_mlc, label)
        return loss + loss_mlc


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq > 0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
