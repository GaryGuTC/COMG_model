from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np

import logging

from pycocoevalcap.bleu.bleu import Bleu

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

Bleu_scorer = None


def init_scorer():
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_self_critical_reward(greedy_res, data_gts, gen_result):
    batch_size = len(data_gts)
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts)  # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    data_gts = data_gts.cpu().numpy()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i])]
    res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i + gen_result_size: gts[i] for i in range(batch_size)})
    _, bleu_scores = Bleu_scorer.compute_score(gts_, res__, verbose = 0)
    bleu_scores = np.array(bleu_scores[3])
    # logger.info('Bleu scores: {:.4f}.'.format(_[3]))
    scores = bleu_scores

    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_result_size)

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards
