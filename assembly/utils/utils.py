# -*- coding: utf-8 -*-
""" Set of utilities """
import numpy as np

from tqdm import tqdm
from torch import nn
import torch
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


class MeanTopKRecallMeter(object):
    def __init__(self, num_classes, k=5):
        self.num_classes = num_classes
        self.k = k
        self.reset()

    def reset(self):
        self.tps = np.zeros(self.num_classes)
        self.nums = np.zeros(self.num_classes)

    def add(self, scores, labels):
        tp = (np.argsort(scores, axis=1)[:, -self.k:] == labels.reshape(-1, 1)).max(1)
        for l in np.unique(labels):
            if l == -1:
                continue
            self.tps[l]+=tp[labels==l].sum()
            self.nums[l]+=(labels==l).sum()

    def value(self):
        recalls = (self.tps/self.nums)[self.nums>0]
        if len(recalls)>0:
            return recalls.mean()*100
        else:
            return -1.

        
class ValueMeter(object):
    def __init__(self):
        self.sum = 0
        self.total = 0

    def add(self, value, n):
        self.sum += value * n
        self.total += n

    def value(self):
        return self.sum / self.total


def topk_accuracy(scores, labels, ks, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        scores: numpy nd array, shape = (instance_count, label_count)
        labels: numpy nd array, shape = (instance_count,)
        ks: tuple of integers
    Returns:
        list of float: TOP-K accuracy for each k in ks
    """
    if selected_class is not None:
        idx = labels == selected_class
        scores = scores[idx]
        labels = labels[idx]
    rankings = scores.argsort()[:, ::-1]
    maxk = np.max(ks)  # trim to max k to avoid extra computation

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]



def topk_recall(scores, labels, k=5, classes=None):
    unique = np.unique(labels)
    #len(labels)
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0

    for c in classes:
        recalls += topk_accuracy(scores, labels, ks=(k,), selected_class=c)[0]
        #print(c, recalls)
    return recalls / len(classes)



def get_marginal_indexes(actions, mode):
    """For each verb/object retrieve the list of actions containing that verb/name
        Input:
            mode: "verb" or "object"
        Output:
            a list of numpy array of indexes. If verb/object 3 is contained in actions 2,8,19,
            then output[3] will be np.array([2,8,19])
    """
    vi = []
    for v in range(actions[mode].max() + 1):
        vals = actions[actions[mode] == v].index.values
        if len(vals) > 0:
            vi.append(vals)
        else:
            vi.append(np.array([0]))
    return vi


def marginalize(probs, indexes):
    mprobs = []
    for ilist in indexes:
        mprobs.append(probs[:, ilist].sum(1))
    return np.array(mprobs).T


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    xx = x
    x = x.reshape((-1, x.shape[-1]))
    e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
    res = e_x / e_x.sum(axis=1).reshape(-1, 1)
    return res.reshape(xx.shape)


def predictions_to_json(task, ids, 
                        action_scores, 
                        verb_scores = None, 
                        object_scores = None):
    """
    Save verb, object and action predictions to .json file for submitting it to Assembly101 leaderboard
    Providing the action scores are mandatory while the verb and object scores are optional.
    """

    predictions = {'task': task, 'results': {}}
    
    for i in tqdm(range(ids.shape[0]), 'Creating [preds.json]'):
        predictions['results'][str(ids[i])] = {}
        if isinstance(verb_scores, np.ndarray):
            predictions['results'][str(ids[i])]['verb'] = list(verb_scores[i])
        if isinstance(object_scores, np.ndarray):
            predictions['results'][str(ids[i])]['object'] = list(object_scores[i])
        predictions['results'][str(ids[i])]['action'] = list(action_scores[i])

    return predictions


def vis_predictions_to_json(task, ids,
                            action_scores, action_labels,
                            lat_action_labels, videos):
    predictions = {'task': task, 'results': {}}
    for i in tqdm(range(ids.shape[0]), 'Creating [preds.json]'):
        predictions['results'][str(ids[i])] = {}
        predictions['results'][str(ids[i])]['action'] = list(action_scores[i])
        predictions['results'][str(ids[i])]['action_label'] = int(action_labels[i])
        predictions['results'][str(ids[i])]['lat_action_label'] = int(lat_action_labels[i]) if len(lat_action_labels) > 0 else -1
        predictions['results'][str(ids[i])]['video'] = str(videos[i])

    return predictions