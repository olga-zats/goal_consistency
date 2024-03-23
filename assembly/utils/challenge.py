import os
import csv
import sys
import numpy as np
import pandas as pd

label_id = ['verb', 'object', 'action']

def get_tail_top5(truth_dir, videos, labels, scores, topk=5):
    labels = np.swapaxes(np.array(labels), 0, 1)

    with open(os.path.join(truth_dir, 'tail_validation_segments.txt'), 'r') as file:
        tail_segs = file.readlines()
    tail_segs = set([int(x.strip(' ').strip('\n')) for x in tail_segs])
    
    split = 'tail'
    _labels = []
    _scores = [[], [], []]

    for i, video in enumerate(videos):
        if split == 'tail':
            if not i in tail_segs:
                continue
        if split == 'head':
            if i in tail_segs:
                continue

        _labels.append(labels[i])
        for j in range(len(scores)):
            _scores[j].append(scores[j][i])

    _labels = np.array(_labels)
    for j in range(len(scores)):
        _scores[j] = np.array(_scores[j])

    res = list(range(len(_scores)))
    for i in range(len(_scores)):
        prec = topk_recall(_scores[i], _labels[:, i], k=topk)
        res[i] = '%.2f' % (prec*100)
    return res


def get_head_top5(truth_dir, videos, labels, scores, topk=5):
    labels = np.swapaxes(np.array(labels), 0, 1)

    with open(os.path.join(truth_dir, 'tail_validation_segments.txt'), 'r') as file:
        tail_segs = file.readlines()
    tail_segs = set([int(x.strip(' ').strip('\n')) for x in tail_segs])
    
    split = 'head'
    _labels = []
    _scores = [[], [], []]

    for i, video in enumerate(videos):
        if split == 'tail':
            if not i in tail_segs:
                continue
        if split == 'head':
            if i in tail_segs:
                continue

        _labels.append(labels[i])
        for j in range(len(scores)):
            _scores[j].append(scores[j][i])

    _labels = np.array(_labels)
    for j in range(len(scores)):
        _scores[j] = np.array(_scores[j])

    res = list(range(len(_scores)))
    for i in range(len(_scores)):
        prec = topk_recall(_scores[i], _labels[:, i], k=topk)
        res[i] = '%.2f' % (prec*100)
    return res


def unseen_div_top5(truth_dir, videos, labels, scores, topk=5):
    labels = np.swapaxes(np.array(labels), 0, 1)

    with open(os.path.join(truth_dir, 'validation_split_seq.txt'), 'r') as file:
        classes = file.readlines()
    classes = [x.strip(' ').strip('\n').split('\t')[:2] for x in classes]

    seen_videos = []
    unseen_videos = []
    for x in classes:
        if x[1] == 'notshared':
            unseen_videos.append(x[0])
        else:
            seen_videos.append(x[0])

    seen_videos = set(seen_videos)
    unseen_videos = set(unseen_videos)

    split = 'unseen'
    split_videos = unseen_videos

    _labels = []
    _scores = [[], [], []]

    for i, video in enumerate(videos):
        if video in split_videos:
            _labels.append(labels[i])
            for j in range(len(scores)):
                _scores[j].append(scores[j][i])

    _labels = np.array(_labels)
    for j in range(len(scores)):
        _scores[j] = np.array(_scores[j])
    
    res = list(range(len(_scores)))
    for i in range(len(_scores)):
        prec = topk_recall(_scores[i], _labels[:, i], k=topk)
        res[i] = '%.2f' % (prec*100)
    return res


def head_tail_div(videos, labels, scores, topk):
    labels = np.swapaxes(np.array(labels), 0, 1)

    with open('tail_validation_segments.txt', 'r') as file:
        tail_segs = file.readlines()
    tail_segs = set([int(x.strip(' ').strip('\n')) for x in tail_segs])
    
    for (split, videos) in [('head', videos), ('tail', videos)]:
        _labels = []
        _scores = [[], [], []]
    
        for i, video in enumerate(videos):
            if split == 'tail':
                if not i in tail_segs:
                    continue
            if split == 'head':
                if i in tail_segs:
                    continue

            _labels.append(labels[i])
            for j in range(len(scores)):
                _scores[j].append(scores[j][i])

        _labels = np.array(_labels)
        for j in range(len(scores)):
            _scores[j] = np.array(_scores[j])

        for i in range(len(_scores)):
            prec = topk_recall(_scores[i], _labels[:, i], k=topk)    
    

def seen_or_unseen_div(videos, labels, scores, topk):
    labels = np.swapaxes(np.array(labels), 0, 1)

    with open('validation_split_seq.txt') as file:
        classes = file.readlines()
    classes = [x.strip(' ').strip('\n').split('\t')[:2] for x in classes]

    seen_videos = []
    unseen_videos = []
    for x in classes:
        if x[1] == 'notshared':
            unseen_videos.append(x[0])
        else:
            seen_videos.append(x[0])

    seen_videos = set(seen_videos)
    unseen_videos = set(unseen_videos)

    
    for (split, split_videos) in [('seen', seen_videos), ('unseen', unseen_videos)]:
        _labels = []
        _scores = [[], [], []]
    
        for i, video in enumerate(videos):
            if video in split_videos:
                _labels.append(labels[i])
                for j in range(len(scores)):
                    _scores[j].append(scores[j][i])

        _labels = np.array(_labels)
        for j in range(len(scores)):
            _scores[j] = np.array(_scores[j])

        for i in range(len(_scores)):
            prec = topk_recall(_scores[i], _labels[:, i], k=topk)



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
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0

    for c in classes:
        recalls += topk_accuracy(scores, labels, ks=(k,), selected_class=c)[0]
    return recalls / len(classes)
