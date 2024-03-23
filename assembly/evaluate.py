import os
import sys
import json
import numpy as np
import pandas as pd
from utils.utils import get_marginal_indexes, marginalize, softmax
from utils.challenge import head_tail_div, seen_or_unseen_div, topk_recall, get_tail_top5, unseen_div_top5, get_head_top5

action_classes = 1064
verb_classes = 17
object_classes = 90

if __name__ == '__main__':
    input_dir = sys.argv[1]
    preds_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    with open(os.path.join(input_dir, 'sequences.txt'), 'r') as f:
        sequences = f.readlines()
    sequences = [x.strip() for x in sequences]
    actions = pd.read_csv(os.path.join(f'{input_dir}/CSVs', 'actions.csv'), index_col='action_id')

    #GT--------------------------
    with open(os.path.join(input_dir, 'GT.json'), 'rb') as f:
        gt = json.load(f)

    gt = gt['results']

    action_labels = np.zeros(len(gt)).astype(int)
    verb_labels = np.zeros(len(gt)).astype(int)
    object_labels = np.zeros(len(gt)).astype(int)

    for segment_id in gt:
        action_labels[int(segment_id)] = int(gt[segment_id]['action'])
        verb_labels[int(segment_id)] = int(gt[segment_id]['verb'])
        object_labels[int(segment_id)] = int(gt[segment_id]['object'])
    
    with open(os.path.join(preds_dir, 'preds.json'), 'rb') as f:
        preds = json.load(f)

    print('[preds.json] loaded')
    preds = preds['results']    

    action_scores = np.zeros((len(preds), action_classes))
    verb_scores = np.zeros((len(preds), verb_classes))
    object_scores = np.zeros((len(preds), object_classes))

    np_segs = []
    for i in range(len(action_labels)):
        if str(i) not in preds:
            np_segs.append(str(i))

    if 'action' in preds['0']:
        for segment_id in preds:
            action_scores[int(segment_id)] = np.array(preds[segment_id]['action']).astype(float)
    else:
        print('[ERROR] actions scores not present.\nEvaluation requires at least action scores to be present.')
        exit()
    if 'verb' in preds['0']:
        for segment_id in preds:
            verb_scores[int(segment_id)] = np.array(preds[segment_id]['verb']).astype(float)
    else:
        vi = get_marginal_indexes(actions, 'verb_id')
        action_prob = softmax(action_scores.reshape(-1, action_scores.shape[-1]))
        verb_scores = marginalize(action_prob, vi)

    if 'object' in preds['0']:
        for segment_id in preds:
            object_scores[int(segment_id)] = np.array(preds[segment_id]['object']).astype(float)
    else:
        ni = get_marginal_indexes(actions, 'noun_id')
        action_prob = softmax(action_scores.reshape(-1, action_scores.shape[-1]))
        object_scores = marginalize(action_prob, ni)

    #----------------------------
    overall_top1_verb = '%.2f' % (topk_recall(verb_scores, verb_labels, k=1)*100)
    overall_top1_object = '%.2f' % (topk_recall(object_scores, object_labels, k=1)*100)
    overall_top1_action = '%.2f' % (topk_recall(action_scores, action_labels, k=1)*100)
    overall_top5_verb = '%.2f' % (topk_recall(verb_scores, verb_labels, k=5)*100)
    overall_top5_object = '%.2f' % (topk_recall(object_scores, object_labels, k=5)*100)
    overall_top5_action = '%.2f' % (topk_recall(action_scores, action_labels, k=5)*100)
    
    print("OVERALL_TOP1_VERB:%s\n" % overall_top1_verb)
    print("OVERALL_TOP1_OBJECT:%s\n" % overall_top1_object)
    print("OVERALL_TOP1_ACTION:%s\n" % overall_top1_action)
    print("OVERALL_TOP5_VERB:%s\n" % overall_top5_verb)
    print("OVERALL_TOP5_OBJECT:%s\n" % overall_top5_object)
    print("OVERALL_TOP5_ACTION:%s\n" % overall_top5_action)
    tail_res = get_tail_top5(input_dir, sequences, [verb_labels, object_labels, action_labels], 
                        [verb_scores, object_scores, action_scores])
    print("TAIL_TOP5_VERB:%s\n" % tail_res[0])
    print("TAIL_TOP5_OBJECT:%s\n" % tail_res[1])
    print("TAIL_TOP5_ACTION:%s\n" % tail_res[2])
    head_res = get_head_top5(input_dir, sequences, [verb_labels, object_labels, action_labels], 
                        [verb_scores, object_scores, action_scores])
    print("HEAD_TOP5_VERB:%s\n" % head_res[0])
    print("HEAD_TOP5_OBJECT:%s\n" % head_res[1])
    print("HEAD_TOP5_ACTION:%s\n" % head_res[2])
    unseen_res = unseen_div_top5(input_dir, sequences, [verb_labels, object_labels, action_labels], 
                        [verb_scores, object_scores, action_scores])
    print("UNSEEN_TOP5_VERB:%s\n" % unseen_res[0])
    print("UNSEEN_TOP5_OBJECT:%s\n" % unseen_res[1])
    print("UNSEEN_TOP5_ACTION:%s\n" % unseen_res[2])
