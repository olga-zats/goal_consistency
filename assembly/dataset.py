# -*- coding: utf-8 -*-

"""
Implements a dataset object which allows to read representations from LMDB datasets.
"""

import os
import lmdb
import numpy as np
import pandas as pd
from torch.utils import data
from tqdm import tqdm

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start(self):
        return int(self._data[1])

    @property
    def end(self):
        return int(self._data[2])

    @property
    def labels(self):
        return self._data[3]

    @property
    def c_labels(self):
        return self._data[4]
    
    @property
    def ts_labels(self):
        return self._data[5]

    @property
    def id(self):
        return self._data[6]


class SequenceDataset(data.Dataset):
    def __init__(self, path_to_lmdb,
                 path_to_csv,
                 time_step=1,
                 label_type='action',
                 c_label_type='c_action',
                 ts_label_type='toy_seq_name',
                 img_tmpl="frame_{:010d}.jpg",
                 challenge=False,
                 fps=30,
                 args=None):
        """
            Inputs:
                path_to_lmdb: path to the folder containing the LMDB dataset
                path_to_csv: path to training/validation csv
                time_step: in seconds
                label_type: which label to return (verb, object, or action)
                img_tmpl: image template to load the features
                challenge: allows to load csvs containing only time-stamp for the challenge
                fps: framerate
        """

        # read the csv file
        if challenge:
            self.annotations = pd.read_csv(path_to_csv, header=0,
                names = ['id', 'video', 'start', 'end', 'shared', 'rgb'])
        else:
            self.annotations = pd.read_csv(path_to_csv, header=0)
                #names = ['id', 'video', 'start', 'end', 'action', 'verb', 'object', 
                #'action_cls', 'verb_cls', 'object_cls', 'toyid', 'toyname',
                #'shared', 'rgb'])

        self.challenge = challenge
        self.path_to_lmdb = path_to_lmdb
        self.time_step = time_step
        self.fps = fps
        self.label_type = label_type
        self.c_label_type = c_label_type
        self.ts_label_type = ts_label_type
        self.img_tmpl = img_tmpl

        self.recent_sec1 = args.recent_sec1
        self.recent_sec2 = args.recent_sec2
        self.recent_sec3 = args.recent_sec3
        self.recent_sec4 = args.recent_sec4
        self.recent_dim = args.recent_dim

        self.spanning_sec = args.spanning_sec
        self.span_dim1 = args.span_dim1
        self.span_dim2 = args.span_dim2
        self.span_dim3 = args.span_dim3

        self.feat_dim = args.video_feat_dim
        self.modality = args.modality
        self.views = args.views

        self.debug_on = args.debug_on

        self.video_list = []
        self.discarded_ids = []
        self.discarded_labels = []
        self.discarded_videos = []

        # populate them
        self.__populate_lists()
        
        print(f'[#processed segments] = {len(self.video_list)}')
        print(f'[#discarded segments] = {len(self.discarded_ids)}')

        self.env = {view:lmdb.open(f'{self.path_to_lmdb}/{view}', readonly=True, lock=False) for view in self.views}

    def __populate_lists(self):
        count_debug = 0
        """ Samples a sequence for each action and populates the lists. """
        for _, a in tqdm(self.annotations.iterrows(), 'Populating Dataset', total=len(self.annotations)):
            count_debug += 1
            if self.debug_on:
                if count_debug > 200:
                    break

            video = a.video.replace('.mp4', '')
            view = video.split('/')[-1]
                    
            if not view in self.views:
                continue

            # sample frames before the beginning of the action
            recent_f, spanning_f = self.__get_snippet_features(a['start_frame'], video)

            if spanning_f is not None and recent_f is not None:
                _labels = -1
                if not self.challenge:
                    # fine
                    if isinstance(self.label_type, list):
                        _labels = a[self.label_type].values.astype(int)
                    else:
                        _labels = a[self.label_type]

                    # coarse
                    if isinstance(self.c_label_type, list):
                        _c_labels = a[self.c_label_type].values.astype(int)
                    else:
                        _c_labels = a[self.c_label_type]

                    # toy seq
                    if isinstance(self.ts_label_type, list):
                        _ts_labels = a[self.ts_label_type].values.astype(int)
                    else:
                        _ts_labels = a[self.ts_label_type]
                
                tmp = [video, a['start_frame']+1, a['end_frame']+1, _labels, _c_labels, _ts_labels, a.id]
                self.video_list.append(VideoRecord(tmp))
            else:
                self.discarded_ids.append(a.id)
                self.discarded_videos.append(a.video)
                if isinstance(self.label_type, list):
                    if self.challenge: # if sampling for the challenge, there are no labels, just add -1
                        self.discarded_labels.append(-1)
                    else:
                        # otherwise get the required labels
                        self.discarded_labels.append(a[self.label_type].values.astype(int))
                else: #single label version
                    if self.challenge:
                        self.discarded_labels.append(-1)
                    else:
                        self.discarded_labels.append(a[self.label_type])
                        

    def __get_snippet_features(self, point, video):
        time_stamps = self.time_step

        # compute the time stamp corresponding to the beginning of the action
        end_time_stamp = point / self.fps

        # subtract time stamps to the timestamp of the last frame
        end_time_stamp = end_time_stamp - time_stamps
        if end_time_stamp < 2:
            return None, None

        # Spanning snippets
        end_spanning = np.floor(end_time_stamp * self.fps).astype(int)
        start_spanning = max(end_spanning - (self.spanning_sec * self.fps), 0)

        # different spanning granularities (scale) for spanning feature
        select_spanning_frames1 = np.linspace(start_spanning, end_spanning, self.span_dim1 + 1, dtype=int)
        select_spanning_frames2 = np.linspace(start_spanning, end_spanning, self.span_dim2 + 1, dtype=int)
        select_spanning_frames3 = np.linspace(start_spanning, end_spanning, self.span_dim3 + 1, dtype=int)

        spanning_past = [self.__get_frames_from_indices(video, select_spanning_frames1),
                         self.__get_frames_from_indices(video, select_spanning_frames2),
                         self.__get_frames_from_indices(video, select_spanning_frames3)]

        # Recent snippets
        end_recent = end_spanning
        # different temporal granularities for recent feature 
        start_recent1 = max(end_recent - self.recent_sec1 * self.fps, 0)
        start_recent2 = max(end_recent - self.recent_sec2 * self.fps, 0)
        start_recent3 = max(end_recent - self.recent_sec3 * self.fps, 0)
        start_recent4 = max(end_recent - self.recent_sec4 * self.fps, 0)

        select_recent_frames1 = np.linspace(start_recent1, end_recent, self.recent_dim + 1, dtype=int)
        select_recent_frames2 = np.linspace(start_recent2, end_recent, self.recent_dim + 1, dtype=int)
        select_recent_frames3 = np.linspace(start_recent3, end_recent, self.recent_dim + 1, dtype=int)
        select_recent_frames4 = np.linspace(start_recent4, end_recent, self.recent_dim + 1, dtype=int)

        recent_past = [self.__get_frames_from_indices(video, select_recent_frames1),
                       self.__get_frames_from_indices(video, select_recent_frames2),
                       self.__get_frames_from_indices(video, select_recent_frames3),
                       self.__get_frames_from_indices(video, select_recent_frames4)]

        return recent_past, spanning_past

    def __get_frames_from_indices(self, video, indices):
        list_data = []
        for kkl in range(len(indices) - 1):
            cur_start = np.floor(indices[kkl]).astype('int')
            cur_end = np.floor(indices[kkl + 1]).astype('int')
            list_frames = list(range(cur_start, cur_end + 1))
            list_data.append(self.__get_frames(list_frames, video))
        return list_data

    def __get_frames(self, frames, video):
        """ format file names using the image template """
        vi = video.split('/')[-1]
        frames = np.array(list(map(lambda x: video + "/" + vi + '_' + self.img_tmpl.format(x+1), frames)))
        return frames

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        """ sample a given sequence """

        video = self.video_list[index]

        vi = video.path.split('/')[-1]
        
        # get spanning and recent frames
        recent_frames, spanning_frames = self.__get_snippet_features(video.start, video.path)

        # return a dictionary containing the id of the current sequence
        # this is useful to produce the jsons for the challenge
        out = {'id': video.id}

        # read representations for spanning and recent frames
        out['recent_features'], out['spanning_features'] = read_data(recent_frames, spanning_frames, self.env[vi],
                                                                     self.feat_dim)

        # get the label of the current sequence
        label = video.labels
        out['label'] = label

        # get the coarse labels of the current sequence
        c_label = video.c_labels
        out['c_label'] = c_label

        # get toyseq label of the current sequence
        ts_label = video.ts_labels
        out['ts_label'] = ts_label

        out['video_name'] = video.path

        return out


def read_representations(recent_frames, spanning_frames, env, feat_dim):
    """ Reads a set of representations, given their frame names and an LMDB environment."""

    recent_features1 = []
    recent_features2 = []
    recent_features3 = []
    recent_features4 = []
    spanning_features1 = []
    spanning_features2 = []
    spanning_features3 = []
    for e in env:
        spanning_features1.append(get_max_pooled_features(e, spanning_frames[0], feat_dim))
        spanning_features2.append(get_max_pooled_features(e, spanning_frames[1], feat_dim))
        spanning_features3.append(get_max_pooled_features(e, spanning_frames[2], feat_dim))

        recent_features1.append(get_max_pooled_features(e, recent_frames[0], feat_dim))
        recent_features2.append(get_max_pooled_features(e, recent_frames[1], feat_dim))
        recent_features3.append(get_max_pooled_features(e, recent_frames[2], feat_dim))
        recent_features4.append(get_max_pooled_features(e, recent_frames[3], feat_dim))

    spanning_features1 = np.concatenate(spanning_features1, axis=-1)
    spanning_features2 = np.concatenate(spanning_features2, axis=-1)
    spanning_features3 = np.concatenate(spanning_features3, axis=-1)

    recent_features1 = np.concatenate(recent_features1, axis=-1)
    recent_features2 = np.concatenate(recent_features2, axis=-1)
    recent_features3 = np.concatenate(recent_features3, axis=-1)
    recent_features4 = np.concatenate(recent_features4, axis=-1)

    spanning_snippet_features = [spanning_features1, spanning_features2, spanning_features3]
    recent_snippet_features = [recent_features1, recent_features2, recent_features3, recent_features4]

    return recent_snippet_features, spanning_snippet_features


def get_max_pooled_features(env, frame_names, feat_dim):
    list_features = []
    missing_features = []

    for kkl in range(len(frame_names)):
        with env.begin() as e:
            pool_list = []
            for name in frame_names[kkl]:
                dd = e.get(name.strip().encode('utf-8'))
                if dd is None:
                    continue
                data_curr = np.frombuffer(dd, 'float32')  # convert to numpy array
                feat_dim = data_curr.shape[0]
                pool_list.append(data_curr)

            if len(pool_list) == 0:  # Missing frames indices
                missing_features.append(kkl)
                list_features.append(np.zeros(feat_dim, dtype='float32'))
            else:
                max_pool = np.max(np.array(pool_list), 0)
                list_features.append(max_pool.squeeze())

    if(len(missing_features)>0):
        if(max(missing_features)>=len(frame_names)-1):
            for index in missing_features[::-1]:
                list_features[index] = list_features[max(index-1, 0)]
        else:
            # Reversing and adding next frames to previous frames to fill in indexes with many empty at start
            for index in missing_features[::-1]:            
                list_features[index] = list_features[index + 1]

    list_features = np.stack(list_features)
    return list_features


def read_data(recent_frames, spanning_frames, env, feat_dim):
    """A wrapper form read_representations to handle loading from more environments.
    This is used for multimodal data loading (e.g., fixed + ego)"""

    # if env is a list
    if isinstance(env, list):
        # read the representations from all environments
        return read_representations(recent_frames, spanning_frames, env, feat_dim)
    else:
        # otherwise, just read the representations
        env = [env]
        return read_representations(recent_frames, spanning_frames, env, feat_dim)
