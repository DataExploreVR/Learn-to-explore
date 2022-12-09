import random
from OptDsm import *
from sklearn import svm
import numpy as np


def reservoir_sample(data, data_num, target_num):
    if target_num >= data_num:
        return data
    reservoir = []
    for i in range(target_num):
        reservoir.append(data[i])
    while(i < data_num):
        j = random.randrange(i+1)
        if (j < target_num):
            reservoir[j] = data[i]
        i += 1
    return reservoir


class Factorization():

    def __init__(self, D_eval, pos_example, neg_example, grouping, sample_ratio, sample_num, accu_threshold, max_epoch, pca_model):
        self.D_eval = D_eval
        self.pos_example = pos_example
        self.neg_example = neg_example
        self.grouping = grouping
        self.group_num = len(grouping)
        self.sample_ratio = sample_ratio
        self.sample_num = sample_num
        self.accu_threshold = accu_threshold
        self.pca = pca_model
        self.max_epoch = max_epoch

        self.D_pos = []
        self.D_neg = []
        self.D_uncet = self.D_eval.copy()
        self.D_labeled = {'pos': [pos_example], 'neg': [neg_example]}
        self.D_unlabeled = self.D_eval.tolist()
        self.D_unlabeled.remove(list(pos_example))
        self.D_unlabeled.remove(list(neg_example))
        self.D_labeled_by_user = {'pos': [], 'neg': []}
        self.user_label_subs = []
        self.D_labeled_by_dsm = {'pos': [], 'neg': []}
        self.accu = 0
        self.init_flag = True

    def updata_data(self):
        if self.init_flag:
            self.init_flag = False
            self.DSMs = []
            for group in self.grouping:
                self.DSMs.append(OptDataSpaceModel([self.pos_example[group[0]],
                                                    self.pos_example[group[1]]],
                                                   [self.neg_example[group[0]],
                                                    self.neg_example[group[1]]]))
        else:
            for pos_point in self.D_labeled_by_user['pos']:
                for j in range(self.group_num):
                    self.DSMs[j].add_pos_point(
                        [pos_point[self.grouping[j][0]],
                            pos_point[self.grouping[j][1]]])
            for i in range(len(self.D_labeled_by_user['neg'])):
                user_label_sub = self.user_label_subs[i]
                neg_point = self.D_labeled_by_user['neg'][i]
                for j in range(self.group_num):
                    if user_label_sub[j] == 1:
                        self.DSMs[j].add_pos_point([neg_point[self.grouping[j][0]],
                                                    neg_point[self.grouping[j][1]]])
                    else:
                        self.DSMs[j].add_neg_point([neg_point[self.grouping[j][0]],
                                                    neg_point[self.grouping[j][1]]])
            self.user_label_subs = []
        mark_list = []
        for i in range(self.group_num):
            mark_list.append(self.DSMs[i].get_points_region(
                self.D_uncet[..., self.grouping[i]]))
        mark_list = np.min(np.vstack(mark_list).T, axis=-1)
        self.D_pos.extend(self.D_uncet[mark_list == 1])
        self.D_neg.extend(self.D_uncet[mark_list == -1])
        self.D_uncet = self.D_uncet[mark_list == 0]
        self.accu = len(self.D_pos)/(len(self.D_pos) + len(self.D_uncet))
        for pos_point in self.D_labeled_by_user['pos']:
            self.D_labeled['pos'].append(pos_point)
            if list(pos_point) in self.D_unlabeled:
                self.D_unlabeled.remove(list(pos_point))
        for neg_point in self.D_labeled_by_user['neg']:
            self.D_labeled['neg'].append(neg_point)
            if list(neg_point) in self.D_unlabeled:
                self.D_unlabeled.remove(list(neg_point))
        for pos_point in self.D_labeled_by_dsm['pos']:
            self.D_labeled['pos'].append(pos_point)
            if list(pos_point) in self.D_unlabeled:
                self.D_unlabeled.remove(list(pos_point))
        for neg_point in self.D_labeled_by_dsm['neg']:
            self.D_labeled['neg'].append(neg_point)
            if list(neg_point) in self.D_unlabeled:
                self.D_unlabeled.remove(list(neg_point))
        self.train_data = []
        self.train_value = []
        for pos_data in self.D_labeled['pos']:
            self.train_data.append(pos_data)
            self.train_value.append(1)
        for neg_data in self.D_labeled['neg']:
            self.train_data.append(neg_data)
            self.train_value.append(0)
        self.clf = svm.SVC(kernel='rbf', gamma="scale")
        self.clf.fit(self.train_data if self.pca is None else self.pca.transform(
            self.train_data), self.train_value)
        self.D_labeled_by_user = {'pos': [], 'neg': []}
        self.D_labeled_by_dsm = {'pos': [], 'neg': []}

    def sample_next(self, mode=0):
        if mode == 0:
            if len(self.D_uncet) == 0:
                pool = [self.D_eval[random.randint(0, len(self.D_eval)-1)]]
            else:
                pool = reservoir_sample(self.D_uncet, len(
                    self.D_uncet), self.sample_num)
        else:
            pool = reservoir_sample(self.D_unlabeled, len(
                self.D_unlabeled), self.sample_num)
        pool_dists = self.clf.decision_function(
            pool if self.pca is None else self.pca.transform(pool))
        pool_dists = [abs(val) for val in pool_dists]
        val, idx = min((val, idx)
                       for (idx, val) in enumerate(pool_dists))
        self.next_sample = pool[idx]
        if mode == 1:
            mark_list = []
            for i in range(self.group_num):
                mark_list.append(self.DSMs[i].get_single_point_region(
                    [self.next_sample[self.grouping[i][0]],
                     self.next_sample[self.grouping[i][1]]]))
            mark = min(mark_list)
            if mark == 1:
                self.D_labeled_by_dsm['pos'].append(self.next_sample.copy())
                return (1, self.next_sample)
            elif mark == -1:
                self.D_labeled_by_dsm['neg'].append(self.next_sample.copy())
                return (-1, self.next_sample)
            else:
                return (0, self.next_sample)
        return (0, self.next_sample)

    def feedback(self, like, like_list):
        if like == 1:
            self.D_labeled_by_user['pos'].append(self.next_sample.copy())
        else:
            self.D_labeled_by_user['neg'].append(self.next_sample.copy())
            self.user_label_subs.append(like_list)

    def predict(self, data):
        mark_list = []
        for i in range(self.group_num):
            mark_list.append(self.DSMs[i].get_points_region(
                data[..., self.grouping[i]]))
        mark_list = np.min(np.vstack(mark_list).T, axis=-1)
        if self.pca is None:
            clf_pre = self.clf.predict(data)
        else:
            clf_pre = self.clf.predict(self.pca.transform(data))
        combine = np.where(mark_list == 0, clf_pre, mark_list)
        return np.where(combine == -1, np.zeros(len(combine)), combine)

    def predict_sub(self, data, groupid):
        dsm_pred = self.DSMs[groupid].get_point_region(
            [data[self.grouping[groupid][0]],
             data[self.grouping[groupid][1]]])
        dsm_pre = self.DSMs[groupid].get_points_region(
            data[..., self.grouping[groupid]])
        if self.pca is None:
            clf_pre = self.clf.predict(data)
        else:
            clf_pre = self.clf.predict(self.pca.transform(data))
        combine = np.where(dsm_pre == 0, clf_pre, dsm_pre)
        return np.where(combine == -1, np.zeros(len(combine)), combine)

    def predict_svm(self, data):
        return self.clf.predict(data if self.pca is None else self.pca.transform(data))

    def predict_dsm(self, data):
        mark_list = []
        for i in range(self.group_num):
            mark_list.append(self.DSMs[i].get_points_region(
                data[..., self.grouping[i]]))
        mark_list = np.min(np.vstack(mark_list).T, axis=-1)
        return np.where(mark_list == -1, np.zeros(len(mark_list)), mark_list)

    def predict_dsm_raw(self, data):
        mark_list = []
        for i in range(self.group_num):
            mark_list.append(self.DSMs[i].get_points_region(
                data[..., self.grouping[i]]))
        mark_list = np.min(np.vstack(mark_list).T, axis=-1)
        return mark_list
