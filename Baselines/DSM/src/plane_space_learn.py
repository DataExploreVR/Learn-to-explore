import random
from sklearn import svm
from OptDsm import *
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


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


def fakeUser(point, error_prob=0.03):
    true_result = point[1] > 1/(5*point[0])
    if random.random() < error_prob:
        if true_result == 1:
            return 0
        else:
            return 1
    else:
        return true_result


class PlaneLearn():
    def __init__(self, D_eval, x_pos, x_neg, sample_ratio, sample_num, fake_user=None):
        self.D_eval = D_eval.copy()
        self.x_pos = x_pos
        self.x_neg = x_neg
        self.sample_ratio = sample_ratio
        self.sample_num = sample_num
        self.fake_user = fake_user
        self.D_pos = []
        self.D_neg = []
        self.D_uncet = self.D_eval.copy()
        self.D_labeled = {'pos': [x_pos], 'neg': [x_neg]}
        self.D_unlabeled = self.D_eval.tolist()
        self.D_unlabeled.remove(list(x_pos))
        self.D_unlabeled.remove(list(x_neg))
        self.D_labeled_by_user = {'pos': [], 'neg': []}
        self.D_labeled_by_dsm = {'pos': [], 'neg': []}
        self.accu = 0
        self.init_flag = True
        self.completed = False

    def update_data(self):
        if len(self.D_uncet) == 0:
            self.completed = True
        if self.completed:
            return 0
        if self.init_flag:
            self.init_flag = False
            self.DSM = OptDataSpaceModel(self.x_pos, self.x_neg)
        else:
            for pos_point in self.D_labeled_by_user['pos']:
                self.DSM.add_pos_point(pos_point)
            for neg_point in self.D_labeled_by_user['neg']:
                self.DSM.add_neg_point(neg_point)
        check_num = len(self.D_uncet)
        print(check_num, 'pieces of data are ready to check...')
        self.D_pos, self.D_neg, self.D_uncet = self.DSM.update_three_sets(
            self.D_pos, self.D_neg, self.D_uncet)
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
        print('training svm with length of ', len(self.train_data))
        self.clf = svm.SVC(kernel='rbf', gamma="scale")
        self.clf.fit(self.train_data, self.train_value)
        self.D_labeled_by_user = {'pos': [], 'neg': []}
        self.D_labeled_by_dsm = {'pos': [], 'neg': []}

    def get_next_to_label(self):
        if len(self.D_uncet) == 0:
            self.completed = True
        if self.completed:
            self.next_sample = self.D_eval[random.randint(
                0, len(self.D_eval)-1)]
            return self.next_sample
        pool = reservoir_sample(self.D_uncet, len(
            self.D_uncet), self.sample_num)
        pool_dists = self.clf.decision_function(pool)
        pool_dists = [abs(val) for val in pool_dists]
        val, idx = min((val, idx)
                       for (idx, val) in enumerate(pool_dists))
        self.next_sample = pool[idx]
        return self.next_sample

    def feedback(self, like):
        if self.completed:
            return 0
        if like == 1:
            self.D_labeled_by_user['pos'].append(self.next_sample.copy())
        else:
            self.D_labeled_by_user['neg'].append(self.next_sample.copy())

    def judge_by_dsm(self):
        if len(self.D_uncet) == 0:
            self.completed = True
        if self.completed:
            return (1, None)
        pool = reservoir_sample(self.D_unlabeled, len(
            self.D_unlabeled), self.sample_num)
        pool_dists = self.clf.decision_function(pool)
        pool_dists = [abs(val) for val in pool_dists]
        val, idx = min((val, idx)
                       for (idx, val) in enumerate(pool_dists))
        self.next_sample = pool[idx]
        next_sample_val = self.DSM.get_single_point_region(self.next_sample)
        if next_sample_val == 1:
            self.D_labeled_by_dsm['pos'].append(self.next_sample.copy())
            return (1, self.next_sample)
        elif next_sample_val == -1:
            self.D_labeled_by_dsm['neg'].append(self.next_sample.copy())
            return (-1, self.next_sample)
        else:
            return (0, self.next_sample)

    def predict(self, points):
        dsm_pre = self.DSM.get_points_region(points)
        clf_pre = self.clf.predict(points)
        combine = np.where(dsm_pre == 0, clf_pre, dsm_pre)
        return np.where(combine == -1, np.zeros(len(combine)), combine)

    def predict_dsm(self, points):
        dsm_pre = self.DSM.get_points_region(points)
        return np.where(dsm_pre == -1, np.zeros(len(dsm_pre)), dsm_pre)

    def predict_dsm_raw(self, points):
        return self.DSM.get_points_region(points)

    def predict_svm(self, points):
        return self.clf.predict(points)

    def plot_dsm_pos(self):
        points_x_pos = []
        points_y_pos = []
        points_x_neg = []
        points_y_neg = []
        for point in self.D_eval:
            if self.DSM.get_single_point_region(point) == 1:
                points_x_pos.append(point[0])
                points_y_pos.append(point[1])
            elif self.DSM.get_single_point_region(point) == -1:
                points_x_neg.append(point[0])
                points_y_neg.append(point[1])
            else:
                pass
        plt.scatter(points_x_pos, points_y_pos, 16)
        plt.scatter(points_x_neg, points_y_neg, 16)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    def plot_svm(self):
        points_x_pos = []
        points_y_pos = []
        points_x_neg = []
        points_y_neg = []
        for point in self.D_eval:
            pred = self.clf.predict([point])[0]
            if pred == 1:
                points_x_pos.append(point[0])
                points_y_pos.append(point[1])
            elif pred == 0:
                points_x_neg.append(point[0])
                points_y_neg.append(point[1])
            else:
                raise ValueError('predict value is not binary')
        plt.scatter(points_x_pos, points_y_pos, 16)
        plt.scatter(points_x_neg, points_y_neg, 16)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()
