import numpy as np
import random
import matplotlib.pyplot as plt
from Factorization import Factorization
from sklearn.metrics import f1_score
from embeded_PCA import EmbedPCA
import time

np.random.seed(42)


class HDTask():
    def __init__(self, tuples, labels, labels_sub, grouping, sample_ratio, sample_num, accu_threshold, max_epoch, enable_PCA=True, target_dim=2, scores_iter=[]):
        self.tuples = tuples
        self.labels = labels
        self.labels_sub = labels_sub
        self.grouping = grouping
        self.sample_ratio = sample_ratio
        self.sample_num = sample_num
        self.accu_threshold = accu_threshold
        self.max_epoch = max_epoch
        self.enable_PCA = enable_PCA
        self.target_dim = target_dim
        self.group_num = len(self.grouping)
        self.scores_iter = scores_iter
        self.scores = {}
        self.times = {}
        tuples_divid = []
        for col in range(self.tuples.shape[1]):
            sub_tuple = self.tuples[..., col]
            sub_tuple = (sub_tuple - np.min(sub_tuple)) / \
                (np.max(sub_tuple) - np.min(sub_tuple))
            tuples_divid.append(sub_tuple)
        self.tuples = np.vstack(tuples_divid).T
        if self.enable_PCA:
            self.pca = EmbedPCA(self.tuples, self.target_dim)
        else:
            self.pca = None
        self.exmp_pos = []
        self.exmp_neg = []
        pos_index = np.where(self.labels == 1)[0]
        real_neg = False
        neg_index = []
        for i in range(len(labels_sub)):
            if (labels_sub[i] == 0).all():
                real_neg = True
                neg_index.append(i)
        if not real_neg:
            neg_index = np.where(self.labels == 0)[0]
        if len(pos_index) == 0 or len(neg_index) == 0:
            raise ValueError('there is only one kind of label data!')
        self.exmp_pos = self.tuples[pos_index[random.randint(
            0, len(pos_index) - 1)]]
        self.exmp_neg = self.tuples[neg_index[random.randint(
            0, len(neg_index) - 1)]]

    def learn(self):
        self.model = Factorization(self.tuples, self.exmp_pos,
                                   self.exmp_neg, self.grouping,
                                   self.sample_ratio, self.sample_num,
                                   self.accu_threshold, self.max_epoch, self.pca)
        accu = 0
        epoch = 1
        labled = 0
        accumulated_time = 0.0
        while(accu < self.accu_threshold and labled < self.max_epoch):
            marked = False
            print('epoch:', epoch, 'accu:', accu)
            start_time = time.time()
            epoch += 1
            self.model.updata_data()
            if random.random() < self.sample_ratio:
                flag, data_to_label = self.model.sample_next(mode=0)
            else:
                flag, data_to_label = self.model.sample_next(mode=1)
            if flag == 0:
                marked = True
                labled += 1
                self.mark(data_to_label)
            accu = self.model.accu
            accumulated_time += time.time()-start_time
            if marked and labled in self.scores_iter:
                self.scores[str(labled)] = {'score': self.f1score(),
                                            'epoch': epoch}
                self.times[str(labled)] = {'time': accumulated_time,
                                           'epoch': epoch}
                accumulated_time = 0.0
        return self.scores, self.times

    def mark(self, data_to_label):
        index = np.zeros(len(self.tuples)) == 0
        for i in range(self.tuples.shape[1]):
            index = index & (self.tuples[..., i] == data_to_label[i])
        if np.sum(index) == 0:
            print(data_to_label)
            raise ValueError('cannot find data to label in tuples')
        like_list = list(self.labels_sub[index][0])
        like = min(like_list)
        self.model.feedback(like, like_list)

    def f1score(self):
        predict_val = self.model.predict(self.tuples)
        return f1_score(self.labels, predict_val)

    def f1score_svm(self):
        predict_val = self.model.predict_svm(self.tuples)
        return f1_score(self.labels, predict_val)

    def f1score_dsm(self):
        predict_val = self.model.predict_dsm(self.tuples)
        return f1_score(self.labels, predict_val)

    def f1score_dsm_raw(self):
        predict_val = self.model.predict_dsm_raw(self.tuples)
        predict_val = np.where(predict_val == 0, self.labels, predict_val)
        predict_val = np.where(
            predict_val == -1, np.zeros(len(predict_val)), predict_val)
        return f1_score(self.labels, predict_val)

    def f1score_sub(self):
        scores = []
        for i in range(self.group_num):
            predict_val = self.model.predict_sub(self.tuples, i)
            scores.append(f1_score(self.labels_sub[..., i], predict_val))
        return scores

    def plot_result(self):
        for i in range(self.group_num):
            points_x_pos = []
            points_y_pos = []
            points_x_neg = []
            points_y_neg = []
            for data in self.tuples:
                pred, flag = self.model.predict_sub(data, i)
                if pred == 1:
                    points_x_pos.append(data[self.grouping[i][0]])
                    points_y_pos.append(data[self.grouping[i][1]])
                elif pred == 0:
                    points_x_neg.append(data[self.grouping[i][0]])
                    points_y_neg.append(data[self.grouping[i][1]])
            plt.scatter(points_x_pos, points_y_pos, 16)
            plt.scatter(points_x_neg, points_y_neg, 16)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()

    def plot_dsm(self):
        for i in range(self.group_num):
            points_x_pos = []
            points_y_pos = []
            points_x_neg = []
            points_y_neg = []
            for data in self.tuples:
                pred, flag = self.model.predict_sub(data, i)
                if flag == 0:
                    if pred == 1:
                        points_x_pos.append(data[self.grouping[i][0]])
                        points_y_pos.append(data[self.grouping[i][1]])
                    elif pred == 0:
                        points_x_neg.append(data[self.grouping[i][0]])
                        points_y_neg.append(data[self.grouping[i][1]])
            plt.scatter(points_x_pos, points_y_pos, 16)
            plt.scatter(points_x_neg, points_y_neg, 16)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()

    def plot_svm(self):
        for i in range(self.group_num):
            points_x_pos = []
            points_y_pos = []
            points_x_neg = []
            points_y_neg = []
            for data in self.tuples:
                pred = self.model.predict_svm(data)
                if pred == 1:
                    points_x_pos.append(data[self.grouping[i][0]])
                    points_y_pos.append(data[self.grouping[i][1]])
                elif pred == 0:
                    points_x_neg.append(data[self.grouping[i][0]])
                    points_y_neg.append(data[self.grouping[i][1]])
            plt.scatter(points_x_pos, points_y_pos, 16)
            plt.scatter(points_x_neg, points_y_neg, 16)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()

    def plot_data(self):
        for i in range(self.group_num):
            points_x_pos = []
            points_y_pos = []
            points_x_neg = []
            points_y_neg = []
            for j in range(len(self.tuples)):
                if self.labels_sub[j][i] == 1:
                    points_x_pos.append(self.tuples[j][self.grouping[i][0]])
                    points_y_pos.append(self.tuples[j][self.grouping[i][1]])
                else:
                    points_x_neg.append(self.tuples[j][self.grouping[i][0]])
                    points_y_neg.append(self.tuples[j][self.grouping[i][1]])
            plt.scatter(points_x_pos, points_y_pos, 16)
            plt.scatter(points_x_neg, points_y_neg, 16)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()

    def plot_uncet(self):
        for i in range(self.group_num):
            points_x = []
            points_y = []
            for data in self.model.D_uncet:
                points_x.append(data[self.grouping[i][0]])
                points_y.append(data[self.grouping[i][1]])
            plt.scatter(points_x, points_y, 16)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
