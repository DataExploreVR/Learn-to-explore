import numpy as np
import random
import matplotlib.pyplot as plt
from plane_space_learn import PlaneLearn
from sklearn.metrics import f1_score
import time


class TDTask():
    def __init__(self, tuples, labels, sample_ratio, sample_num, accu_threshold, max_epoch, scores_iter=[]):
        self.sample_ratio = sample_ratio
        self.sample_num = sample_num
        self.accu_threshold = accu_threshold
        self.max_epoch = max_epoch
        self.scores_iter = scores_iter
        self.scores = {}
        self.times = {}

        x_tuples = tuples[..., 0]
        y_tuples = tuples[..., 1]
        x_tuples = (x_tuples - np.min(x_tuples)) / \
            (np.max(x_tuples) - np.min(x_tuples))
        y_tuples = (y_tuples - np.min(y_tuples)) / \
            (np.max(y_tuples) - np.min(y_tuples))
        self.tuples = np.vstack((x_tuples, y_tuples)).T
        self.labels = labels
        self.D_eval = self.tuples
        pos = [i for i, x in enumerate(self.labels) if x == 1]
        if len(pos) == 0:
            raise ValueError('no positive data in tuples')
        else:
            self.x_pos = self.tuples[random.sample(pos, 1)[0]]
        neg = [i for i, x in enumerate(self.labels) if x == 0]
        if len(neg) == 0:
            raise ValueError('no negative data in tuples')
        else:
            self.x_neg = self.tuples[random.sample(neg, 1)[0]]

    def learn(self):
        self.model = PlaneLearn(self.D_eval, self.x_pos,
                                self.x_neg, self.sample_ratio, self.sample_num, None)
        accu = 0
        epoch = 1
        labled = 0
        accumulated_time = 0.0
        while(accu < self.accu_threshold and epoch < self.max_epoch):
            marked = False
            print('epoch:', epoch, 'accu:', accu)
            start_time = time.time()
            epoch += 1
            self.model.update_data()
            if random.random() < self.sample_ratio:
                self.mark(None)
            else:
                flag, data_to_label = self.model.judge_by_dsm()
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
                accumulated_time = 0
        return self.scores, self.times

    def mark(self, data_to_label):
        if data_to_label == None:
            data_to_label = self.model.get_next_to_label()
        index = (self.tuples[..., 0] == data_to_label[0]) & (
            self.tuples[..., 1] == data_to_label[1])
        if np.sum(index) == 0:
            raise ValueError('no matched data in query tuples')
        else:
            self.model.feedback(self.labels[index][0])

    def f1score(self):
        predict_val = self.model.predict(self.D_eval)
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

    def plot_data(self):
        pos = [i for i, x in enumerate(self.labels) if x == 1]
        if len(pos) == 0:
            raise ValueError('no positive data in tuples')
        else:
            points_x = []
            points_y = []
            for index in pos:
                points_x.append(self.tuples[index][0])
                points_y.append(self.tuples[index][1])
            plt.scatter(points_x, points_y, 16)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()

    def plot_result(self):
        predict_val = self.model.predict(self.D_eval)
        points_x = np.where(predict_val == 1, self.D_eval, None)
        points_y = np.where(predict_val == 0, self.D_eval, None)
        plt.scatter(points_x[..., 0], points_x[..., 1], 16)
        plt.scatter(points_y[..., 0], points_y[..., 1], 16)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()
        self.model.plot_dsm_pos()
        self.model.plot_svm()
