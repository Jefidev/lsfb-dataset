import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from ..utils.annotations import get_annotations_durations, create_coerc_vec
from typing import Optional
from os import path
from lsfb_dataset.visualisation.annotations import plot_annotations_prediction


def compute_accuracy_from_conf_matrix(conf):
    return np.trace(conf) / conf.sum()


def compute_balanced_accuracy_from_conf_matrix(conf):
    recall = compute_recall_from_conf_matrix(conf)
    return recall.sum() / recall.shape[0]


def compute_recall_from_conf_matrix(conf):
    total = conf.sum(axis=0)
    return np.divide(np.diag(conf), total, where=(total != 0))
    #return np.diag(conf) / conf.sum(axis=0)


def get_binary_conf_matrix(conf):
    if conf.shape == (2, 2):
        return conf
    assert conf.shape == (3, 3), 'Unsupported confusion matrix size.'
    bin_conf = np.zeros((2, 2))
    bin_conf[0, 0] = conf[0, 0] + conf[2, 2]
    bin_conf[1, 0] = conf[1:, 0].sum() + conf[:-1, 2].sum()
    bin_conf[0, 1] = conf[0, 1] + conf[0, 2]
    bin_conf[1, 1] = conf[1, 1]
    return bin_conf


def compute_accuracy(y_true, y_pred):
    assert y_true.shape == y_pred.shape, 'True sequence has not the same shape than predicted one.'
    assert len(y_true.shape) != 1, 'Sequences are not 1-D vectors.'
    length = len(y_true.shape[0])
    assert length != 0, 'The length of the sequence is 0.'
    np.sum((y_true == y_pred)) / length


def windowed_matching(y_true, y_pred, w: int):
    assert y_true.shape == y_pred.shape, 'target and predicted vectors have different shapes'
    n = y_true.shape[0]
    offset = w // 2

    if w == 0:
        return y_true & y_true

    dist_vec = np.zeros(n, dtype=bool)
    for idx in range(n):
        in_true = y_true[idx - offset:idx - offset + w].any()
        in_pred = y_pred[idx - offset:idx - offset + w].any()
        dist_vec[idx] = in_true == in_pred

    return dist_vec


def get_transition_matrix(y, num_classes=3):
    trans_mat = np.zeros((num_classes, num_classes))
    last_x = y[0]
    for x in y:
        if x != last_x:
            trans_mat[last_x][x] += 1
            last_x = x
    return trans_mat


def plot_distributions(dist_true, dist_pred, title=None):
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.hist(dist_true, bins=100, range=(0, 5000), alpha=0.4, density=True, label='Target')
    plt.hist(dist_pred, bins=100, range=(0, 5000), alpha=0.4, density=True, label='Prediction')
    plt.legend()
    plt.show()


class ClassifierMetrics:
    def __init__(self, num_classes=2, labels=None):
        self.num_classes = num_classes

        if labels is None:
            labels = range(num_classes)
        else:
            assert len(labels) == num_classes, \
                'The number of labels must be the same as the number of classes'

        self.labels = labels
        self.best_iter_index = -1

        self.loss_evolution = []

        self.true_duration = []
        self.pred_duration = []
        self.true_transitions = []
        self.pred_transitions = []

        self.roc_curves = []
        self.confs = []

        self.current_conf = None

        # Testing metrics

        self.current_true_durations = []
        self.current_pred_durations = []
        self.current_true_transitions = []
        self.current_pred_transitions = []

        self.transitions = None

    def state_dict(self):
        return {
            'best_iter_index': self.best_iter_index,
            'loss_evolution': self.loss_evolution,
            'true_duration': self.true_duration,
            'pred_duration': self.pred_duration,
            'true_transitions': self.true_transitions,
            'pred_transitions': self.pred_transitions,
            'roc_curves': self.roc_curves,
            'confs': self.confs,
            'transitions': self.transitions,
        }

    def load_state_dict(self, state):
        self.best_iter_index = state['best_iter_index']
        self.loss_evolution = state['loss_evolution']
        self.true_duration = state['true_duration']
        self.pred_duration = state['pred_duration']
        self.true_transitions = state['true_transitions']
        self.pred_transitions = state['pred_transitions']
        self.roc_curves = state['roc_curves']
        self.confs = state['confs']

        if 'transitions' in state:
            self.transitions = state['transitions']

    def conf(self):
        return self.confs[self.best_iter_index]

    def normalized_conf(self, index=None):
        if index is None:
            index = self.best_iter_index
        conf = self.confs[index]
        return conf / conf.sum(axis=0)

    @property
    def loss(self):
        return self.loss_evolution[self.best_iter_index]

    def accuracy(self, index=None):
        if index is None:
            index = self.best_iter_index
        return compute_accuracy_from_conf_matrix(self.confs[index])

    @property
    def accuracy_evolution(self):
        acc = []
        for conf in self.confs:
            acc.append(compute_accuracy_from_conf_matrix(conf))
        return acc

    def binary_accuracy(self, index=None):
        if index is None:
            index = self.best_iter_index
        conf = get_binary_conf_matrix(self.confs[index])
        return compute_accuracy_from_conf_matrix(conf)

    @property
    def binary_accuracy_evolution(self):
        acc = []
        for idx in range(len(self.confs)):
            acc.append(self.binary_accuracy(index=idx))
        return acc

    def recall(self, index: Optional[int] = None):
        if index is None:
            index = self.best_iter_index
        conf = self.confs[index]
        recall = []
        for c in range(self.num_classes):
            class_total = conf[:, c].sum()
            if class_total == 0:
                recall.append(0)
            else:
                recall.append(conf[c, c] / class_total)
        return np.array(recall)

    def recall_evolution(self):
        recall_evo = []
        for class_index in range(self.num_classes):
            recall_evo.append([])
        for index in range(len(self.confs)):
            recall = self.recall(index=index)
            for class_index, value in enumerate(recall):
                recall_evo[class_index].append(value)

        return recall_evo

    def balanced_accuracy(self, index=None):
        return self.recall(index=index).sum() / self.num_classes

    @property
    def balanced_accuracy_evolution(self):
        acc = []
        for idx, conf in enumerate(self.confs):
            acc.append(self.recall(index=idx).sum() / self.num_classes)
        return acc

    def binary_balanced_accuracy(self, index=None):
        if index is None:
            index = self.best_iter_index
        conf = get_binary_conf_matrix(self.confs[index])
        return compute_balanced_accuracy_from_conf_matrix(conf)

    @property
    def binary_balanced_accuracy_evolution(self):
        acc = []
        for idx in range(len(self.confs)):
            acc.append(self.binary_balanced_accuracy(index=idx))
        return acc

    @property
    def roc_curve(self):
        return self.roc_curves[self.best_iter_index]

    def roc_auc(self, index=None):
        if index is None:
            index = self.best_iter_index
        curves = self.roc_curves[index]

        auc_scores = []
        for c in range(len(curves)):
            fpr, tpr, _ = curves[c]
            auc_scores.append(auc(fpr, tpr))

        return auc_scores

    def add_predictions(self, y_true, y_pred):
        if self.current_conf is None:
            self.current_conf = np.zeros((self.num_classes, self.num_classes))
        assert y_true.shape == y_pred.shape, 'Targets and predictions have different shapes.'

        for idx in range(y_true.shape[0]):
            self.current_conf += confusion_matrix(y_true[idx], y_pred[idx], labels=self.labels)

    def add_duration_distributions(self, y_true, y_pred):
        for idx in range(y_true.shape[0]):
            self.current_true_durations += get_annotations_durations(y_true[idx])
            self.current_pred_durations += get_annotations_durations(y_pred[idx] > 0)

    def add_transition_distributions(self, y_true, y_pred):
        assert self.num_classes == 2 or self.num_classes == 3, \
            'Wrong classes number for transitions.'

        for idx in range(y_true.shape[0]):
            annot_true = y_true[idx]
            annot_pred = y_pred[idx]

            if self.num_classes == 2:
                annot_true = create_coerc_vec(annot_true)
                annot_pred = create_coerc_vec(annot_pred > 0)

            self.current_true_transitions += get_annotations_durations(annot_true, value=2)
            self.current_pred_transitions += get_annotations_durations(annot_pred, value=2)

    def add_roc_curve(self, y_true, y_score):
        curves = []
        for c in range(self.num_classes):
            fpr, tpr, thresholds = roc_curve(y_true, y_score[:, c], pos_label=c)
            curves.append((fpr, tpr, thresholds))
        self.roc_curves.append(curves)

    def add_loss(self, loss: float):
        self.loss_evolution.append(loss)

    def add_transitions_matrix(self, y_pred):
        if self.transitions is None:
            self.transitions = torch.zeros((self.num_classes, self.num_classes))
        for pred in y_pred:
            self.transitions += get_transition_matrix(pred, num_classes=self.num_classes)

    def commit(self):
        current_balanced_acc = compute_balanced_accuracy_from_conf_matrix(self.current_conf)
        if current_balanced_acc > self.balanced_accuracy():
            self.best_iter_index = len(self.confs)

        self.confs.append(self.current_conf)
        self.current_conf = None

        if len(self.current_true_durations) > 0:
            self.true_duration.append(self.current_true_durations)
            self.pred_duration.append(self.current_pred_durations)
            self.true_transitions.append(self.current_true_transitions)
            self.pred_transitions.append(self.current_pred_transitions)
            self.current_true_durations = []
            self.current_pred_durations = []
            self.current_true_transitions = []
            self.current_pred_transitions = []

    def refresh_best_iter_index(self):
        self.best_iter_index = np.argmax(self.balanced_accuracy_evolution)

    def plot_conf(self, normalized=False, index=None):
        if index is None:
            index = self.best_iter_index

        plt.figure()
        plt.title('Confusion matrix')

        if normalized:
            sn.heatmap(self.normalized_conf(index), annot=True, fmt='.4f', cmap='flare')
        else:
            sn.heatmap(self.confs[index], annot=True, fmt='.0f', cmap='flare')

        plt.show()

    def plot_transition_matrix(self):
        assert self.transitions is not None, 'Transition matrix not added to this metric.'

        plt.figure()
        plt.title('Transition matrix')

        sn.heatmap(self.transitions, annot=True, fmt='.0f', cmap='flare')

        plt.show()

    def plot_duration_distributions(self, index=None):
        if index is None:
            index = self.best_iter_index

        plot_distributions(self.true_duration[index], self.pred_duration[index], 'Annotation duration distributions')

    def plot_transition_distributions(self, index=None):
        if index is None:
            index = self.best_iter_index

        plot_distributions(self.true_transitions[index], self.pred_transitions[index],
                           'Annotation transition duration distributions')

    def plot_roc_curve(self, index=None):
        if index is None:
            index = self.best_iter_index

        curves = self.roc_curves[index]
        roc_auc = self.roc_auc(index)

        fig, ax1 = plt.subplots()
        ax1.set_title('Roc curve')
        for c in range(len(curves)):
            fpr, tpr, _ = curves[c]
            score = roc_auc[c]
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=score)
            display.plot(ax=ax1, label=f'AUC of class {c}: {score:.4f}')
        plt.show()

    def get_results(self):
        if len(self.confs) == 0:
            return None

        data = {
            'epoch': range(1, len(self.confs) + 1),
            'accuracy': self.accuracy_evolution,
            'balanced_accuracy': self.balanced_accuracy_evolution,
            'loss': self.loss_evolution,
        }

        recall_evolution = self.recall_evolution()
        for class_index, class_recall_evo in enumerate(recall_evolution):
            data[f'recall_{class_index}'] = class_recall_evo

        return pd.DataFrame.from_dict(data)


class VideoSegmentationRecords:
    def __init__(self, model_name: str, isolate_transitions=False, video_names=None, plots_dir=None):
        self.model_name = model_name
        self.isolate_transitions = isolate_transitions

        self.video_names = video_names.reset_index(drop=True)
        self.plots_dir = plots_dir

        self.num_classes = 2
        self.columns = ['acc', 'balanced_acc', 'recall_waiting', 'recall_talking']

        if isolate_transitions:
            self.num_classes = 3
            self.columns += ['acc_with_transitions', 'balanced_acc_with_transitions',
                             'recall_waiting_with_transitions', 'recall_talking_with_transitions', 'recall_transitions']

        self.records = []

    def add_record(self, y_true: torch.Tensor, y_pred: torch.Tensor, likelihood=None):
        conf = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        acc = compute_accuracy_from_conf_matrix(conf)
        balanced_acc = compute_balanced_accuracy_from_conf_matrix(conf)
        recall = compute_recall_from_conf_matrix(conf)

        if self.plots_dir is not None:
            video_idx = len(self.records)
            filename = f'Video_{video_idx}.png'
            if self.video_names is not None:
                filename = f'{self.video_names.iloc[video_idx]}.png'

            fig, _ = plot_annotations_prediction(filename, y_true, y_pred, likelihood=likelihood)
            fig.savefig(path.join(self.plots_dir, filename), dpi=300)
            plt.close()

        if self.isolate_transitions:
            bin_conf = get_binary_conf_matrix(conf)
            bin_balanced_acc = compute_balanced_accuracy_from_conf_matrix(bin_conf)
            bin_acc = compute_accuracy_from_conf_matrix(bin_conf)
            bin_recall = compute_recall_from_conf_matrix(bin_conf)
            record = (bin_acc, bin_balanced_acc, *bin_recall, acc, balanced_acc, *recall)
            self.records.append(record)
        else:
            record = (acc, balanced_acc, *recall)
            self.records.append(record)

    def get_records(self) -> pd.DataFrame:
        records = pd.DataFrame.from_records(self.records, columns=self.columns)

        if self.video_names is not None:
            assert self.video_names.shape[0] == records.shape[0], 'Invalid number of video names.'
            records['filename'] = self.video_names

        return records
