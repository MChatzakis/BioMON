import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate
from methods.heads import *

import wandb


class BioMetaOptNet(MetaTemplate):
    """
    BioMetaOptNet is a MetaOptNet variant for Biomedical data collections.
    """

    def __init__(self, backbone, n_way, n_support, head_model_params):
        """
        Initialize BioMetaOptNet model.

        Args:
            backbone (model): Backbone model to use for feature extraction.
            n_way (int): Number of classes per task.
            n_support (int): Number of support samples per class.
            head_model_params (dict): Dictionary of parameters for the head model.

        """

        super(BioMetaOptNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.head_model_params = head_model_params

    def set_forward(self, x, is_feature=False, calc_scores=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)

        z_support_labels = torch.from_numpy(
            np.repeat(range(self.n_way), self.n_support)
        )

        z_query_labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))

        head_args = self.head_model_params
        head = DISPATCHER[head_args["model"]](
            **head_args["args"], feat_dim=z_support.shape[1]
        )
        head.fit(z_support, z_support_labels)

        scores = head.get_logits(z_query)

        head_test_acc = head.test(z_query, z_query_labels)
        head_train_acc = head.test(z_support, z_support_labels)
        head_fit_time = head.fit_time

        return scores, head_train_acc, head_test_acc, head_fit_time

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        if torch.cuda.is_available():
            y_query = Variable(y_query.cuda())
        else:
            y_query = Variable(y_query)

        scores, head_train_acc, head_test_acc, head_fit_time = self.set_forward(x)

        return (
            self.loss_fn(scores, y_query),
            head_train_acc,
            head_test_acc,
            head_fit_time,
        )

    def correct(self, x):
        scores, head_train_acc, head_test_acc, head_fit_time = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        if torch.cuda.is_available():
            y_query = Variable(y_query.cuda())
        else:
            y_query = Variable(y_query)
        loss = self.loss_fn(scores, y_query)

        return (
            float(top1_correct),
            len(y_query),
            head_train_acc,
            head_test_acc,
            head_fit_time,
            loss.item(),
        )

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10

        total_head_fit_time = 0

        avg_loss = 0
        avg_head_train_acc = 0
        avg_head_test_acc = 0
        avg_head_fit_time = 0
        for i, (x, _) in enumerate(train_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else:
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
            optimizer.zero_grad()
            loss, head_train_acc, head_test_acc, head_fit_time = self.set_forward_loss(
                x
            )
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            avg_head_train_acc = avg_head_train_acc + head_train_acc
            avg_head_test_acc = avg_head_test_acc + head_test_acc
            avg_head_fit_time = avg_head_fit_time + head_fit_time

            total_head_fit_time += head_fit_time

            if i % print_freq == 0:
                print(
                    "Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Head Train Acc {:f} | Head Test Acc {:f} | Head Fit Time {:f}".format(
                        epoch,
                        i,
                        len(train_loader),
                        avg_loss / float(i + 1),
                        avg_head_train_acc / float(i + 1),
                        avg_head_test_acc / float(i + 1),
                        avg_head_fit_time / float(i + 1),
                    )
                )
                wandb.log({"loss": avg_loss / float(i + 1)})
                wandb.log({"head_train_acc": avg_head_train_acc / float(i + 1)})
                wandb.log({"head_test_acc": avg_head_test_acc / float(i + 1)})
                wandb.log({"head_fit_time": avg_head_test_acc / float(i + 1)})

        avg_head_train_acc = avg_head_train_acc / len(train_loader)
        avg_head_test_acc = avg_head_test_acc / len(train_loader)
        avg_head_fit_time = avg_head_fit_time / len(train_loader)
        avg_loss = avg_loss / len(train_loader)

        return (
            avg_loss,
            avg_head_train_acc,
            avg_head_test_acc,
            avg_head_fit_time,
            total_head_fit_time,
        )

    def test_loop(self, test_loader, record=None, return_std=False):
        correct = 0
        count = 0

        acc_all = []
        loss_all = []

        head_train_acc_all = []
        head_test_acc_all = []
        head_fit_time_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else:
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)

            (
                correct_this,
                count_this,
                head_train_acc,
                head_test_acc,
                head_fit_time,
                loss,
            ) = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

            loss_all.append(loss)
            head_train_acc_all.append(head_train_acc)
            head_test_acc_all.append(head_test_acc)
            head_fit_time_all.append(head_fit_time)

        acc_all = np.asarray(acc_all)

        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print(
            "%d Test Acc = %4.2f%% +- %4.2f%%"
            % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        )

        head_train_acc_all = np.asarray(head_train_acc_all)
        head_test_acc_all = np.asarray(head_test_acc_all)
        head_fit_time_all = np.asarray(head_fit_time_all)

        head_train_acc_mean = np.mean(head_train_acc_all)
        head_train_acc_std = np.std(head_train_acc_all)

        head_test_acc_mean = np.mean(head_test_acc_all)
        head_test_acc_std = np.std(head_test_acc_all)

        head_fit_time_mean = np.mean(head_fit_time_all)
        head_fit_time_std = np.std(head_fit_time_all)

        loss_all = np.asarray(loss_all)
        loss_mean = np.mean(loss_all)
        loss_std = np.std(loss_all)

        total_head_fit_time = np.sum(head_fit_time_all)

        if return_std:
            return (
                acc_mean,
                acc_std,
                head_train_acc_mean,
                head_train_acc_std,
                head_test_acc_mean,
                head_test_acc_std,
                head_fit_time_mean,
                head_fit_time_std,
                loss_mean,
                loss_std,
                total_head_fit_time,
            )
        else:
            return (
                acc_mean,
                head_train_acc_mean,
                head_test_acc_mean,
                head_fit_time_mean,
                loss_mean,
                total_head_fit_time,
            )
