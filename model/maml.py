import os.path as osp
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import backbones
from util.file import exists_file
from util.metrics import Metricser, MetricsList
from util.torchtool import load_checkpoint, set_requries_grad, load_pretrained_weights, SigmoidMapper


class MAML(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """

        :param args:
        """
        super(MAML, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        
        # feture extraction---parameters fixed
        model_class = backbones[args.backbone]
        if args.pretrain_type == 'imagenet':
            print(f'Using pretrained backbone({args.backbone}) from imagenet')
            self.net = model_class(True, num_classes=1)
        elif args.pretrain_type == 'fea':
            pth_file = osp.join(args.work_dir, args.dataset, args.backbone, 'models', 'epoch-last.pth')
            print(f'Using pretrained backbone({args.backbone}) from fea file({pth_file})')
            self.net = model_class(False, num_classes=1)
            assert exists_file(pth_file), f'pth file({pth_file}) not found'
            state_dict = load_checkpoint(pth_file)
            load_pretrained_weights(self.net, state_dict)
        elif args.pretrain_type == 'none':
            print(f'Warning: Using the backbone({args.backbone}) without pretrained-initialization')
            self.net = model_class(False, num_classes=1)
        else:
            raise RuntimeError(f'Unknown pretrain type:{args.pretrain_type}')
        self.net = self.net.cuda()
        # fc Layer--->meta
        self.meta = nn.Linear(self.net.fea_dim, 1, bias=True).cuda()
        self.optimizer = torch.optim.Adam(self.meta.parameters(), lr=self.meta_lr)
        self.score_mapper = SigmoidMapper(args.n_way - 1)

        nn.init.kaiming_normal_(self.meta.weight.data)
        nn.init.constant_(self.meta.bias.data, 0)

#    def clip_grad_by_norm_(self, grad, max_norm):
#        """
#        in-place gradient clipping.
#        :param grad: list of gradients
#        :param max_norm: maximum norm allowable
#        :return:sle
#        """
#
#        total_norm = 0
#        counter = 0
#        for g in grad:
#            param_norm = g.data.norm(2)
#            total_norm += param_norm.item() ** 2
#            counter += 1
#        total_norm = total_norm ** (1. / 2)
#
#        clip_coef = max_norm / (total_norm + 1e-6)
#        if clip_coef < 1:
#            for g in grad:
#
    #                g.data.mul_(clip_coef)
#
#        return total_norm/counter

    def meta_train(self, x_spt, y_spt, x_qry, y_qry, mode='rebirth'):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        set_requries_grad(self.net, False)
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        y_spt = y_spt.type(torch.FloatTensor).cuda()
        y_qry = y_qry.type(torch.FloatTensor).cuda()

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        correlations = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            feat = self.net(x_spt[i], feature_only=True)
            pred = self.meta(feat.detach()).squeeze(1)
            # pdb.set_trace()
            # pred = torch.sigmoid(pred) * 4
            pred = self.score_mapper(pred)
            loss = F.mse_loss(pred, y_spt[i])

            # Adaption
            loss.backward()
            with torch.no_grad():
                grad = self.meta.weight.grad.clone().detach()
                # Only adapt the meta layer
            weight_adapted = self.meta.weight - self.update_lr * grad
            self.meta.zero_grad()

            # this is the loss and accuracy before first update
            with torch.no_grad():
                feat_q = self.net(x_qry[i], feature_only=True)
                pred_q = self.meta(feat_q.detach()).squeeze(1)
                # pred_q = torch.sigmoid(pred_q) * 4
                pred_q = self.score_mapper(pred_q)
                loss_q = F.mse_loss(pred_q, y_qry[i])
                losses_q[0] += loss_q

                # pdb.set_trace()
                correlation = np.corrcoef(pred_q.cpu().numpy(), y_qry[i].cpu().numpy())[0][1]
                # correlation = abs(pred_q - y_qry[i]).mean()
                correlations[0] = correlations[0] + correlation

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                pred_q = F.linear(feat_q.detach(), weight_adapted).squeeze(1)
                # pred_q = torch.sigmoid(pred_q) * 4
                pred_q = self.score_mapper(pred_q)
                loss_q = F.mse_loss(pred_q, y_qry[i])
                losses_q[1] += loss_q

                correlation = np.corrcoef(pred_q.cpu().numpy(), y_qry[i].cpu().numpy())[0][1]
                correlations[1] = correlations[1] + correlation

            for k in range(1, self.update_step):
                # run the i-th task and compute loss for k=1~K-1
                pred = F.linear(feat.detach(), weight_adapted).squeeze(1)
                # pred = torch.sigmoid(pred) * 4
                pred = self.score_mapper(pred)

                loss = F.mse_loss(pred, y_spt[i])
                grad = torch.autograd.grad(loss, weight_adapted)
                weight_adapted = weight_adapted - self.update_lr * grad[0]

                pred_q = F.linear(feat_q.detach(), weight_adapted).squeeze(1)
                # pred_q = torch.sigmoid(pred_q) * 4
                pred_q = self.score_mapper(pred_q)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.mse_loss(pred_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    correlation = np.corrcoef(pred_q.cpu().numpy(), y_qry[i].cpu().numpy())[0][1]
                    correlations[k + 1] = correlations[k + 1] + correlation

        # pdb.set_trace()
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_sum = losses_q[-1] / task_num
        # loss_sum.requires_grad = True
        accs_sum = np.array(correlations, dtype=np.float64) / task_num
        # accs_sum = np.array(correlations, dtype=np.float64) / (task_num * querysz)
        # optimize theta parameters
        self.optimizer.zero_grad()
        # loss_sum.requires_grad = True
        loss_sum.backward()
        self.optimizer.step()
        set_requries_grad(self.net, True)

        return loss_sum, accs_sum

    def meta_test(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        net = deepcopy(self.net)
        set_requries_grad(self.net, False)
        # self.net.eval()
        assert len(x_spt.shape) == 4
        querysz = x_qry.size(0)
        y_spt = y_spt.type(torch.FloatTensor).cuda()
        y_qry = y_qry.type(torch.FloatTensor).cuda()

        metricser = Metricser()
        res_list = MetricsList(self.update_step_test + 1, 0.0)
        # correlations = [0 for _ in range(self.update_step_test + 1)]
        
        # meta = copy.deepcopy(self.meta)
        meta = nn.Linear(self.net.fea_dim, 1, bias=True).cuda()
        for p_new, p_model in zip(meta.parameters(), self.meta.parameters()):
            device = p_new.device
            p_model_ = p_model.detach().to(device)
            p_new.detach().copy_(p_model_)
        feat = net(x_spt, feature_only=True)
        pred = meta(feat.detach()).squeeze(1)
        # pred = torch.sigmoid(pred) * 4
        pred = self.score_mapper(pred)
        loss = F.mse_loss(pred, y_spt)

        # Adaption
        loss.backward()
        with torch.no_grad():
            grad = meta.weight.grad.clone().detach()
        weight_adapted = meta.weight - self.update_lr * grad
        meta.zero_grad()

        # this is the loss and accuracy before first update
        with torch.no_grad():
            feat_q = net(x_qry, feature_only=True)
            pred_q = meta(feat_q.detach()).squeeze(1)
            # pred_q = torch.sigmoid(pred_q) * 4
            pred_q = self.score_mapper(pred_q)

            res_list.add(0, metricser(pred_q.cpu().numpy(), y_qry.cpu().numpy()))
            # correlation = np.corrcoef(pred_q.cpu().numpy(), y_qry.cpu().numpy())[0][1]
            # correlations[0] = correlations[0] + correlation

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            pred_q = F.linear(feat_q.detach(), weight_adapted).squeeze(1)
            # pred_q = torch.sigmoid(pred_q) * 4
            pred_q = self.score_mapper(pred_q)
            # pdb.set_trace()

            res_list.add(1, metricser(pred_q.cpu().numpy(), y_qry.cpu().numpy()))
            # correlation = np.corrcoef(pred_q.cpu().numpy(), y_qry.cpu().numpy())[0][1]
            # correlations[1] = correlations[1] + correlation

        for k in range(1, self.update_step_test):
            pred = F.linear(feat.detach(), weight_adapted).squeeze(1)
            # pred = torch.sigmoid(pred) * 4
            pred = self.score_mapper(pred)
            loss = F.mse_loss(pred, y_spt)
            grad = torch.autograd.grad(loss, weight_adapted)
            weight_adapted = weight_adapted - self.update_lr * grad[0]

            with torch.no_grad():
                pred_q = F.linear(feat_q.detach(), weight_adapted).squeeze(1)
                # pred_q = torch.sigmoid(pred_q) * 4
                pred_q = self.score_mapper(pred_q)

                res_list.add(k + 1, metricser(pred_q.cpu().numpy(), y_qry.cpu().numpy()))
                # correlation = np.corrcoef(pred_q.cpu().numpy(), y_qry.cpu().numpy())[0][1]
                # correlations[k + 1] = correlations[k + 1] + correlation

        del meta
        # accs_sum = np.array(correlations, dtype=np.float64)
        # self.net.train()
        set_requries_grad(self.net, True)
        # accs_sum = np.array(correlations, dtype=np.float64) / querysz
        return res_list.unpack()



def main():
    pass


if __name__ == '__main__':
    main()
