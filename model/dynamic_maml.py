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

REBIRTH = 'rebirth'
TUNING = 'tuning'
available_dymodes = [REBIRTH, TUNING]


class DynamicLearner(nn.Module):
    """
    Meta + MAML
    """
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=1, bias=True):
        super().__init__()
        self.bias = bias
        # dynamic Linear layer
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        # parameter generator
        self.generator = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True),
                                       # nn.BatchNorm1d(hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, in_dim * out_dim + (out_dim if bias else 0), bias=True))
        # initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

    def gen_forward(self, x, param_dict, mode=REBIRTH):
        """
        internal update
        :param x: output of feature extractor
        :param param_dict: parameter of K-th update
        :param mode: rebirth or tuning
        :return: output: prediction with temporary parameter
        """
        bz = x.size(0)
        x = x.view(bz, -1)
        params = x
        for i, module in enumerate(self.generator):
            if isinstance(module, nn.Linear):
                # print(i, 'Linear')
                weight_name = f'{i}.weight'
                bias_name = f'{i}.bias'
                bias_data = param_dict[bias_name] if bias_name in param_dict else None
                params = F.linear(params, param_dict[weight_name], bias_data)
            elif isinstance(module, nn.ReLU):
                # print(i, 'ReLU')
                params = F.relu(params, inplace=True)
            else:
                # print(i, type(module))
                raise NotImplementedError(f'The module({type(module)}) is not supported!')
                pass
        out = self.ma_forward(x, params, mode=mode)
        return out

    def ma_forward(self, x, params,  mode=REBIRTH):
        """
        use the dynamic parameter to instead of the original parameter of the linear layer to calculation the prediction
        :param params: output of parameter generator
        :param mode: tuning or rebirth
        :return: output: prediction with every image in a batch
        """
        bz = x.size(0)
        x = x.view(bz, -1)
        weight_shape = [bz, self.fc.weight.shape[1], 1]
        bias_shape = [bz, self.fc.weight.shape[0]]
        weight_num = self.fc.weight.numel()
        bias_num = self.fc.bias.numel() if self.bias else None
        if mode == REBIRTH:
            weight_data = 0.01 * params[:, :weight_num].reshape(weight_shape)
            bias_data = (0.01 * params[:, weight_num:].reshape(bias_shape)) if bias_num is not None else None
        elif mode == TUNING:
            weight_data = 0.01 * params[:, :weight_num].reshape(weight_shape) + self.fc.weight.view(-1, 1)
            bias_data = (0.01 * params[:, weight_num:].reshape(bias_shape) + self.fc.bias.view(-1, 1)) if bias_num is not None else None
        else:
            raise NotImplementedError(f'mode={mode} is not supported!')
        output = torch.bmm(x.unsqueeze(1), weight_data).squeeze(2) + (bias_data if bias_data is not None else 0)
        # output = torch.matmul(x.unsqueeze(1), weight_data).squeeze(2) + (bias_data if bias_data is not None else 0)
        return output

    def forward(self, x, mode=REBIRTH):
        bz = x.size(0)
        x = x.view(bz, -1)
        params = self.generator(x)
        return self.ma_forward(x, params, mode=mode)

class DynamicMAML(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args):
        """

        :param args:
        """
        super(DynamicMAML, self).__init__()
        self.update_lr = args.update_lr
        self.fc_lr = args.meta_lr
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
        # self.fc = dynamic_class.DynamicClass(512, 1, bias=True).cuda()
        self.meta = DynamicLearner(in_dim=self.net.fea_dim, out_dim=1, bias=True).cuda()
        self.optimizer = torch.optim.Adam(self.meta.parameters(), lr=self.fc_lr)
        self.score_mapper = SigmoidMapper(args.n_way - 1)
        # nn.init.kaiming_normal_(self.meta.fc.weight.data)
        # nn.init.constant_(self.meta.fc.bias.data, 0)

    def meta_train(self, x_spt, y_spt, x_qry, y_qry, mode=REBIRTH):
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
            pred = self.meta(feat.detach(), mode=mode).squeeze(1)
            # pdb.set_trace()
            # pred = torch.sigmoid(pred) * 4
            pred = self.score_mapper(pred)
            loss = F.mse_loss(pred, y_spt[i])

            # Adaption
            loss.backward()
            grad_dict = dict()
            weight_dict = dict()
            with torch.no_grad():
                # Only adapt the meta layer
                # grad = self.meta.weight.grad.clone().detach()
                for name, param in self.meta.generator.named_parameters():
                    grad_dict[name] = param.grad.clone().detach()
            # weight_adapted = self.meta.weight - self.update_lr * grad
            for name, param in self.meta.generator.named_parameters():
                weight_dict[name] = param - self.update_lr * grad_dict[name]
            self.meta.generator.zero_grad()

            # this is the loss and accuracy before first update
            with torch.no_grad():
                feat_q = self.net(x_qry[i], feature_only=True)
                pred_q = self.meta(feat_q.detach(), mode=mode).squeeze(1)
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
                # pred_q = F.linear(feat_q.detach(), weight_adapted).squeeze(1)
                pred_q = self.meta.gen_forward(feat_q.detach(), weight_dict, mode=mode).squeeze(1)
                # pred_q = torch.sigmoid(pred_q) * 4
                pred_q = self.score_mapper(pred_q)
                loss_q = F.mse_loss(pred_q, y_qry[i])
                losses_q[1] += loss_q

                correlation = np.corrcoef(pred_q.cpu().numpy(), y_qry[i].cpu().numpy())[0][1]
                correlations[1] = correlations[1] + correlation

            for k in range(1, self.update_step):
                # run the i-th task and compute loss for k=1~K-1
                # pred = F.linear(feat.detach(), weight_adapted).squeeze(1)
                pred = self.meta.gen_forward(feat.detach(), weight_dict, mode=mode).squeeze(1)
                # pred = torch.sigmoid(pred) * 4
                pred = self.score_mapper(pred)

                loss = F.mse_loss(pred, y_spt[i])
                # grad = torch.autograd.grad(loss, weight_adapted)
                for name, param in self.meta.generator.named_parameters():
                    grad_dict[name] = torch.autograd.grad(loss, weight_dict[name], retain_graph=True)[0]
                # weight_adapted = weight_adapted - self.update_lr * grad[0]
                for name, param in self.meta.generator.named_parameters():
                    weight_dict[name] = weight_dict[name] - self.update_lr * grad_dict[name]

                # pred_q = F.linear(feat_q.detach(), weight_adapted).squeeze(1)
                pred_q = self.meta.gen_forward(feat_q.detach(), weight_dict, mode=mode).squeeze(1)
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

    def meta_test(self, x_spt, y_spt, x_qry, y_qry, mode=REBIRTH):
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
        meta = DynamicLearner(in_dim=self.net.fea_dim, out_dim=1, bias=True).cuda()
        for p_new, p_model in zip(meta.parameters(), self.meta.parameters()):
            device = p_new.device
            p_model_ = p_model.detach().to(device)
            p_new.detach().copy_(p_model_)

        feat = net(x_spt, feature_only=True)
        pred = meta(feat.detach(), mode=mode).squeeze(1)
        # pred = torch.sigmoid(pred) * 4
        pred = self.score_mapper(pred)
        loss = F.mse_loss(pred, y_spt)

        # Adaption
        loss.backward()
        grad_dict = dict()
        weight_dict = dict()

        with torch.no_grad():
            # grad = meta.weight.grad.clone().detach()
            for name, param in meta.generator.named_parameters():
                grad_dict[name] = param.grad.clone().detach()
        # weight_adapted = meta.weight - self.update_lr * grad
        for name, param in meta.generator.named_parameters():
            weight_dict[name] = param - self.update_lr * grad_dict[name]
        meta.generator.zero_grad()

        # this is the loss and accuracy before first update
        with torch.no_grad():
            feat_q = net(x_qry, feature_only=True)
            pred_q = meta(feat_q.detach(), mode=mode).squeeze(1)
            # pred_q = torch.sigmoid(pred_q) * 4
            pred_q = self.score_mapper(pred_q)

            res_list.add(0, metricser(pred_q.cpu().numpy(), y_qry.cpu().numpy()))
            # correlation = np.corrcoef(pred_q.cpu().numpy(), y_qry.cpu().numpy())[0][1]
            # correlations[0] = correlations[0] + correlation

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # pred_q = F.linear(feat_q.detach(), weight_adapted).squeeze(1)
            pred_q = meta.gen_forward(feat_q.detach(), weight_dict, mode=mode).squeeze(1)
            # pred_q = torch.sigmoid(pred_q) * 4
            pred_q = self.score_mapper(pred_q)
            # pdb.set_trace()

            res_list.add(1, metricser(pred_q.cpu().numpy(), y_qry.cpu().numpy()))
            # correlation = np.corrcoef(pred_q.cpu().numpy(), y_qry.cpu().numpy())[0][1]
            # correlations[1] = correlations[1] + correlation

        for k in range(1, self.update_step_test):
            # pred = F.linear(feat.detach(), weight_adapted).squeeze(1)
            pred = meta.gen_forward(feat.detach(), weight_dict, mode=mode).squeeze(1)
            # pred = torch.sigmoid(pred) * 4
            pred = self.score_mapper(pred)
            loss = F.mse_loss(pred, y_spt)
            # grad = torch.autograd.grad(loss, weight_adapted)
            for name, param in meta.generator.named_parameters():
                grad_dict[name] = torch.autograd.grad(loss, weight_dict[name], retain_graph=True)[0]
            # weight_adapted = weight_adapted - self.update_lr * grad[0]
            for name, param in meta.generator.named_parameters():
                weight_dict[name] = weight_dict[name] - self.update_lr * grad_dict[name]

            with torch.no_grad():
                # pred_q = F.linear(feat_q.detach(), weight_adapted).squeeze(1)
                pred_q = meta.gen_forward(feat_q.detach(), weight_dict, mode=mode).squeeze(1)
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
