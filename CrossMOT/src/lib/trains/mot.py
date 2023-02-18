from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.baseline = self.opt.baseline
        self.baseline_view = self.opt.baseline_view
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.view_nID = opt.view_nID
        if self.baseline == 0:
            self.classifier = nn.Linear(int(self.emb_dim), self.nID)
            self.view_classifier = nn.Linear(int(self.emb_dim), self.view_nID)
        else:
            if self.baseline_view == 0:
                self.view_classifier = nn.Linear(int(self.emb_dim), self.view_nID)
            else:
                self.classifier = nn.Linear(int(self.emb_dim), self.nID)
        # if opt.id_loss == 'focal':
        #     torch.nn.init.normal_(self.classifier.weight, std=0.01)
        #     prior_prob = 0.01
        #     bias_value = -math.log((1 - prior_prob) / prior_prob)
        #     torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.zero = torch.tensor(0.).cuda()
        self.single_loss_array = opt.single_loss_array
        self.single_view_id_split_loss = opt.single_view_id_split_loss
        self.cross_loss_array = opt.cross_loss_array
        self.cross_view_id_split_loss = opt.cross_view_id_split_loss

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                if self.baseline == 0:
                    id_head = _tranpose_and_gather_feat(output['cross_view_id'], batch['ind'])
                    id_head = id_head[batch['reg_mask'] > 0].contiguous()
                    id_head = self.emb_scale * F.normalize(id_head)
                    id_target = batch['ids'][batch['reg_mask'] > 0]
                    id_output = self.classifier(id_head).contiguous()
                    single_view_id_head = _tranpose_and_gather_feat(output['single_view_id'], batch['ind'])
                    single_view_id_head = single_view_id_head[batch['reg_mask'] > 0].contiguous()
                    single_view_id_head = self.emb_scale * F.normalize(single_view_id_head)
                    single_view_id_target = batch['single_view_ids'][batch['reg_mask'] > 0]
                    single_view_id_output = self.view_classifier(single_view_id_head).contiguous()
                else:
                    if self.baseline_view == 0:
                        single_view_id_head = _tranpose_and_gather_feat(output['single_view_id'], batch['ind'])
                        single_view_id_head = single_view_id_head[batch['reg_mask'] > 0].contiguous()
                        single_view_id_head = self.emb_scale * F.normalize(single_view_id_head)
                        single_view_id_target = batch['single_view_ids'][batch['reg_mask'] > 0]
                        single_view_id_output = self.view_classifier(single_view_id_head).contiguous()
                    else:
                        id_head = _tranpose_and_gather_feat(output['cross_view_id'], batch['ind'])
                        id_head = id_head[batch['reg_mask'] > 0].contiguous()
                        id_head = self.emb_scale * F.normalize(id_head)
                        id_target = batch['ids'][batch['reg_mask'] > 0]
                        id_output = self.classifier(id_head).contiguous()

                if self.opt.id_loss == 'focal':
                    id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1,
                                                                                                  id_target.long().view(
                                                                                                      -1, 1), 1)
                    id_loss += sigmoid_focal_loss_jit(id_output, id_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="sum"
                                                      ) / id_output.size(0)
                else:
                    if self.baseline == 0:
                        if len(single_view_id_target) > 0:
                            if self.single_view_id_split_loss:
                                single_id_loss = 0
                                single_loop_target = single_view_id_target
                                single_loop_output = single_view_id_output
                                single_view_num = 0
                                while len(single_loop_target) > 0:
                                    single_view_num += 1
                                    sample_id = single_loop_target[0]
                                    small_id, big_id = 0, 0
                                    for i in range(len(self.single_loss_array)):
                                        if sample_id > self.single_loss_array[i]:
                                            continue
                                        else:
                                            small_id = self.single_loss_array[i-1] if i != 0 else 0
                                            big_id = self.single_loss_array[i]
                                            break
                                    temp_single_output = single_loop_output
                                    temp_single_target = single_loop_target
                                    for i in range(len(single_loop_target)):
                                        if (single_loop_target[i] <= small_id or single_loop_target[i] > big_id):
                                            temp_single_output = single_loop_output[:i].clone()
                                            temp_single_target = single_loop_target[:i].clone()
                                            single_loop_target = single_loop_target[i:].clone()
                                            single_loop_output = single_loop_output[i:].clone()
                                            break
                                        else:
                                            if i == len(single_loop_target) - 1:
                                                temp_single_output = single_loop_output
                                                temp_single_target = single_loop_target
                                                single_loop_target = []
                                    temp_single_output = temp_single_output[:,small_id:big_id].clone()
                                    temp_single_target[::] = temp_single_target[::].clone() - (small_id + 1)
                                    single_id_loss += self.IDLoss(temp_single_output, temp_single_target)
                                single_id_loss /= single_view_num
                            else:
                                single_id_loss = self.IDLoss(single_view_id_output, single_view_id_target)
                        else:
                            single_id_loss = self.zero
    
                        if len(id_target) > 0:
                            if self.cross_view_id_split_loss:
                                cross_id_loss = 0
                                cross_loop_target = id_target
                                cross_loop_output = id_output
                                cross_view_num = 0
                                while len(cross_loop_target) > 0:
                                    cross_view_num += 1
                                    sample_id = cross_loop_target[0]
                                    small_id, big_id = 0, 0
                                    for i in range(len(self.cross_loss_array)):
                                        if sample_id > self.cross_loss_array[i]:
                                            continue
                                        else:
                                            small_id = self.cross_loss_array[i-1]
                                            big_id = self.cross_loss_array[i]
                                            break
                                    temp_cross_output = cross_loop_output
                                    temp_cross_target = cross_loop_target
                                    for i in range(len(cross_loop_target)):
                                        if (cross_loop_target[i] <= small_id or cross_loop_target[i] > big_id):
                                            temp_cross_output = cross_loop_output[:i].clone()
                                            temp_cross_target = cross_loop_target[:i].clone()
                                            cross_loop_target = cross_loop_target[i:].clone()
                                            cross_loop_output = cross_loop_output[i:].clone()
                                            break
                                        else:
                                            if i == len(cross_loop_target)-1:
                                                temp_cross_output = cross_loop_output
                                                temp_cross_target = cross_loop_target
                                                cross_loop_target = []
                                    temp_cross_output = temp_cross_output[:,small_id:big_id].clone()
                                    temp_cross_target[::] = temp_cross_target[::].clone() - (small_id + 1)
                                    cross_id_loss += self.IDLoss(temp_cross_output, temp_cross_target)
                                cross_id_loss /= cross_view_num
                            else:
                                cross_id_loss = self.IDLoss(id_output, id_target)
                        else:
                            cross_id_loss = self.zero

                        id_loss = single_id_loss + cross_id_loss
                    else:
                        if self.baseline_view == 0:
                            single_id_loss = self.IDLoss(single_view_id_output, single_view_id_target)
                            id_loss = single_id_loss
                        else:
                            cross_id_loss = self.IDLoss(id_output, id_target)
                            id_loss = cross_id_loss

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss if id_loss != 0. else self.zero
        if opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss + 0.1 * id_loss

        if self.baseline == 0:
            loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                        'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss, 'single_id_loss': single_id_loss, 'cross_id_loss': cross_id_loss}
        else:
            if self.baseline_view == 0:
                loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss, 'single_id_loss': id_loss}
            else: 
                loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss, 'cross_id_loss': id_loss}   
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        if opt.baseline == 0:
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss', 'single_id_loss', 'cross_id_loss']
        else:
            if opt.baseline_view == 0:
                loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'single_id_loss']
            else:
                loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'cross_id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
