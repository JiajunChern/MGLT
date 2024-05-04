# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import torch
from einops import rearrange
from torch import nn

from models.clip import ContextDecoder, CLIPTextContextEncoder
from util import box_ops
from models.structures import Boxes, Instances, pairwise_iou
from util.token import tokenize


def random_drop_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    if drop_probability > 0 and len(track_instances) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_idxes]
    return track_instances


class QueryInteractionBase(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


class QueryInteractionModule(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError


class QueryInteractionModulev2(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        self.score_thr = 0.5

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:  # 随机删掉track
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:  # 随机添加track（选择与跟踪框最大iou），表示消失儿
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # torch.bernoulli提取二进制随机数

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)  # 添加的个数
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            active_track_instances = track_instances[active_idxes]
            active_track_instances.obj_idxes[active_track_instances.iou <= 0.5] = -1
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        is_pos = track_instances.scores > self.score_thr
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)  # 选择活的（即有ID的目标，因为之前已经经过score的判断为活的目标分配了ID）
        active_track_instances = self._update_track_embedding(
            active_track_instances)  # 根据update_query_pos的不同（仅对当前帧置信度高的目标更新embedding，有ID的目标可能有当前帧消失，但前几帧存在的目标）
        return active_track_instances


class QueryInteractionModuleGroup(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        self.score_thr = 0.5

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:  # 随机删掉track
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:  # 随机添加track（选择与跟踪框最大iou），表示消失儿
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # torch.bernoulli提取二进制随机数

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)  # 添加的个数
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict, g_size=1) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            num_queries, bs = track_instances.obj_idxes.shape[:2]
            min_prev_target_ind = min(sum((track_instances.obj_idxes.reshape(-1, g_size, bs) >= 0).any(1) | (
                    track_instances.scores.reshape(-1, g_size, bs) > 0.5).any(1)))

            active_track_instances = []
            for i in range(bs):
                topk_proposals = \
                    torch.topk(track_instances.scores.reshape(-1, g_size, bs).min(1)[0][..., i], min_prev_target_ind)[1]
                index_all = torch.full((num_queries // g_size, g_size), False, dtype=torch.bool,
                                       device=topk_proposals.device)
                index_all[topk_proposals] = True
                index_all = index_all.reshape(-1)
                active_track_instances.append(track_instances[index_all])

            active_track_instances = Instances.merge(active_track_instances)

            # active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            # active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            # active_track_instances = track_instances[active_idxes]
            del_idxes = active_track_instances.iou <= 0.5
            del_idxes = del_idxes.reshape(-1, g_size, bs).any(dim=1).view(-1, 1, bs).repeat(1, g_size, 1).view(-1, bs)
            active_track_instances.obj_idxes[del_idxes] = -1
        else:
            assert track_instances.obj_idxes.shape[1] == 1
            active_idxes = track_instances.obj_idxes[:, 0] >= 0
            active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            # active_idxes = active_idxes.reshape(-1, g_size).all(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances = track_instances[active_idxes]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        is_pos = track_instances.scores > self.score_thr
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q, k, value=tgt)[0]  # bacht 
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # if self.update_query_pos:
        #     query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
        #     query_pos = query_pos + self.dropout_pos2(query_pos2)
        #     query_pos = self.norm_pos(query_pos)
        #     track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data, g_size=1) -> Instances:
        active_track_instances = self._select_active_tracks(data, g_size)  # 选择活的（即有ID的目标，因为之前已经经过score的判断为活的目标分配了ID）
        active_track_instances = self._update_track_embedding(
            active_track_instances)  # 根据update_query_pos的不同（仅对当前帧置信度高的目标更新embedding，有ID的目标可能有当前帧消失，但前几帧存在的目标）
        return active_track_instances


class QueryInteractionModuleGroup2(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        self.score_thr = 0.5

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:  # 随机删掉track
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:  # 随机添加track（选择与跟踪框最大iou），表示消失儿
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # torch.bernoulli提取二进制随机数

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)  # 添加的个数
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            active_track_instances = track_instances[active_idxes]
            active_track_instances.obj_idxes[active_track_instances.iou <= 0.5] = -1
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances, g_size=1) -> Instances:
        is_pos = track_instances.scores > self.score_thr
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        attn_mask = torch.full((len(query_pos), len(query_pos)), float('-inf'), dtype=torch.float32,
                               device=query_pos.device)
        group_ids = track_instances.group_ids
        for ith in range(g_size):
            attn_mask1 = torch.full((len(query_pos), len(query_pos)), False, dtype=torch.bool, device=query_pos.device)
            attn_mask2 = torch.full((len(query_pos), len(query_pos)), False, dtype=torch.bool, device=query_pos.device)
            attn_mask1[group_ids == ith] = True
            attn_mask2[:, group_ids == ith] = True
            attn_mask[attn_mask1 & attn_mask2] = 0
        # row_groups = group_ids.view(1, -1).repeat(len(track_instances), 1)
        # # attn_mask[torch.where(row_groups==row_groups.transpose(1, 0))] = 0
        # activate_index = torch.where(row_groups==row_groups.transpose(1, 0))
        # attn_mask = attn_mask.index_put(activate_index, torch.zeros(len(activate_index[0]), dtype=torch.float32, device=query_feat.device))

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None], attn_mask=attn_mask)[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data, g_size=1) -> Instances:
        active_track_instances = self._select_active_tracks(data)  # 选择活的（即有ID的目标，因为之前已经经过score的判断为活的目标分配了ID）
        active_track_instances = self._update_track_embedding(active_track_instances,
                                                              g_size=g_size)  # 根据update_query_pos的不同（仅对当前帧置信度高的目标更新embedding，有ID的目标可能有当前帧消失，但前几帧存在的目标）
        return active_track_instances


class QueryInteractionModuleAE(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        self.score_thr = 0.5

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:  # 随机删掉track
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:  # 随机添加track（选择与跟踪框最大iou），表示消失儿
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # torch.bernoulli提取二进制随机数

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)  # 添加的个数
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = torch.full((len(track_instances),), False, dtype=torch.bool,
                                      device=track_instances.obj_idxes.device)
            active_id = []
            for ith, idx in enumerate(track_instances.obj_idxes.flip(0)):
                if idx > 0 and idx not in active_id:
                    active_id.append(idx)
                    active_idxes[ith] = True
            active_idxes = active_idxes.flip(0)

            active_idxes = (active_idxes) | (track_instances.scores > 0.5)
            active_track_instances = track_instances[active_idxes]
            active_track_instances.obj_idxes[active_track_instances.iou <= 0.5] = -1
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        is_pos = track_instances.scores > self.score_thr
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)  # 选择活的（即有ID的目标，因为之前已经经过score的判断为活的目标分配了ID）
        active_track_instances = self._update_track_embedding(
            active_track_instances)  # 根据update_query_pos的不同（仅对当前帧置信度高的目标更新embedding，有ID的目标可能有当前帧消失，但前几帧存在的目标）
        return active_track_instances


class QueryInteractionModuleSet(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        self.score_thr = 0.5

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:  # 随机删掉track
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:  # 随机添加track（选择与跟踪框最大iou），表示消失儿
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # torch.bernoulli提取二进制随机数

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)  # 添加的个数
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict, g_size=1) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances = track_instances[active_idxes]
            del_idxes = active_track_instances.iou <= 0.5
            del_idxes = del_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances.obj_idxes[del_idxes] = -1
        else:
            active_idxes = track_instances.obj_idxes >= 0
            active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            # active_idxes = active_idxes.reshape(-1, g_size).all(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances = track_instances[active_idxes]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        is_pos = track_instances.scores > self.score_thr
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data, g_size=1) -> Instances:
        active_track_instances = self._select_active_tracks(data, g_size)  # 选择活的（即有ID的目标，因为之前已经经过score的判断为活的目标分配了ID）
        active_track_instances = self._update_track_embedding(
            active_track_instances)  # 根据update_query_pos的不同（仅对当前帧置信度高的目标更新embedding，有ID的目标可能有当前帧消失，但前几帧存在的目标）
        return active_track_instances


class QueryInteractionModulePrompt(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        self.score_thr = 0.5

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:  # 随机删掉track
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:  # 随机添加track（选择与跟踪框最大iou），表示消失儿
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # torch.bernoulli提取二进制随机数

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)  # 添加的个数
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict, g_size=1) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances = track_instances[active_idxes]
            del_idxes = active_track_instances.iou <= 0.5
            del_idxes = del_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances.obj_idxes[del_idxes] = -1
        else:
            active_idxes = track_instances.obj_idxes >= 0
            active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            # active_idxes = active_idxes.reshape(-1, g_size).all(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances = track_instances[active_idxes]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances, g_size) -> Instances:
        active_indxes = track_instances.scores > 0
        # active_num = torch.arange(len(track_instances)).reshape(-1, g_size)
        # active_num = active_num[(torch.arange(len(active_num)), track_instances.scores.reshape(-1, 5).argmin(-1))]
        # active_indxes[active_num] = False
        active_indxes[::g_size] = False

        is_pos = (track_instances.scores > self.score_thr) & active_indxes
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        assert not self.update_query_pos
        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data, g_size=1) -> Instances:
        active_track_instances = self._select_active_tracks(data, g_size)  # 选择活的（即有ID的目标，因为之前已经经过score的判断为活的目标分配了ID）
        active_track_instances = self._update_track_embedding(active_track_instances,
                                                              g_size)  # 根据update_query_pos的不同（仅对当前帧置信度高的目标更新embedding，有ID的目标可能有当前帧消失，但前几帧存在的目标）
        return active_track_instances


class QueryInteractionModulePrompt2(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        self.score_thr = 0.5

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:  # 随机删掉track
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:  # 随机添加track（选择与跟踪框最大iou），表示消失儿
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # torch.bernoulli提取二进制随机数

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)  # 添加的个数
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict, g_size=1) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances = track_instances[active_idxes]
            del_idxes = active_track_instances.iou <= 0.5
            del_idxes = del_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances.obj_idxes[del_idxes] = -1
        else:
            active_idxes = track_instances.obj_idxes >= 0
            active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            # active_idxes = active_idxes.reshape(-1, g_size).all(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances = track_instances[active_idxes]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances, g_size) -> Instances:
        active_indxes = track_instances.scores > 0
        # active_num = torch.arange(len(track_instances)).reshape(-1, g_size)
        # active_num = active_num[(torch.arange(len(active_num)), track_instances.scores.reshape(-1, 5).argmin(-1))]
        # active_indxes[active_num] = False
        active_indxes[::g_size] = False

        is_pos = track_instances.scores > self.score_thr
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]
        is_pos = (track_instances.scores > self.score_thr) & active_indxes

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        assert not self.update_query_pos
        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data, g_size=1) -> Instances:
        active_track_instances = self._select_active_tracks(data, g_size)  # 选择活的（即有ID的目标，因为之前已经经过score的判断为活的目标分配了ID）
        active_track_instances = self._update_track_embedding(active_track_instances,
                                                              g_size)  # 根据update_query_pos的不同（仅对当前帧置信度高的目标更新embedding，有ID的目标可能有当前帧消失，但前几帧存在的目标）
        return active_track_instances


class QueryInteractionModulePrompt3(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        self.score_thr = 0.5

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:  # 随机删掉track
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:  # 随机添加track（选择与跟踪框最大iou），表示消失儿
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # torch.bernoulli提取二进制随机数

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)  # 添加的个数
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict, g_size=1) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances = track_instances[active_idxes]
            del_idxes = active_track_instances.iou <= 0.5
            del_idxes = del_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances.obj_idxes[del_idxes] = -1
        else:
            active_idxes = track_instances.obj_idxes >= 0
            active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            # active_idxes = active_idxes.reshape(-1, g_size).all(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances = track_instances[active_idxes]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances, g_size) -> Instances:
        active_indxes = track_instances.scores > 0
        # active_num = torch.arange(len(track_instances)).reshape(-1, g_size)
        # active_num = active_num[(torch.arange(len(active_num)), track_instances.scores.reshape(-1, 5).argmin(-1))]
        # active_indxes[active_num] = False

        is_pos = track_instances.scores > self.score_thr
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]
        active_indxes[::g_size] = False
        active_indxes[1::g_size] = False
        is_pos = (track_instances.scores > self.score_thr) & active_indxes

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        assert not self.update_query_pos
        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data, g_size=1) -> Instances:
        active_track_instances = self._select_active_tracks(data, g_size)  # 选择活的（即有ID的目标，因为之前已经经过score的判断为活的目标分配了ID）
        active_track_instances = self._update_track_embedding(active_track_instances,
                                                              g_size)  # 根据update_query_pos的不同（仅对当前帧置信度高的目标更新embedding，有ID的目标可能有当前帧消失，但前几帧存在的目标）
        return active_track_instances


class QueryInteractionModuleIOU(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        self.score_thr = 0.5

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:  # 随机删掉track
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:  # 随机添加track（选择与跟踪框最大iou），表示消失儿
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # torch.bernoulli提取二进制随机数

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)  # 添加的个数
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            active_track_instances = track_instances[active_idxes]
            active_track_instances.obj_idxes[active_track_instances.pred_ious[:, 0] < 0] = -1
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        is_pos = track_instances.scores > self.score_thr
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)  # 选择活的（即有ID的目标，因为之前已经经过score的判断为活的目标分配了ID）
        active_track_instances = self._update_track_embedding(
            active_track_instances)  # 根据update_query_pos的不同（仅对当前帧置信度高的目标更新embedding，有ID的目标可能有当前帧消失，但前几帧存在的目标）
        return active_track_instances


class QueryAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = output_dim

        super(QueryAttention, self).__init__()

        self.q_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.input_dim, self.hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim, self.out_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, query, key):
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(key)

        B, L, N = Q.shape

        scaling = float(self.hidden_dim) ** -0.5
        Q = Q * scaling

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        atten_weights = torch.bmm(Q, K.transpose(1, 2))
        attention = self.drop1(torch.softmax(atten_weights, dim=-1))
        attn_ouput = torch.bmm(attention, V).transpose(0, 1).contiguous().view(B, L, N)
        output = self.drop2(self.out_proj(attn_ouput))

        return output


class ClipAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


from models.deformable_detr import MLP


class QueryInteractionModuleGroupCLIP(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        self.score_thr = 0.5

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

        self.context_length = args.clip_token_lengths
        self.context_decoder = ContextDecoder(transformer_width=256, transformer_heads=4, transformer_layers=3,
                                              visual_dim=256, dropout=0.1, outdim=256)
        self.text_encoder = CLIPTextContextEncoder(context_length=self.context_length, transformer_width=512,
                                                   transformer_heads=8,
                                                   transformer_layers=12, embed_dim=1024)
        self.text_encoder.init_weights(pretrained='./weights/RN50.pt')
        self.query_attn = QueryAttention(input_dim=1024, hidden_dim=256, output_dim=256, dropout=0.1)
        self.clip_adapter = ClipAdapter(input_dim=1024, hidden_dim=512, dropout=0.1)
        self.feat_resize = FeatureResizer(input_feat_size=512 + 256, output_feat_size=256, dropout=0.1)
        self.caption_proj = MLP(256, 512, 512, 3)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:  # 随机删掉track
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:  # 随机添加track（选择与跟踪框最大iou），表示消失儿
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # torch.bernoulli提取二进制随机数

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)  # 添加的个数
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict, g_size=1) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            num_queries, bs = track_instances.obj_idxes.shape[:2]
            min_prev_target_ind = min(sum((track_instances.obj_idxes.reshape(-1, g_size, bs) >= 0).any(1) | (
                    track_instances.scores.reshape(-1, g_size, bs) > 0.5).any(1)))

            active_track_instances = []
            for i in range(bs):
                topk_proposals = \
                    torch.topk(track_instances.scores.reshape(-1, g_size, bs).min(1)[0][..., i], min_prev_target_ind)[1]
                index_all = torch.full((num_queries // g_size, g_size), False, dtype=torch.bool,
                                       device=topk_proposals.device)
                index_all[topk_proposals] = True
                index_all = index_all.reshape(-1)
                active_track_instances.append(track_instances[index_all])

            active_track_instances = Instances.merge(active_track_instances)

            # active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            # active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            # active_track_instances = track_instances[active_idxes]
            del_idxes = active_track_instances.iou <= 0.5
            del_idxes = del_idxes.reshape(-1, g_size, bs).any(dim=1).view(-1, 1, bs).repeat(1, g_size, 1).view(-1, bs)
            active_track_instances.obj_idxes[del_idxes] = -1
        else:
            assert track_instances.obj_idxes.shape[1] == 1
            active_idxes = track_instances.obj_idxes[:, 0] >= 0
            active_idxes = active_idxes.reshape(-1, g_size).any(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            # active_idxes = active_idxes.reshape(-1, g_size).all(dim=1).view(-1, 1).repeat(1, g_size).view(-1)
            active_track_instances = track_instances[active_idxes]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances, visual_context) -> Instances:
        is_pos = track_instances.scores > self.score_thr
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q, k, value=tgt)[0]  # batch t
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # if self.update_query_pos:
        #     query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
        #     query_pos = query_pos + self.dropout_pos2(query_pos2)
        #     query_pos = self.norm_pos(query_pos)
        #     track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)

        # TODO: batch size  == 1 !!!!
        query_feat = query_feat.squeeze(1)
        query_context = self.context_decoder(query_feat, visual_context).unsqueeze(1)

        # B, L, K, C = query_context.shape
        # [55,15] [1,14] [1,1,9,512]
        # text_embeddings = self.text_encoder(self.texts, torch.cat([self.contexts.expand(B, -1, K, -1), query_context], dim=1), 'querytext')    # concat a learnable context
        # text_embeddings = self.text_encoder(self.texts, self.caption_proj(query_context.squeeze(1)).unsqueeze(1),
        #                                     'querytext')  # only query context
        # text_embeddings2 = self.text_encoder(self.track_textstrack_texts, None, 'onlytext')     # onlt text context
        query_caption = self.caption_proj(query_context.squeeze(1)).unsqueeze(1)
        text_embeddings, text_embeddings2 = self.text_encoder([self.track_texts, self.texts], query_caption,
                                                              'query_concat_text')  # query concat track text
        adapter_embedding = self.clip_adapter(torch.cat([text_embeddings, text_embeddings2], dim=1))

        query_feat3 = self.query_attn(adapter_embedding[:, :text_embeddings.shape[1], :],
                                      adapter_embedding[:, text_embeddings.shape[1]:, :])
        # query_feat3 = self.query_attn(text_embeddings, text_embeddings2)
        # query_feat3 = self.dropout_feat3(self.activation(self.linear_feat3(text_embeddings)))
        query_feat = self.feat_resize(
            torch.cat([query_feat, query_feat3.squeeze(0), query_context.reshape(*query_feat.shape)], dim=-1))

        query_feat = query_feat.unsqueeze(1)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data, g_size=1) -> Instances:
        active_track_instances = self._select_active_tracks(data, g_size)  # 选择活的（即有ID的目标，因为之前已经经过score的判断为活的目标分配了ID）
        active_track_instances = self._update_track_embedding(
            active_track_instances,
            data['context'])  # 根据update_query_pos的不同（仅对当前帧置信度高的目标更新embedding，有ID的目标可能有当前帧消失，但前几帧存在的目标）
        return active_track_instances

    def tokenize(self, sentences, device):
        self.track_book = sentences
        # self.base_class = [en_core_web_lg_dict[sentence] for sentence in sentences]
        self.base_class = ['']
        self.texts = torch.cat([tokenize(c, context_length=self.context_length - 1) for c in self.base_class]).to(
            device).long()
        self.track_texts = torch.cat(
            [tokenize(c, context_length=self.context_length) for c in self.track_book]).to(device).long()
        return self.track_texts


def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb


def build(args, layer_name, dim_in, hidden_dim, dim_out):
    interaction_layers = {
        # 'QIM': QueryInteractionModule,
        # 'QIMv2': QueryInteractionModulev2,
        # 'GQIM2': QueryInteractionModuleGroup2,
        # 'QIMAE': QueryInteractionModuleAE,
        # 'SQIM': QueryInteractionModuleSet,
        # 'SPQIM': QueryInteractionModulePrompt,
        # 'SPQIM2': QueryInteractionModulePrompt2,
        # 'SPQIM': QueryInteractionModulePrompt3,
        #
        # 'QIMIOU': QueryInteractionModuleIOU,

        'GQIM': QueryInteractionModuleGroup,
        'GQIM_CLIP': QueryInteractionModuleGroupCLIP,
    }
    assert layer_name in interaction_layers, 'invalid query interaction layer: {}'.format(layer_name)
    return interaction_layers[layer_name](args, dim_in, hidden_dim, dim_out)
