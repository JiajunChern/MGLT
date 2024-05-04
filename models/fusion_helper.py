import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from typing import Optional, List

from torch import Tensor

from util.misc import NestedTensor


class PositionEmbeddingSine1D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # [B, C, T]
        mask = tensor_list.mask  # [B, T]
        assert mask is not None
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)  # [B, T]
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # [B, T, C]
        # n,c,t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos_x.permute(0, 2, 1)  # [B, C, T]
        return pos


def agg_lang_feat(features, mask, pool_type="average"):
    """average pooling of language features"""
    # feat: (bs, seq_len, C)
    # mask: (bs, seq_len)
    if pool_type == "average":
        embedded = features * mask.unsqueeze(-1).float()  # use mask to zero out invalid token features
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())
    elif pool_type == "max":
        out = []
        for i in range(len(features)):
            pool_feat, _ = torch.max(features[i][mask[i]], 0)  # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0)  # (bs, C)
    else:
        raise ValueError("pool_type should be average or max")
    return aggregate


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)  # (bs * 8, -1, embed_dim//8)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)  # (bs * 8, seq_len_img, embed_dim//8)
        key_states = key_states.view(*proj_shape)  # (bs * 8, seq_len_text, embed_dim//8)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # (bs * 8, seq_len_img, seq_len_text)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights,
                                       min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights,
                                       max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l,
                                         min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l,
                                         max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)
        # assert attention_mask_l.dtype == torch.int64
        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)  # (bs, seq_len)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, seq_len)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l

    def forward_language(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)  # (bs * 8, -1, embed_dim//8)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)  # (bs * 8, seq_len_img, embed_dim//8)
        key_states = key_states.view(*proj_shape)  # (bs * 8, seq_len_text, embed_dim//8)
        value_v_states = value_v_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # (bs * 8, seq_len_img, seq_len_text)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights,
                                       min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights,
                                       max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l,
                                         min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l,
                                         max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)
        # assert attention_mask_l.dtype == torch.int64
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)
        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_l


class BiAttentionBlockForCheckpoint(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1,
                 drop_path=.0, init_values=1e-4, ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_l=None):
        # v: visual features, (bs, sigma(HW), 256)
        # l: language features, (bs, seq_len, 768)
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l

    def forward_language(self, v, l, attention_mask_l=None):
        # v: visual features, (bs, sigma(HW), 256)
        # l: language features, (bs, seq_len, 768)
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_l = self.attn.forward_language(v, l, attention_mask_l=attention_mask_l)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return l


class VL_Fuse(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(self, img_dim=256, txt_dim=768):
        super(VL_Fuse, self).__init__()

        self.img_dim = img_dim
        self.txt_dim = txt_dim
        # mha params by default
        self.n_head = 8
        self.embed_dim = 2048

        # early fusion module
        # bi-direction (text->image, image->text)
        self.b_attn = BiAttentionBlockForCheckpoint(
            v_dim=self.img_dim,  # 256
            l_dim=self.txt_dim,  # 768
            embed_dim=self.embed_dim,  # 2048
            num_heads=self.n_head,  # 8
            dropout=0.1,
            drop_path=.0,
            init_values=1.0 / 6.0,
        )
        # TODO: save memory
        # self.use_checkpoint = False

    def forward(self, x: dict):
        visual_features = x["visual"]
        language_dict_features = x["language"]
        language_features = language_dict_features['hidden']

        fused_visual_features, fused_language_features = self.b_attn(
            visual_features, language_features, language_dict_features['masks']
        )

        language_dict_features['hidden'] = fused_language_features
        features_dict = {
            "visual": fused_visual_features, "language": language_dict_features
        }
        return features_dict


class VL_Align(nn.Module):
    def __init__(self, img_dim=256, txt_dim=768):
        super().__init__()
        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # dot product soft token head
        self.dot_product_projection_image = nn.Identity()
        self.dot_product_projection_language = nn.Linear(txt_dim, img_dim, bias=True)  # 768 -> 256
        self.log_scale = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(txt_dim), requires_grad=True)  # (768，)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)  # size (1,)

    def forward(self, x, embedding):
        """
        x: visual features (bs, num_query, 256)
        embedding: language features (bs, L, 768)
        """
        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)  # (bs, L, 768) L is maximum sentence length
        dot_product_proj_tokens = self.dot_product_projection_language(embedding / 2.0)  # 768 -> 256
        dot_product_proj_tokens_bias = torch.matmul(
            embedding, self.bias_lang
        ) + self.bias0  # (bs, L, 768) x (768, ) + (1, ) -> (bs, L)

        dot_product_proj_queries = self.dot_product_projection_image(x)  # (bs, num_query, 256)
        A = dot_product_proj_queries.shape[1]  # num_query
        bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)  # (bs, num_query, L)

        dot_product_logit = (torch.matmul(
            dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)
        ) / self.log_scale.exp()) + bias  # (bs, num_query, 256) x (bs, 256, L) -> (bs, num_query, L)
        dot_product_logit = torch.clamp(dot_product_logit, max=50000)
        dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
        return dot_product_logit


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + tgt2
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

        # self.fc1 = nn.Linear(output_feat_size, output_feat_size, bias=True)
        # self.layer_norm1 = nn.LayerNorm(output_feat_size, eps=1e-12)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        # output = self.dropout(x)
        # x = x.mean(1).unsqueeze(1)
        # x = self.fc1(x)
        # if self.do_ln:
        #     x = self.layer_norm1(x)
        return x
