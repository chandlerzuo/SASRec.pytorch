import numpy as np
import torch

from .model import PointWiseFeedForward


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py
# key difference from v2: add dual cross network


class SASRecIHM(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRecIHM, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, args.hidden_units, padding_idx=0
        )
        self.user_emb = torch.nn.Embedding(
            self.user_num + 1, args.hidden_units, padding_idx=0
        )
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.model = args.model
        self.dual_dim = 0
        if args.has_dual:
            self.dual_dim = args.hidden_units
        assert self.model in ["CDN", "IHM", "SASRec"]
        if self.dual_dim > 0:
            self.dual_item_emb = torch.nn.Embedding(
                self.item_num + 1, self.dual_dim, padding_idx=0
            )
            self.dual_user_emb = torch.nn.Embedding(
                self.user_num + 1, self.dual_dim, padding_idx=0
            )
            self.dual_item_fusion = torch.nn.Sequential(
                torch.nn.Linear(
                    self.dual_dim + args.hidden_units, self.dual_dim + args.hidden_units
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dual_dim + args.hidden_units, args.hidden_units),
                torch.nn.ReLU(),
                torch.nn.Linear(args.hidden_units, args.hidden_units),
                torch.nn.ReLU(),
            )
            self.dual_user_fusion = torch.nn.Sequential(
                torch.nn.Linear(
                    self.dual_dim + args.hidden_units, self.dual_dim + args.hidden_units
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dual_dim + args.hidden_units, args.hidden_units),
                torch.nn.ReLU(),
                torch.nn.Linear(args.hidden_units, args.hidden_units),
                torch.nn.ReLU(),
            )
        self.fuse_ihm = args.fuse_ihm
        if args.fuse_ihm:
            self.ihm_fusion = torch.nn.Sequential(
                torch.nn.Linear(2 * args.hidden_units, 2 * args.hidden_units),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * args.hidden_units, args.hidden_units),
                torch.nn.ReLU(),
                torch.nn.Linear(args.hidden_units, args.hidden_units),
                torch.nn.ReLU(),
            )
        self.num_experts = args.num_experts

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
        self.ihm_attn_layer_norm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ihm_attn_layer = torch.nn.MultiheadAttention(
            args.hidden_units, self.num_experts, args.dropout_rate
        )
        self.ihm_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ihm_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
        if self.model != "CDN":
            self.ihm_gating = torch.nn.Sequential(
                torch.nn.Linear(2 * args.hidden_units, 1), torch.nn.Softmax(dim=0)
            )
        else:
            self.ihm_gating = torch.nn.Sequential(
                torch.nn.Linear(1, 1), torch.nn.Sigmoid()
            )
        self.ihm_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

    def log2feats(self, user_ids, log_seqs):
        # log_seq: (B, L)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim**0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)  # (L, B, C)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs  # (L, B, C)
            seqs = torch.transpose(seqs, 0, 1)  # (B, L, C)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)
        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        if self.dual_dim > 0:
            # user_ids: (B, )
            dual_user_embs = (
                self.dual_user_emb(torch.LongTensor(user_ids).to(self.dev))
                .unsqueeze(1)
                .tile((1, log_feats.shape[1], 1))
            )  # (B, T, d_dual)
            log_feats = torch.nn.functional.normalize(
                self.dual_item_fusion(torch.cat([log_feats, dual_user_embs], dim=-1)),
                p=2.0,
                dim=-1,
            )
            log_feats *= ~timeline_mask.unsqueeze(-1)
            log_feats = self.last_layernorm(seqs)
        else:
            dual_user_embs = torch.zeros(1)  # placeholder
        user_embs = self.user_emb(torch.LongTensor(user_ids).to(self.dev))  # (B, C)
        log_feats_clone = torch.nn.functional.normalize(
            log_feats.clone().detach(), p=2.0, dim=-1
        )
        user_embs_delta = user_embs.unsqueeze(1) - log_feats_clone
        user_embs_delta *= ~timeline_mask.unsqueeze(-1)

        return log_feats, dual_user_embs, torch.square(user_embs_delta).mean()

    def ihm_emb(self, i_seq, ih_seq):
        # i_seq: (B, L), ih_seq: (B, L, max_ih_len)
        batch_size = i_seq.shape[0]
        uih_len = ih_seq.shape[1]
        max_ih_len = ih_seq.shape[-1]
        timeline_mask = torch.BoolTensor(i_seq == 0).to(self.dev)  # (B, L)
        ih_mask = (
            torch.BoolTensor(ih_seq == 0).to(self.dev).view(batch_size * uih_len, -1)
        )
        ih_mask = torch.concat(
            [ih_mask, torch.zeros(batch_size * uih_len, 1, device=self.dev).bool()],
            dim=1,
        )
        item_embs_raw = self.item_emb(i_seq)  # (B, L, C)

        if self.dual_dim > 0:
            dual_item_embs = self.dual_item_emb(i_seq)  # (B, L, dual_dim)
            item_embs_raw = torch.nn.functional.normalize(
                self.dual_item_fusion(
                    torch.cat([item_embs_raw, dual_item_embs], dim=-1)
                ),  # (B, L, C+dual_dim)
                p=2.0,
                dim=-1,
            )
            dual_item_embs *= ~timeline_mask.unsqueeze(-1)
        else:
            dual_item_embs = torch.zeros(1)  # placeholder
        item_embs = item_embs_raw.clone().detach()
        ih_embs = self.user_emb(ih_seq)  # (B, L, max_ih_len, C)
        item_embs *= self.item_emb.embedding_dim**0.5
        # Q = self.ihm_attn_layer_norm(item_embs).view(1, batch_size * uih_len, -1) # (1, B*L, C)
        ih_embs = torch.concat(
            [ih_embs, item_embs.unsqueeze(2)], dim=2
        )  # (B, L, max_ih_len+1, C)
        ih_embs *= self.item_emb.embedding_dim**0.5
        V = ih_embs.permute(2, 0, 1, 3).view(
            max_ih_len + 1, batch_size * uih_len, -1
        )  # (max_ih_len+1, B*L, C)
        # mha_output, _ = self.ihm_attn_layer(Q, V, V, key_padding_mask=ih_mask) # (1, B*L, C)
        _, attn_weights = self.ihm_attn_layer(
            V,
            V,
            V,
            key_padding_mask=ih_mask,
            need_weights=True,
            average_attn_weights=False,
        )  # (B*L, num_heads, max_ih_len+1, max_ih_len+1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn_weights[:, :, -1, :].permute(
            1, 2, 0
        )  # (num_heads, max_ih_len+1, B*L)
        ihm_embs = (attn_weights.unsqueeze(-1) * V.unsqueeze(0)).sum(
            dim=1
        )  # (num_heads, B*L, C)
        ihm_embs = ihm_embs.reshape(self.num_experts, batch_size, uih_len, -1)
        ihm_embs = self.ihm_fwd_layer(
            self.ihm_fwd_layernorm(ihm_embs).view(
                self.num_experts * batch_size, uih_len, -1
            )
        ).reshape(
            self.num_experts, batch_size, uih_len, -1
        )  # (num_heads, B, L, C)

        ihm_gating_weights = self.ihm_gating(
            torch.cat(
                [torch.tile(item_embs, (self.num_experts, 1, 1, 1)), ihm_embs],
                dim=3,
            )  # (num_experts, B, L, 2*C)
        )  # (num_experts, B, L, 1)
        final_embs = (ihm_gating_weights * ihm_embs).sum(dim=0)  # (B, L, C)

        if self.fuse_ihm:
            final_embs = torch.nn.functional.normalize(
                self.ihm_fusion(torch.cat([final_embs, item_embs_raw], dim=2)),
                p=2.0,
                dim=-1,
            )
        final_embs *= ~timeline_mask.unsqueeze(-1)

        item_distillation_loss = torch.nn.functional.mse_loss(
            final_embs, item_embs_raw.clone().detach()
        )

        return final_embs, item_embs_raw, dual_item_embs, item_distillation_loss

    def cdn_emb(self, i_seq, ih_seq):
        # i_seq: (B, L), ih_seq: (B, L, max_ih_len)
        timeline_mask = torch.BoolTensor(i_seq == 0).to(self.dev)  # (B, L)
        ih_mask = torch.BoolTensor(ih_seq == 0).to(self.dev)  # (B, L, max_ih_len)
        item_embs_raw = self.item_emb(i_seq)  # (B, L, C)
        if self.dual_dim > 0:
            dual_item_embs = self.dual_item_emb(i_seq)  # (B, L, dual_dim)
            item_embs_raw = torch.nn.functional.normalize(
                self.dual_item_fusion(
                    torch.cat([item_embs_raw, dual_item_embs], dim=-1)
                ),  # (B, L, C+dual_dim)
                p=2.0,
                dim=-1,
            )
            dual_item_embs *= ~timeline_mask.unsqueeze(-1)
        else:
            dual_item_embs = torch.zeros(1)  # placeholder
        item_embs_raw *= ~timeline_mask.unsqueeze(-1)
        item_embs = item_embs_raw.clone().detach()
        ihm_embs = self.user_emb(ih_seq)  # (B, L, max_ih_len, C)
        ihm_embs[ih_mask] = 0
        ihm_embs = torch.nn.functional.normalize(
            ihm_embs.sum(dim=2), p=2, dim=-1
        )  # (B, L, C)

        ihm_gating_weights = self.ihm_gating(
            ih_mask.to(torch.float32).sum(dim=2).unsqueeze(-1)
        )  # (B, L, 1)
        final_embs = (
            ihm_gating_weights * item_embs + (1 - ihm_gating_weights) * ihm_embs
        )  # (B, L, C)

        final_embs *= ~timeline_mask.unsqueeze(-1)

        item_distillation_loss = torch.nn.functional.mse_loss(
            final_embs, item_embs_raw.clone().detach()
        )
        return final_embs, item_embs_raw, dual_item_embs, item_distillation_loss

    def forward(
        self, user_ids, log_seqs, pos_seqs, neg_seqs, pos_ih_seqs, neg_ih_seqs
    ):  # for training
        log_feats, user_dual, user_emb_loss = self.log2feats(user_ids, log_seqs)

        if self.model != "CDN":
            pos_embs, pos_raw_embs, pos_item_dual, item_distill_loss = self.ihm_emb(
                torch.LongTensor(pos_seqs).to(self.dev),
                torch.LongTensor(pos_ih_seqs).to(self.dev),
            )
            neg_embs, neg_raw_embs, _, _ = self.ihm_emb(
                torch.LongTensor(neg_seqs).to(self.dev),
                torch.LongTensor(neg_ih_seqs).to(self.dev),
            )
        else:
            pos_embs, pos_raw_embs, pos_item_dual, item_distill_loss = self.cdn_emb(
                torch.LongTensor(pos_seqs).to(self.dev),
                torch.LongTensor(pos_ih_seqs).to(self.dev),
            )
            neg_embs, neg_raw_embs, _, _ = self.cdn_emb(
                torch.LongTensor(neg_seqs).to(self.dev),
                torch.LongTensor(neg_ih_seqs).to(self.dev),
            )
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_logits_raw = (log_feats * pos_raw_embs).sum(dim=-1)
        neg_logits_raw = (log_feats * neg_raw_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        # dual embedding loss
        if self.dual_dim > 0:
            log_feats_clone = torch.nn.functional.normalize(
                log_feats.clone().detach(), p=2.0, dim=-1
            )
            item_embs_clone = torch.nn.functional.normalize(
                pos_raw_embs.clone().detach(), p=2.0, dim=-1
            )
            dual_loss = torch.nn.functional.mse_loss(
                pos_item_dual, log_feats_clone
            ) + torch.nn.functional.mse_loss(user_dual, item_embs_clone)
        else:
            dual_loss = torch.zeros(1)  # placeholder
        return (
            pos_logits,
            neg_logits,
            pos_logits_raw,
            neg_logits_raw,
            dual_loss,
            user_emb_loss,
            item_distill_loss,
        )

    def predict(self, user_ids, log_seqs, item_indices, ih_seqs):  # for inference
        log_feats, _, _ = self.log2feats(user_ids, log_seqs)

        final_feat = log_feats[
            :, -1, :
        ]  # only use last QKV classifier, a waste, (U, C)

        if self.model != "CDN":
            ihm_embs, sasrec_embs, _, _ = self.ihm_emb(
                torch.LongTensor(item_indices).unsqueeze(0).to(self.dev),
                torch.LongTensor(ih_seqs).unsqueeze(0).to(self.dev),
            )  # (U, I, C)
        else:
            ihm_embs, sasrec_embs, _, _ = self.cdn_emb(
                torch.LongTensor(item_indices).unsqueeze(0).to(self.dev),
                torch.LongTensor(ih_seqs).unsqueeze(0).to(self.dev),
            )  # (U, I, C)
        if self.model == "SASRec":
            item_embs = sasrec_embs
        else:
            item_embs = ihm_embs
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)
