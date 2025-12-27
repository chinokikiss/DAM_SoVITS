import torch
import random
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

from AR.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from AR.modules.transformer import LayerNorm, TransformerEncoder, TransformerEncoderLayer


class Text2SemanticEncoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticEncoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.audio_vocab_size = config["model"]["audio_vocab_size"]
        self.output_vocab_size = config["model"]["output_vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]
        self.SEP = config["model"]["SEP"]
        self.MASK = config["model"]["MASK"]
        self.CLS = config["model"]["CLS"]
        self.min_mask_ratio = config["train"]["min_mask_ratio"]
        self.max_mask_ratio = config["train"]["max_mask_ratio"]
        self.duration_loss_weight = config["train"]["duration_loss_weight"]

        self.emo_proj = nn.Linear(768, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim,
            self.phoneme_vocab_size,
            self.p_dropout,
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim,
            self.audio_vocab_size,
            self.p_dropout,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.type_embedding = nn.Embedding(3, self.embedding_dim)

        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )

        self.ar_predict_layer = nn.Linear(self.model_dim, self.output_vocab_size, bias=False)

        self.duration_predictor = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(self.model_dim // 2, self.model_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(self.model_dim // 4, 1),
        )

        self.ar_accuracy_metric = MulticlassAccuracy(
            self.output_vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )

    
    def forward(self, x, x1_lens, x2_lens, y, y1_lens, y2_lens, emo_feature):
        # x1 sep x2 pad emo y1 cls y2 eos pad

        B = x.size(0)
        x_len = x.size(1)
        y_len = y.size(1)
        device = x.device
        src_len = x_len+1+y_len
        x_lens = x1_lens+1+x2_lens
        y_lens = y1_lens+1+y2_lens
        prompt_lens = x_len + 1 + y1_lens

        mask_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)

        indices = torch.arange(y.size(1), device=device).unsqueeze(0)
        valid_mask = (indices >= y1_lens.unsqueeze(1)+1) & (indices < y_lens.unsqueeze(1)-1)
        y_mask = torch.bernoulli(torch.full(valid_mask.shape, mask_ratio, device=device) * valid_mask).bool()

        # 确保至少有一个被mask
        no_mask_rows = ~y_mask.any(dim=1)
        if no_mask_rows.any():
            for i in torch.where(no_mask_rows)[0]:
                valid_indices = torch.where(valid_mask[i])[0]
                if len(valid_indices) > 0:
                    random_idx = valid_indices[torch.randint(len(valid_indices), (1,))]
                    y_mask[i, random_idx] = True

        y_target = y.clone()

        # 准备输入
        x_emb = self.ar_text_embedding(x) + self.type_embedding(torch.tensor([0], device=device))
        x_pos = self.ar_text_position(x_emb)

        emo = self.emo_proj(emo_feature) + self.type_embedding(torch.tensor([1], device=device))
        emo = emo.unsqueeze(1)

        y[y_mask] = self.MASK
        y_emb = self.ar_audio_embedding(y) + self.type_embedding(torch.tensor([2], device=device))
        y_pos = self.ar_audio_position(y_emb)

        h_input = torch.concat([x_pos, emo, y_pos], dim=1)

        ## 生成掩码
        # x1和x2只能关注自身
        x_attn_mask = torch.ones((B, x_len, src_len), dtype=torch.bool, device=device)
        indices = torch.arange(x_len, device=device).unsqueeze(0)
        mask = (indices < x1_lens.unsqueeze(1))
        indices = torch.arange(src_len, device=device).unsqueeze(0)
        x1_attn_mask = (indices < x1_lens.unsqueeze(1))
        x1_attn_mask = x1_attn_mask.unsqueeze(1).expand(-1, x_len, -1)
        x2_attn_mask = (indices >= x1_lens.unsqueeze(1)) & (indices < x_lens.unsqueeze(1))
        x2_attn_mask = x2_attn_mask.unsqueeze(1).expand(-1, x_len, -1)
        x_attn_mask[mask, :] = ~x1_attn_mask[mask, :]
        x_attn_mask[~mask, :] = ~x2_attn_mask[~mask, :]

        # emo只能关注自身
        emo_attn_mask = torch.ones((B, 1, src_len), dtype=torch.bool, device=device)
        emo_attn_mask[:, :, x_len] = 0

        y_attn_mask = torch.ones((B, y_len, src_len), dtype=torch.bool, device=device)

        # y1能关注x1以及自身
        indices = torch.arange(y_len, device=device).unsqueeze(0)
        mask = (indices < y1_lens.unsqueeze(1))
        indices = torch.arange(src_len, device=device).unsqueeze(0)
        y1_attn_mask = (indices < x1_lens.unsqueeze(1)) | (indices > x_len) & (indices <= x_len + y1_lens.unsqueeze(1))
        y1_attn_mask = y1_attn_mask.unsqueeze(1).expand(-1, y_len, -1)
        y_attn_mask[mask] = ~y1_attn_mask[mask]

        # cls能关注除y2以外的token
        indices = torch.arange(y_len, device=device).unsqueeze(0)
        mask = (indices == y1_lens.unsqueeze(1))
        indices = torch.arange(src_len, device=device).unsqueeze(0)
        cls_attn_mask = (indices < x_lens.unsqueeze(1)) | (indices == x_len) | (indices > x_len) & (indices <= x_len + y1_lens.unsqueeze(1) + 1)
        cls_attn_mask = cls_attn_mask.unsqueeze(1).expand(-1, y_len, -1)
        y_attn_mask[mask] = ~cls_attn_mask[mask]

        # y2能关注除mask token以外的token
        indices = torch.arange(y_len, device=device).unsqueeze(0)
        mask = (indices > y1_lens.unsqueeze(1)) & (indices < y_lens.unsqueeze(1))
        indices = torch.arange(src_len, device=device).unsqueeze(0)
        y2_attn_mask = (indices < x_lens.unsqueeze(1)) | (indices == x_len) | (indices > x_len) & (indices <= x_len + y_lens.unsqueeze(1))
        y2_attn_mask[:, x_len+1:] &= ~y_mask
        y2_attn_mask = y2_attn_mask.unsqueeze(1).expand(-1, y_len, -1)
        y_attn_mask[mask] = ~y2_attn_mask[mask]


        attn_mask = torch.concat([x_attn_mask, emo_attn_mask, y_attn_mask], dim=1)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1).reshape(B * self.num_head, src_len, src_len)
        new_attn_mask = torch.zeros_like(attn_mask, dtype=emo_feature.dtype, device=device)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

        enc, _ = self.h(
            (h_input, None),
            mask=attn_mask,
        )

        cls_hidden = enc[torch.arange(B, device=device), prompt_lens]
        predicted_duration = self.duration_predictor(cls_hidden).squeeze(-1) 
        target_duration = y2_lens.float() / x2_lens.float()
        duration_loss = F.huber_loss(predicted_duration, target_duration, reduction="mean", delta=1.0)

        logits = self.ar_predict_layer(enc[:, x_len+1:][y_mask])
        y_target = y_target[y_mask]
        token_loss = F.cross_entropy(logits, y_target, reduction="mean")
        acc = self.ar_accuracy_metric(logits.detach(), y_target).item()

        total_loss = token_loss + self.duration_loss_weight * duration_loss
        return total_loss, acc
    
    def test(self, x, x1_lens, x2_lens, y, y1_lens, y2_lens, emo_feature):
        # x1 sep x2 pad emo y1 cls y2 eos pad

        B = x.size(0)
        x_len = x.size(1)
        y_len = y.size(1)
        device = x.device
        src_len = x_len+1+y_len
        x_lens = x1_lens+1+x2_lens
        y_lens = y1_lens+1+y2_lens
        prompt_lens = x_len + 1 + y1_lens

        mask_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)

        indices = torch.arange(y.size(1), device=device).unsqueeze(0)
        valid_mask = (indices >= y1_lens.unsqueeze(1)+1) & (indices < y_lens.unsqueeze(1)-1)
        y_mask = torch.bernoulli(torch.full(valid_mask.shape, mask_ratio, device=device) * valid_mask).bool()

        # 确保至少有一个被mask
        no_mask_rows = ~y_mask.any(dim=1)
        if no_mask_rows.any():
            for i in torch.where(no_mask_rows)[0]:
                valid_indices = torch.where(valid_mask[i])[0]
                if len(valid_indices) > 0:
                    random_idx = valid_indices[torch.randint(len(valid_indices), (1,))]
                    y_mask[i, random_idx] = True

        y_target = y.clone()

        # 准备输入
        x_emb = self.ar_text_embedding(x) + self.type_embedding(torch.tensor([0], device=device))
        x_pos = self.ar_text_position(x_emb)

        emo = self.emo_proj(emo_feature) + self.type_embedding(torch.tensor([1], device=device))
        emo = emo.unsqueeze(1)

        y[y_mask] = self.MASK
        y_emb = self.ar_audio_embedding(y) + self.type_embedding(torch.tensor([2], device=device))
        y_pos = self.ar_audio_position(y_emb)

        h_input = torch.concat([x_pos, emo, y_pos], dim=1)

        ## 生成掩码
        # x1和x2只能关注自身
        x_attn_mask = torch.ones((B, x_len, src_len), dtype=torch.bool, device=device)
        indices = torch.arange(x_len, device=device).unsqueeze(0)
        mask = (indices < x1_lens.unsqueeze(1))
        indices = torch.arange(src_len, device=device).unsqueeze(0)
        x1_attn_mask = (indices < x1_lens.unsqueeze(1))
        x1_attn_mask = x1_attn_mask.unsqueeze(1).expand(-1, x_len, -1)
        x2_attn_mask = (indices >= x1_lens.unsqueeze(1)) & (indices < x_lens.unsqueeze(1))
        x2_attn_mask = x2_attn_mask.unsqueeze(1).expand(-1, x_len, -1)
        x_attn_mask[mask, :] = ~x1_attn_mask[mask, :]
        x_attn_mask[~mask, :] = ~x2_attn_mask[~mask, :]

        # emo只能关注自身
        emo_attn_mask = torch.ones((B, 1, src_len), dtype=torch.bool, device=device)
        emo_attn_mask[:, :, x_len] = 0

        y_attn_mask = torch.ones((B, y_len, src_len), dtype=torch.bool, device=device)

        # y1能关注x1以及自身
        indices = torch.arange(y_len, device=device).unsqueeze(0)
        mask = (indices < y1_lens.unsqueeze(1))
        indices = torch.arange(src_len, device=device).unsqueeze(0)
        y1_attn_mask = (indices < x1_lens.unsqueeze(1)) | (indices > x_len) & (indices <= x_len + y1_lens.unsqueeze(1))
        y1_attn_mask = y1_attn_mask.unsqueeze(1).expand(-1, y_len, -1)
        y_attn_mask[mask] = ~y1_attn_mask[mask]

        # cls能关注除y2以外的token
        indices = torch.arange(y_len, device=device).unsqueeze(0)
        mask = (indices == y1_lens.unsqueeze(1))
        indices = torch.arange(src_len, device=device).unsqueeze(0)
        cls_attn_mask = (indices < x_lens.unsqueeze(1)) | (indices == x_len) | (indices > x_len) & (indices <= x_len + y1_lens.unsqueeze(1) + 1)
        cls_attn_mask = cls_attn_mask.unsqueeze(1).expand(-1, y_len, -1)
        y_attn_mask[mask] = ~cls_attn_mask[mask]

        # y2能关注除mask token以外的token
        indices = torch.arange(y_len, device=device).unsqueeze(0)
        mask = (indices > y1_lens.unsqueeze(1)) & (indices < y_lens.unsqueeze(1))
        indices = torch.arange(src_len, device=device).unsqueeze(0)
        y2_attn_mask = (indices < x_lens.unsqueeze(1)) | (indices == x_len) | (indices > x_len) & (indices <= x_len + y_lens.unsqueeze(1))
        y2_attn_mask[:, x_len+1:] &= ~y_mask
        y2_attn_mask = y2_attn_mask.unsqueeze(1).expand(-1, y_len, -1)
        y_attn_mask[mask] = ~y2_attn_mask[mask]


        attn_mask = torch.concat([x_attn_mask, emo_attn_mask, y_attn_mask], dim=1)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1).reshape(B * self.num_head, src_len, src_len)
        new_attn_mask = torch.zeros_like(attn_mask, dtype=emo_feature.dtype, device=device)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

        enc, _ = self.h(
            (h_input, None),
            mask=attn_mask,
        )

        cls_hidden = enc[torch.arange(B, device=device), prompt_lens]
        predicted_duration = self.duration_predictor(cls_hidden).squeeze(-1)
        logits = self.ar_predict_layer(enc[:, x_len+1:][y_mask]) # y_start_idx=x_len+emo_len
        return predicted_duration * x2_lens.float(), torch.argmax(logits, dim=-1), y_mask, y_target