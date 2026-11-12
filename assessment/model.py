"""Turn-level text, acoustic and visual regression of PHQ-8 totals."""

import torch
from torch import nn


class TurnRegressor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=.2, target_mean=0., target_std=1.):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.sequence = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.regressor = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_dim, 1))
        self.register_buffer('target_mean', torch.tensor(float(target_mean)))
        self.register_buffer('target_std', torch.tensor(float(target_std)))

    def forward(self, features, mask):
        if features.ndim != 3 or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('batched_turn_features_and_boolean_mask_required')
        lengths = mask.sum(1)
        if (lengths == 0).any() or not torch.equal(mask, torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]):
            raise ValueError('nonempty_right_padded_turn_mask_required')
        projected = self.project(features)
        packed = nn.utils.rnn.pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
        sequence, _ = self.sequence(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True, total_length=features.shape[1])
        weights = self.attention(sequence).squeeze(-1).masked_fill(~mask, float('-inf')).softmax(-1)
        pooled = (sequence * weights.unsqueeze(-1)).sum(1)
        # Learn standardized continuous total without sigmoid saturation.
        # Presentation clipping, when requested, is separately reported.
        return self.regressor(pooled).squeeze(-1) * self.target_std + self.target_mean


class TrimodalRegressor(nn.Module):
    def __init__(self, text_architecture, modality_dropout=.1):
        super().__init__()
        self.text = TurnRegressor(**text_architecture)
        width = int(text_architecture['hidden_dim'])
        self.audio_project = nn.Sequential(nn.Linear(316, width), nn.GELU(), nn.Dropout(.2))
        self.visual_project = nn.Sequential(nn.Linear(152, width), nn.GELU(), nn.Dropout(.2))
        self.audio_gate = nn.Linear(2 * width + 1, 1)
        self.visual_gate = nn.Linear(2 * width + 1, 1)
        for gate in (self.audio_gate, self.visual_gate):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -4.)
        self.modality_dropout = float(modality_dropout)

    def new_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith('text.')]

    def freeze_text(self, freeze):
        self.text.requires_grad_(not freeze)

    def configure_training_phase(self, freeze_text):
        """Freeze copied weights/dropout while retaining cuDNN input gradients."""
        self.freeze_text(freeze_text)
        self.train()
        if freeze_text:
            self.text.eval()
            # AV residuals must backpropagate THROUGH the frozen GRU. cuDNN
            # allocates its backward reserve only for an RNN training forward.
            # The copied single-layer GRU has no internal dropout, so this
            # preserves deterministic copied-text dropout during warmup.
            if self.text.sequence.num_layers != 1 or self.text.sequence.dropout != 0:
                raise ValueError('frozen_text_warmup_requires_single_layer_GRU_without_internal_dropout')
            self.text.sequence.train()

    def forward(self, text, mask, audio, audio_valid, visual, visual_valid):
        lengths = mask.sum(1)
        expected = torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
        if mask.dtype != torch.bool or (lengths == 0).any() or not torch.equal(mask, expected):
            raise ValueError('nonempty_right_padded_turn_mask_required')
        projected = self.text.project(text)
        text_projected = projected
        for value, valid, projector, gate in (
            (audio, audio_valid, self.audio_project, self.audio_gate),
            (visual, visual_valid, self.visual_project, self.visual_gate)):
            if valid.dtype != torch.bool or valid.shape != value.shape or not torch.isfinite(value).all():
                raise ValueError('finite_modality_with_explicit_valid_mask_required')
            active = valid.any(-1) & mask
            if self.training and self.modality_dropout:
                # Independent subject/modality drop, preserving the text route.
                active = active & (torch.rand((len(text), 1), device=text.device) >= self.modality_dropout)
            joined = torch.cat((value.masked_fill(~valid, 0.), valid.to(value.dtype)), -1)
            modality_projected = projector(joined)
            coverage = valid.to(value.dtype).mean(-1, keepdim=True)
            coefficient = gate(torch.cat((text_projected, modality_projected, coverage), -1)).sigmoid()
            residual = modality_projected * coefficient
            projected = projected + residual * active.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
        sequence, _ = self.text.sequence(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True, total_length=text.shape[1])
        weights = self.text.attention(sequence).squeeze(-1).masked_fill(~mask, float('-inf')).softmax(-1)
        pooled = (sequence * weights.unsqueeze(-1)).sum(1)
        return self.text.regressor(pooled).squeeze(-1) * self.text.target_std + self.text.target_mean


def collate(rows, device):
    lengths = [len(r['text']) for r in rows]
    values = []
    for key, width, dtype in (('text', 768, torch.float32), ('audio', 158, torch.float32),
                              ('audio_valid', 158, torch.bool), ('visual', 76, torch.float32),
                              ('visual_valid', 76, torch.bool)):
        out = torch.zeros((len(rows), max(lengths), width), dtype=dtype, device=device)
        for i, row in enumerate(rows):
            out[i, :lengths[i]] = row[key].to(device=device, dtype=dtype)
        values.append(out)
    mask = torch.arange(max(lengths), device=device)[None] < torch.tensor(lengths, device=device)[:, None]
    return (values[0], mask, *values[1:])


def fit_normalization(rows):
    """Train-only equal subject moments; missing coordinates contribute nothing."""
    result = {}
    for kind in ('audio', 'visual'):
        means, squares, available = [], [], []
        for row in rows:
            x, valid = row[kind].float(), row[kind + '_valid']
            n = valid.sum(0)
            means.append(x.masked_fill(~valid, 0.).sum(0) / n.clamp_min(1))
            squares.append(x.square().masked_fill(~valid, 0.).sum(0) / n.clamp_min(1))
            available.append(n > 0)
        present = torch.stack(available)
        count = present.sum(0)
        mean = torch.stack(means).sum(0) / count.clamp_min(1)
        var = torch.stack(squares).sum(0) / count.clamp_min(1) - mean.square()
        std = var.clamp_min(1e-6).sqrt()
        result[kind] = dict(mean=mean, std=std, train_valid=count > 0)
    return result


def normalize(rows, normalization, text_normalization):
    result = []
    for row in rows:
        out = dict(row)
        out['text'] = (row['text'] - text_normalization['mean']) / text_normalization['std']
        for kind in ('audio', 'visual'):
            stat = normalization[kind]
            valid = row[kind + '_valid'] & stat['train_valid']
            out[kind + '_valid'] = valid
            out[kind] = ((row[kind] - stat['mean']) / stat['std']).masked_fill(~valid, 0.)
        out['has_av'] = bool(out['audio_valid'].any() or out['visual_valid'].any())
        result.append(out)
    return result
