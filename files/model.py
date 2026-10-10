"""The caption decoder: Step-Wise Visually Grounded 2-layer GRU -> words."""
import torch
import torch.nn as nn


class CaptionGRU(nn.Module):
    def __init__(self, vocab_size, feat_dim=2048, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.init_h = nn.Linear(feat_dim, hidden_dim * num_layers)  # image feature -> starting memory
        self.feat_proj = nn.Linear(feat_dim, embed_dim)             # image feature -> visual context token
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # GRU receives word embedding (embed_dim) + visual context (embed_dim) at EVERY timestep
        self.gru = nn.GRU(embed_dim * 2, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, feats):
        h = torch.tanh(self.init_h(feats))  # (B, L*H)
        return h.view(-1, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()  # (L, B, H)

    def forward(self, feats, caps):
        """Teacher forcing with step-wise visual grounding.
        feats: (B, 2048)
        caps: (B, T)
        """
        seq_len = caps.size(1) - 1
        w_emb = self.drop(self.embed(caps[:, :-1]))  # (B, T-1, embed_dim)
        v_proj = self.drop(torch.relu(self.feat_proj(feats))).unsqueeze(1).expand(-1, seq_len, -1)  # (B, T-1, embed_dim)
        gru_input = torch.cat([w_emb, v_proj], dim=-1)  # (B, T-1, 2 * embed_dim)
        out, _ = self.gru(gru_input, self.init_hidden(feats))
        return self.fc(self.drop(out))

    @torch.no_grad()
    def generate(self, feats, start_idx, end_idx, max_len=20):
        """Greedy decoding with perpetual visual grounding."""
        batch_size = feats.size(0)
        h = self.init_hidden(feats)
        v_proj = torch.relu(self.feat_proj(feats)).unsqueeze(1)  # (B, 1, embed_dim)
        tok = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=feats.device)
        ids = []
        for _ in range(max_len):
            w_emb = self.embed(tok)
            gru_input = torch.cat([w_emb, v_proj], dim=-1)
            out, h = self.gru(gru_input, h)
            tok = self.fc(out[:, -1]).argmax(-1, keepdim=True)
            ids.append(tok)
            if (tok == end_idx).all():
                break
        return torch.cat(ids, dim=1)

    @torch.no_grad()
    def beam_search(self, feat, start_idx, end_idx, beam=3, max_len=20):
        """Visually-grounded beam search for ONE image (feat shape: (1, feat_dim))."""
        beam = max(1, beam)
        h0 = self.init_hidden(feat)
        v_proj = torch.relu(self.feat_proj(feat)).unsqueeze(1)  # (1, 1, embed_dim)
        beams = [(0.0, [start_idx], h0)]
        done = []
        for step in range(max_len):
            cand = []
            for score, seq, h in beams:
                tok = torch.tensor([[seq[-1]]], dtype=torch.long, device=feat.device)
                w_emb = self.embed(tok)
                gru_input = torch.cat([w_emb, v_proj], dim=-1)
                out, h2 = self.gru(gru_input, h)
                logp = torch.log_softmax(self.fc(out[:, -1]), dim=-1)[0]
                k = min(beam * 2, logp.size(-1))
                top = logp.topk(k)
                for lp, i in zip(top.values.tolist(), top.indices.tolist()):
                    penalty = 0.0
                    if len(seq) > 1 and i == seq[-1]:
                        penalty += 2.5
                    elif i in seq[-3:]:
                        penalty += 1.0
                    elif i in seq:
                        penalty += 0.5
                    if i == end_idx and len(seq) < 4:
                        penalty += 3.0
                    cand.append((score + lp - penalty, seq + [i], h2))
            cand.sort(key=lambda c: c[0], reverse=True)
            beams = []
            for score, seq, h2 in cand[:beam]:
                if seq[-1] == end_idx:
                    done.append((score / (len(seq) ** 0.7), seq))
                else:
                    beams.append((score, seq, h2))
            if not beams:
                break
        if not done:
            done = [(s / (len(q) ** 0.7), q) for s, q, _ in beams]
        return max(done, key=lambda d: d[0])[1][1:]
