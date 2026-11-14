"""Deney 44: görüntüler arası piksel karşılaştırmalı öğrenme (Wang ve ark., ICCV 2021, "Exploring Cross-Image
Pixel Contrast for Semantic Segmentation").

Decoder çıktısına (SegmentationHead girdisi) yalnız eğitimde kullanılan 64 boyutlu bir projeksiyon başlığı
eklenir. Filament pikselleri (hedef = 1: görüntüdeki tüm anotatörler çizmiş) ile arka plan pikselleri (hedef = 0)
özellik uzayında ayrıştırılır. 0 < hedef < 1 olan uyuşmazlık pikselleri bu kayba girmez. Önceki batch'lerden
tespit edilmiş (detach) örnekler bir bellek kuyruğunda tutulur (görüntüler arası karşılaştırma). Arka plan
örneklerinin yarısı "zor" (modelin olasılığı yüksek) piksellerden seçilir.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelContrast(nn.Module):
    def __init__(self, in_ch=16, dim=64, n_per_class=256, mem=4096, tau=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1), nn.ReLU(inplace=True), nn.Conv2d(in_ch, dim, 1))
        self.n, self.tau, self.mem = n_per_class, tau, mem
        self.register_buffer("bank", torch.zeros(0, dim))
        self.register_buffer("bank_lab", torch.zeros(0, dtype=torch.long))

    def attach(self, model):
        """SegmentationHead girdisini (decoder çıktısı) self.dec'e yakalar."""
        model.segmentation_head.register_forward_hook(lambda m, i, o: setattr(self, "dec", i[0]))
        return self

    def _sample(self, feat, y, p):
        """feat (C,H,W), y (H,W) hedef, p (H,W) olasılık -> (N,C) özellik, (N,) etiket."""
        out_f, out_l = [], []
        pos = (y >= 0.999).nonzero(as_tuple=False)
        neg = (y <= 0.001).nonzero(as_tuple=False)
        if len(pos) == 0 or len(neg) == 0:
            return None
        k = min(self.n, len(pos))
        out_f.append(pos[torch.randperm(len(pos), device=y.device)[:k]])
        out_l.append(torch.ones(k, dtype=torch.long, device=y.device))
        pn = p[neg[:, 0], neg[:, 1]]
        hard = neg[pn > 0.3]
        kh = min(self.n // 2, len(hard))
        if kh:
            out_f.append(hard[torch.randperm(len(hard), device=y.device)[:kh]])
        ke = self.n - kh
        out_f.append(neg[torch.randperm(len(neg), device=y.device)[:ke]])
        out_l.append(torch.zeros(kh + ke, dtype=torch.long, device=y.device))
        idx = torch.cat(out_f)
        return feat[:, idx[:, 0], idx[:, 1]].t(), torch.cat(out_l)

    def forward(self, dec, y, logits):
        """dec: (B,C,H,W) decoder çıktısı; y: (B,1,H,W) yumuşak hedef; logits: (B,1,H,W)."""
        z = F.normalize(self.proj(dec.float()), dim=1)
        p = torch.sigmoid(logits.detach().float())
        feats, labs = [], []
        for b in range(z.shape[0]):
            s = self._sample(z[b], y[b, 0], p[b, 0])
            if s is not None:
                feats.append(s[0])
                labs.append(s[1])
        if not feats:
            return z.sum() * 0
        f, lab = torch.cat(feats), torch.cat(labs)
        allf, alll = torch.cat([f, self.bank]), torch.cat([lab, self.bank_lab])
        sim = f @ allf.t() / self.tau
        self_mask = torch.zeros_like(sim, dtype=torch.bool)
        self_mask[:, : len(f)] = torch.eye(len(f), dtype=torch.bool, device=f.device)
        sim = sim.masked_fill(self_mask, -1e9)
        same = (lab[:, None] == alll[None, :]) & ~self_mask
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        loss = -(logp * same).sum(1) / same.sum(1).clamp(min=1)
        with torch.no_grad():  # görüntüler arası bellek (FIFO)
            self.bank = torch.cat([f.detach(), self.bank])[: self.mem]
            self.bank_lab = torch.cat([lab, self.bank_lab])[: self.mem]
        return loss.mean()
