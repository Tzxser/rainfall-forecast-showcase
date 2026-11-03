"""Task62 v2 free five-class output; old Task54 modules/assets stay immutable."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from precip_forecast.models.task54_rank import Task54RankCorrector, Task54RankConfig


INPUT_KEYS = ('observation', 'radar', 'nwp', 'nwp_prior_raw_mm',
              'nwp_prior_valid', 'modality_available')


class Task62FreeClass(Task54RankCorrector):
    """Reuse encoder/decoder definitions, never old weights or prior logits."""
    def __init__(self, class_frequency=None):
        super().__init__(Task54RankConfig())
        del self.rank_delta_head
        self.class_head = nn.Conv2d(64, 5, 1)
        if class_frequency is not None:
            f = torch.as_tensor(class_frequency, dtype=torch.float32)
            if f.shape != (5,) or not torch.isfinite(f).all() or (f <= 0).any():
                raise ValueError('Smoothed LEARN natural frequencies required')
            with torch.no_grad(): self.class_head.bias.copy_(f.log())

    def forward(self, batch):
        if any(k.startswith('target') or k in ('sampler_category','is_natural') for k in batch):
            raise PermissionError('Training labels/sampling reason must not enter model.forward')
        observation, radar, nwp = (batch[k] for k in INPUT_KEYS[:3])
        prior, valid, available = (batch[k] for k in INPUT_KEYS[3:])
        b = observation.shape[0]
        if observation.shape[1] != 18 or radar.shape[1] != 18 or nwp.shape[:3] != (b,3,84):
            raise ValueError('Inherited input schema mismatch')
        if prior.shape != valid.shape or prior.shape != (b,3,*observation.shape[-2:]):
            raise ValueError('Prior shape mismatch')
        if available.shape != (b,3) or not torch.isfinite(available).all():
            raise ValueError('Availability shape/values')
        if not torch.isfinite(prior).all() or (prior < 0).any():
            raise ValueError('Invalid causal prior')
        available = available.to(observation.dtype).clamp(0,1)
        half = self.observation_half(observation)
        quarter = self.observation_quarter(half)
        eighth = self.observation_eighth(quarter)
        a = available[:,0,None,None,None]
        half,quarter,eighth = half*a,quarter*a,eighth*a
        rf = self.radar_encoder(radar)*available[:,1,None,None,None]
        flat = nwp.reshape(b*3,*nwp.shape[2:])
        nf = self.nwp_encoder(flat).reshape(b,3,self.config.width,*flat.shape[-2:])
        nf = nf*available[:,2,None,None,None,None]
        embeddings = self.lead_embedding(torch.arange(3,device=observation.device))
        prior_log = torch.log1p(prior.clamp(0,200))
        output=[]
        for lead in range(3):
            shape=eighth.shape[-2:]
            pl=F.adaptive_avg_pool2d(torch.stack((prior_log[:,lead],valid[:,lead].to(prior_log.dtype)),1),shape)
            pf=self.prior_encoder(pl)
            lf=embeddings[lead].view(1,-1,1,1).expand(b,-1,*shape)
            f=self.fusion(torch.cat((eighth,rf,nf[:,lead],pf,lf),1))
            f=self.decode_quarter(f,quarter)
            f=self.decode_half(f,half)
            f=F.interpolate(f,size=observation.shape[-2:],mode='bilinear',align_corners=False)
            output.append(self.class_head(self.decode_full(f)))
        return {'class_logits':torch.stack(output,1)}


def input_only(batch, device):
    out={k:batch[k].to(device,non_blocking=True) for k in INPUT_KEYS}
    for k in ('observation','radar'):
        out[k]=out[k].contiguous(memory_format=torch.channels_last)
    return out
