"""New fixed six-frame branch; inherited models and their weights stay untouched."""
import torch
from torch import nn
import torch.nn.functional as F
from precip_forecast.models.task62_freeclass import Task62FreeClass,INPUT_KEYS as OLD_KEYS
from precip_forecast.models.task54_rank import ConvNormAct,ResidualBlock,DecoderBlock

INPUT_KEYS=OLD_KEYS+('radar_seq4','seq_available','seq_slot_valid')

def repeat_latest(sequence,slots):
    if sequence.ndim!=5 or sequence.shape[1:3]!=(6,4) or slots.shape!=sequence.shape[:2]:raise ValueError('Six real slots')
    index=torch.where(slots,torch.arange(6,device=slots.device)[None,:],-1).max(1).values
    latest=sequence[torch.arange(len(sequence),device=sequence.device),index.clamp_min(0)]
    latest=latest*(index>=0)[:,None,None,None]
    return latest[:,None].expand(-1,6,-1,-1,-1)

class Task65Sequence(Task62FreeClass):
    def __init__(self,class_frequency=None,*,control=False):
        super().__init__(class_frequency)
        self.control=bool(control)
        self.seq_stem=nn.Sequential(nn.Conv2d(4,32,3,padding=1,bias=False),nn.GroupNorm(8,32),nn.SiLU(inplace=True),ResidualBlock(32))
        self.seq_temporal=nn.Sequential(nn.Conv2d(192,64,1,bias=False),nn.GroupNorm(8,64),nn.SiLU(inplace=True),ResidualBlock(64))
        self.seq_eighth=ConvNormAct(64,256,stride=2)
        self.fusion=nn.Sequential(ConvNormAct(256*6,256,kernel_size=1),ResidualBlock(256),ResidualBlock(256))
        self.decode_quarter=DecoderBlock(256,192,192)
    def forward(self,batch):
        if set(batch)!=set(INPUT_KEYS):raise PermissionError('Only explicit causal input whitelist')
        obs,radar,nwp,prior,valid,available=(batch[k] for k in OLD_KEYS)
        b,c,h,w=obs.shape;seq=batch['radar_seq4'];slots=batch['seq_slot_valid'];sq=batch['seq_available']
        if c!=18 or radar.shape!=(b,18,h//8,w//8) or nwp.shape!=(b,3,84,h//8,w//8):raise ValueError('Inherited grid/channels')
        if seq.shape!=(b,6,4,h//4,w//4) or slots.shape!=(b,6) or sq.shape!=(b,):raise ValueError('New sequence schema')
        if prior.shape!=(b,3,h,w) or valid.shape!=prior.shape or available.shape!=(b,3):raise ValueError('Prior/availability schema')
        if not all(torch.isfinite(batch[k]).all() for k in INPUT_KEYS):raise ValueError('Nonfinite input, not fallback')
        if (prior<0).any() or ((available<0)|(available>1)).any():raise ValueError('Causal value range')
        if slots.dtype!=torch.bool or sq.dtype!=torch.bool:raise ValueError('Actual sequence quality masks')
        available=available.to(obs.dtype)
        half=self.observation_half(obs);quarter=self.observation_quarter(half);eighth=self.observation_eighth(quarter)
        a=available[:,0,None,None,None];half,quarter,eighth=half*a,quarter*a,eighth*a
        rf=self.radar_encoder(radar)*available[:,1,None,None,None]
        nf=self.nwp_encoder(nwp.reshape(b*3,84,h//8,w//8)).reshape(b,3,256,h//8,w//8)*available[:,2,None,None,None,None]
        if self.control:seq=repeat_latest(seq,slots)
        f=self.seq_stem(seq.reshape(b*6,4,h//4,w//4).contiguous(memory_format=torch.channels_last))
        f4=self.seq_temporal(f.reshape(b,192,h//4,w//4));f8=self.seq_eighth(f4)
        gate=(sq.to(obs.dtype)*available[:,1])[:,None,None,None];f4=f4*gate;f8=f8*gate
        skip=torch.cat((quarter,f4),1);embedding=self.lead_embedding(torch.arange(3,device=obs.device));plog=torch.log1p(prior.clamp(0,200))
        result=[]
        for lead in range(3):
            p=F.adaptive_avg_pool2d(torch.stack((plog[:,lead],valid[:,lead].to(plog.dtype)),1),eighth.shape[-2:])
            pf=self.prior_encoder(p);lf=embedding[lead].view(1,-1,1,1).expand(b,-1,h//8,w//8)
            fused=self.fusion(torch.cat((eighth,rf,nf[:,lead],pf,lf,f8),1))
            q=self.decode_quarter(fused,skip);q=self.decode_half(q,half);q=F.interpolate(q,size=(h,w),mode='bilinear',align_corners=False)
            result.append(self.class_head(self.decode_full(q)))
        return {'class_logits':torch.stack(result,1)}

def input_only(batch,device):
    x={k:batch[k].to(device,non_blocking=True) for k in INPUT_KEYS}
    for k in ('observation','radar'):x[k]=x[k].contiguous(memory_format=torch.channels_last)
    return x
