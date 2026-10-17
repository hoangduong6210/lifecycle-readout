"""Recorded-artist temporal-neighborhood model with matched dose controls."""
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from .event_coupling_study import EventCouplingControl,ARMS


class RecordedArtistControl(EventCouplingControl):
    def __init__(self,kind,settings,phi_mean,phi_std,time_unit,exposure_unit,*,artist_vocabulary,seed):
        super().__init__(kind,settings['artist_embedding_dim']+2,settings,phi_mean,phi_std,time_unit,exposure_unit)
        if artist_vocabulary<1:raise ValueError('Training-known artist vocabulary required')
        self.artist_vocabulary=int(artist_vocabulary)
        # Branch constructors consume different random draws. Isolate lookup
        # initialization so every arm still shares EXACT common initial tensors.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed)+9173)
            self.artist_embedding=nn.Embedding(self.artist_vocabulary+1,settings['artist_embedding_dim'],padding_idx=0)

    def forward(self,x,dt,mask,phi,pair_dt):
        encoded=torch.cat((self.artist_embedding(x[...,0].long()),x[...,1:]),dim=-1)
        return super().forward(encoded,dt,mask,phi,pair_dt)


def artist_arm_stream(stream,extra,kind):
    if kind not in ARMS:raise ValueError('Unknown declared arm')
    phi=stream.pair_features.copy()
    if kind=='activity_mlp':phi[:,:,7]=0
    if kind=='shuffled_phase':
        phi[:,:,7]=np.log1p(extra['alternate_exposure']).astype(np.float32)
        phi[:,:,8]=phi[:,:,10]  # unavailable alternate is explicitly flagged
    return replace(stream,pair_features=phi)
