from __future__ import annotations
from pathlib import Path
import timm,torch
from torch import nn
from torch.nn import functional as F
BACKBONE_NAME="vit_small_patch14_dinov2.lvd142m"; INPUT_SIZE=518; VIEW_ORDER=("small","medium","full")
class MaterialMagicWandEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone=timm.create_model(BACKBONE_NAME,pretrained=False,num_classes=0,img_size=INPUT_SIZE)
        if not hasattr(self.backbone,"mask_token"): self.backbone.mask_token=nn.Parameter(torch.zeros(1,384))
        self.head=nn.Sequential(nn.Linear(1152,384),nn.ReLU(),nn.Linear(384,128))
    def forward(self,views):
        if views.ndim!=5 or tuple(views.shape[1:3])!=(3,3): raise ValueError(f"Expected [B,3,3,H,W], got {tuple(views.shape)}")
        f=self.backbone(views.reshape(-1,*views.shape[2:])); x=f.reshape(views.shape[0],-1)
        return x,F.normalize(self.head(x),dim=-1)
def load_released_encoder(checkpoint:str|Path):
    state=torch.load(Path(checkpoint),map_location="cpu",weights_only=True); model=MaterialMagicWandEncoder()
    incompatible=model.load_state_dict(state,strict=True); model.eval()
    return model,{"strict_load":not incompatible.missing_keys and not incompatible.unexpected_keys,
                  "missing_keys":list(incompatible.missing_keys),"unexpected_keys":list(incompatible.unexpected_keys),
                  "backbone_name":BACKBONE_NAME,"tensor_count":len(state)}
