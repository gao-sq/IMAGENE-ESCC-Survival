
import timm
import torch
import torch.nn as nn

from mmengine.registry import MODELS

@MODELS.register_module()
class GigaPathPatchEncoder(nn.Module):
    def __init__(self, pretrained='path/to/prov_gigapath_weights.pth'):
        super(GigaPathPatchEncoder, self).__init__()
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
        if pretrained:
            self.model.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=True)

    def forward(self, x):
        self.model(x)
        return x