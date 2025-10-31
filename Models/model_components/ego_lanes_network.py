from .pre_trained_backbone import PreTrainedBackbone
from .bev_path_context import BEVPathContext
from .ego_lanes_head import EgoLanesHead

import torch.nn as nn

class EgoLanesNetwork(nn.Module):
    def __init__(self, pretrained):
        super(EgoLanesNetwork, self).__init__()

        # Upstream blocks
        self.PreTrainedBackbone = PreTrainedBackbone(pretrained)

        # Path Context
        self.PathContext = BEVPathContext()

        # EgoLanes Head
        self.EgoLanesHead = EgoLanesHead()
    

    def forward(self, image):
        features = self.PreTrainedBackbone(image)
        deep_features = features[4]
        context = self.PathContext(deep_features)
        ego_left_lane, ego_right_lane = self.EgoLanesHead(context)
        return ego_left_lane, ego_right_lane