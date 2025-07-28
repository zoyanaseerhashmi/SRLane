import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from srlane.models.registry import HEADS
from srlane.models.losses.seg_loss import SegLoss


@HEADS.register_module
class LocalAngleHead(nn.Module):
    """Local angle prediction head.

    Args:
        num_points: Number of lane points.
        in_channel: Input channels.
        cfg: Model config.
    """

    def __init__(self,
                 num_points: int = 72,
                 in_channel: int = 64,
                 cfg=None,
                 ):
        super(LocalAngleHead, self).__init__()
        self.n_offsets = num_points
        self.cfg = cfg
        self.img_w = cfg.img_w
        self.img_h = cfg.img_h
        self.aux_seg = self.cfg.get("seg_loss_weight", 0.) > 0.
        self.feat_h, self.feat_w = self.cfg.angle_map_size
        # Cartesian coordinates
        self.register_buffer(name="prior_ys",
                             tensor=torch.linspace(0, self.feat_h,
                                                   steps=self.n_offsets,
                                                   dtype=torch.float32))
        grid_y, grid_x = torch.meshgrid(torch.arange(self.feat_h - 0.5, 0,
                                                     -1, dtype=torch.float32),
                                        torch.arange(0.5, self.feat_w,
                                                     1, dtype=torch.float32),
                                        indexing="ij")
        grid = torch.stack((grid_x, grid_y), 0)
        grid.unsqueeze_(0)  # (1, 2, h, w)
        self.register_buffer(name="grid", tensor=grid)

        self.angle_conv = nn.ModuleList()
        for _ in range(self.cfg.n_fpn):
            self.angle_conv.append(nn.Conv2d(in_channel, 1,
                                             1, 1, 0, bias=False))

        if self.aux_seg:
            num_classes = self.cfg.max_lanes + 1
            self.seg_conv = nn.ModuleList()
            for _ in range(self.cfg.n_fpn):
                self.seg_conv.append(nn.Conv2d(in_channel, num_classes,
                                               1, 1, 0))
            self.seg_criterion = SegLoss(num_classes=num_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.angle_conv.parameters():
            nn.init.normal_(m, 0., 1e-3)

    # def tan(self, theta):
    #     return

    # def tan(self,theta):
    #     i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))  # Define the imaginary unit i
    #     e_i_theta = torch.exp(i * theta)  # Compute e^(i*theta)
    #     e_minus_i_theta = torch.exp(-i * theta)  # Compute e^(-i*theta)
        
    #     numerator = -i * (e_i_theta - e_minus_i_theta)  # Compute the numerator
    #     denominator = e_i_theta + e_minus_i_theta  # Compute the denominator
        
    #     return (numerator / denominator).real  # Compute the tangent

    def gt(self, a, b):
        # """
        # Implements a > b without direct comparison operators or sign.
        # Returns 1.0 if a > b, else 0.0.
        # """
        # diff = a - b
        # # Compute relu(diff) using abs
        # positive_diff = (diff + torch.abs(diff)) / 2
        # eps = 1e-10 # Small epsilon to prevent division by zero
        # result = positive_diff / (positive_diff + eps)
        # return torch.round(result)

        """
        Implements a > b without direct comparison operators or sign.
        Uses torch.ceil to return 1.0 if a > b, else 0.0.
        """
        diff = a - b
        # If diff > 0, we want 1.
        # If diff <= 0, we want 0.

        # Clamp diff to be non-negative.
        # This acts like a ReLU. If diff is negative, it becomes 0.
        # If diff is positive, it stays positive.
        clamped_diff = torch.clamp(diff, min=0.0)

        # Now, if clamped_diff > 0, ceil(clamped_diff) will be >= 1.0
        # If clamped_diff == 0, ceil(clamped_diff) will be 0.0
        # We can then clamp this result to be between 0 and 1.
        result = torch.ceil(clamped_diff)
        return torch.clamp(result, min=0.0, max=1.0)

    def gte(self, a, b): # a >= b
        """
        Implements a >= b.
        Returns 1.0 if a >= b, else 0.0.
        Leverages (a > b) OR (a == b).
        """
        # Option 1: a >= b is equivalent to NOT (b > a)
        # Not (b > a) means 1 - (b > a)
        return 1.0 - self.gt(b, a)

        # Option 2: a >= b is also equivalent to a + small_num > b (to handle equality)
        # Or, it's 1 - (b > a) where b > a means b is strictly greater than a.

    def lt(self, a, b): # a < b
        """
        Implements a < b.
        Returns 1.0 if a < b, else 0.0.
        """
        # a < b is equivalent to b > a
        return self.gt(b, a)

    def lte(self, a, b): # a <= b
        """
        Implements a <= b.
        Returns 1.0 if a <= b, else 0.0.
        """
        # a <= b is equivalent to NOT (a > b)
        return 1.0 - self.gt(a, b)

    def tan(self, theta):
        itheta = torch.complex(torch.tensor(0.0),theta)
        return -torch.tanh(itheta).imag

    def forward(self,
                feats: List[Tensor], ):
        """This method performs the forward propagation process.

        Args:
        - feats: List of feature maps.

        Returns:
        - Tensor: Lane proposals.
        - Optional[List[Tensor]]: predicted angle map, used for training.
        """
        theta_list = []
        # In testing mode, only the deepest feature is used.
        if not self.training:
            feats = feats[-1:]
        for i, feat in enumerate(feats, 1):
            theta = self.angle_conv[len(feats) - i](feat).sigmoid()
            theta_list.append(theta)
        if self.aux_seg:
            seg_list = []
            for i, feat in enumerate(feats, 1):
                seg = self.seg_conv[len(feats) - i](feat)
                seg_list.append(seg)
        # theta = theta_list[-1].squeeze(1)
        theta = theta_list[-1]
        angle = F.interpolate(theta,
                              size=[self.feat_h, self.feat_w],
                            #   scale_factor=[1,1],
                              mode="bilinear",
                              align_corners=True)
        # bs, _, h, w = angle.shape
        bs = theta.shape[0]
        h,w = self.feat_h, self.feat_w
        angle = angle.reshape(bs,1,h,w)
        # angle = angle.squeeze(1)

        # # if len(angle.size()) == 4 and angle.size(1) == 1:
        # B,C,H,W = angle.size()
        # # angle = angle.view(B,H,W)
        # # angle = angle.permute(1, 0, 2, 3)
        # angle = angle.view(C*B,H,W)
 
        angle = angle.detach()
        # Remove excessively tilted angles, optional
        angle.clamp_(min=0.05, max=0.95)
        # Build lane proposals
        # k = (angle * math.pi).tan()
        # k = self.tan(angle * math.pi)
        k = angle.clone()
        # k = k.permute(0,2,3,1)
        # k = torch.rand_like(angle) * (0.1583 - (-0.1583)) + (-0.1583)
        # bs, _, h, w = angle.shape
        # bs, h, w = angle.shape
                
        grid = self.grid
        ws = ((self.prior_ys.view(1, 1, self.n_offsets)
               - grid[:, 1].view(1, h * w, 1)) / k.view(bs, h * w, 1)
              + grid[:, 0].view(1, h * w, 1))  # (bs, h*w, n_offsets)
        # ws = ((self.prior_ys.view(1, 1, self.n_offsets)
        # - grid[:, 1].view(1, h * w, 1)) / k.view(bs, h * w, 1)
        # + grid[:, 0].view(1, h * w, 1))  # (bs, h*w, n_offsets)
        ws = ws / w # (bs, h*w, n_offsets) -> 160,40,72
        ###### uncomment here
        valid_mask = self.lt(ws, 1)
        # # # valid_mask = (0 <= ws) & (ws < 1)
        # valid_mask = torch.clamp(ws, 0, 0.999999)
        valid_mask = valid_mask.unsqueeze(-1)
        # # indices = valid_mask[:,:,0]
        # valid_mask = valid_mask.float()  # Convert boolean tensor to float tensor
        _, indices = valid_mask.max(2)
        # # print(indices.shape)  # (bs, h*w) -> 160,40
        start_y = indices / (self.n_offsets - 1)  # (bs, h*w) -> 160,4*10
        # # start_y = start_y.unsqueeze(2)  # (bs, h*w, 1)
        # # print(start_y.shape)  # (bs, 1, h*w) -> 160,1,40
        # # print()
        ###### upto here

        ### 2D start_y
        priors = ws.new_zeros(
            (bs, h * w, 2 + 2 + self.n_offsets), device=ws.device) #160,40,76  -> 1,160,40,76 ??
        priors[..., 2:3] =  start_y ## 160,40 ##.unsqueeze(0) # 160,40,1
        priors[..., 4:] = ws # 160,40,72
        priors = priors.unsqueeze(-1)  # Remove the first dimension (1)
        # # Create priors with the new shape (1, bs, h * w, 2 + 2 + self.n_offsets)
        # priors = ws.new_zeros(
        #     (1, bs, h * w, 2 + 2 + self.n_offsets), device=ws.device)  # Shape: (1, 160, 40, 76)

        # # Expand start_y to match the new shape and assign it to priors
        # start_y = start_y.view(1, bs, h * w, 1)  # Shape: (1, 160, 40, 1)
        # priors[..., 2:3] = start_y  # Assign start_y to the 3rd column (index 2)
        # priors = priors.squeeze(0)  # Remove the first dimension (1)

        # # Expand ws to match the new shape and assign it to priors
        # # ws = ws.view(1, bs, h * w, self.n_offsets)  # Shape: (1, 160, 40, 72)
        # priors[..., 4:] = ws  # Assign ws starting from the 5th column (index 4)

        # # Concatenate tensors to avoid Gather 
        # start_y = start_y.unsqueeze(2)  # (bs, h*w, 1)
        # priors = torch.cat([torch.zeros((bs, h * w, 2), device=ws.device),
        #                     start_y,
        #                     torch.zeros((bs, h * w, 1), device=ws.device),
        #                     ws], dim=2)

        return dict(priors=priors,
                    pred_angle=[theta.squeeze(1) for theta in theta_list]
                    if self.training else None,
                    pred_seg=seg_list
                    if (self.training and self.aux_seg) else None)

    def loss(self,
             pred_angle: List[Tensor],
             pred_seg: List[Tensor],
             gt_angle: List[Tensor],
             gt_seg: List[Tensor],
             loss_weight: Tuple[float] = [0.2, 0.2, 1.],
             ignore_value: float = 0.,
             **ignore_kwargs):
        """ L1 loss for local angle estimation over multi-level features.

        Args:
        - pred_angle: List of estimated angle maps.
        - gt_angle: List of target angle maps.
        - loss_weight: Loss weights of each map.
        - ignore_value: Placeholder value for non-target.

        Returns:
        - Tensor: The calculated angle loss.
        """
        angle_loss = 0
        for pred, target, weight in zip(pred_angle, gt_angle, loss_weight):
            valid_mask = target > ignore_value
            angle_loss = (angle_loss
                          + ((pred - target).abs() * valid_mask).sum()
                          / (valid_mask.sum() + 1e-4)) * weight
        if self.aux_seg:
            seg_loss = 0
            for pred, target, weight in zip(pred_seg, gt_seg, loss_weight):
                seg_loss = seg_loss + self.seg_criterion(pred, target) * weight
            return {"angle_loss": angle_loss,
                    "seg_loss": seg_loss, }

        return {"angle_loss": angle_loss}

    # def __repr__(self):
    #     num_params = sum(map(lambda x: x.numel(), self.parameters()))
    #     return f"#Params of {self._get_name()}: {num_params / 10 ** 3:<.2f}[K]"
