import torch
import torch.nn.functional as F

import sys

from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help

# def print_memory(tensor, name):
#     print(f"{name}: {sys.getsizeof(tensor.storage()) / (1024 ** 2):.2f} MB")

# def bilinear_interpolate_torch(im, x, y):

#     N, C, H, W = im.size() 
    
#     _, n_queries, n_points = x.size() #160,40,36
#     x0 = torch.floor(x)
#     x1 = x0 + 1
#     y0 = torch.floor(y)
#     y1 = y0 + 1

#     x0 = torch.clamp(x0, 0, W - 1)
#     x1 = torch.clamp(x1, 0, W - 1)
#     y0 = torch.clamp(y0, 0, H - 1)
#     y1 = torch.clamp(y1, 0, H - 1)

#     x0 = x0.long()
#     x1 = x1.long()
#     y0 = y0.long()
#     y1 = y1.long()

    
#     # Flatten the indices and the input tensor for sampling
#     flat_im = im.view(N, C, -1)
#     flat_indices = y0 * W + x0
#     flat_indices = flat_indices.view(N, -1)
#     # print_memory(flat_indices, "flat_indices")
#     Ia = torch.gather(flat_im, 2, flat_indices.unsqueeze(1).expand(-1, C, -1)).view(N, C, n_queries, n_points)
#     # print_memory(Ia, "Ia")s
#     flat_indices = y1 * W + x0
#     flat_indices = flat_indices.view(N, -1)
#     Ib = torch.gather(flat_im, 2, flat_indices.unsqueeze(1).expand(-1, C, -1)).view(N, C, n_queries, n_points)
#     flat_indices = y0 * W + x1
#     flat_indices = flat_indices.view(N, -1)
#     Ic = torch.gather(flat_im, 2, flat_indices.unsqueeze(1).expand(-1, C, -1)).view(N, C, n_queries, n_points)
#     flat_indices = y1 * W + x1
#     flat_indices = flat_indices.view(N, -1)
#     Id = torch.gather(flat_im, 2, flat_indices.unsqueeze(1).expand(-1, C, -1)).view(N, C, n_queries, n_points)

#     # # Create a mesh grid for batch and channel dimensions
#     # N_idx = torch.arange(N, device=im.device).view(N, 1, 1).expand(N, n_queries, n_points)
#     # C_idx = torch.arange(C, device=im.device).view(1, C, 1, 1).expand(N, C, n_queries, n_points)

#     # # Gather values from the input tensor
#     # Ia = im[N_idx, C_idx, y0.unsqueeze(1).expand(N, C, -1, -1), x0.unsqueeze(1).expand(N, C, -1, -1)]
#     # Ib = im[N_idx, C_idx, y1.unsqueeze(1).expand(N, C, -1, -1), x0.unsqueeze(1).expand(N, C, -1, -1)]
#     # Ic = im[N_idx, C_idx, y0.unsqueeze(1).expand(N, C, -1, -1), x1.unsqueeze(1).expand(N, C, -1, -1)]
#     # Id = im[N_idx, C_idx, y1.unsqueeze(1).expand(N, C, -1, -1), x1.unsqueeze(1).expand(N, C, -1, -1)]
    
#     # # # # # Expand dimensions to match the input tensor
#     # # # # x0 = x0.unsqueeze(1).expand(-1, C, -1, -1)
#     # # # # x1 = x1.unsqueeze(1).expand(-1, C, -1, -1)
#     # # # # y0 = y0.unsqueeze(1).expand(-1, C, -1, -1)
#     # # # # y1 = y1.unsqueeze(1).expand(-1, C, -1, -1)
    
#     # # # a = torch.concat([x0.unsqueeze(1), y0.unsqueeze(1)])
#     # # # b = torch.concat([x1.unsqueeze(1), y0.unsqueeze(1)])
#     # # # c = torch.concat([x0.unsqueeze(1), y1.unsqueeze(1)])
#     # # # d = torch.concat([x1.unsqueeze(1), y1.unsqueeze(1)])

#     # # # # Gather values from the input tensor
#     # # # Ia = im[...,a]
#     # # # Ib = im[...,b]
#     # # # Ic = im[...,c]
#     # # # Id = im[...,d]

#     wa = (x1.float() - x) * (y1.float() - y)
#     wb = (x1.float() - x) * (y - y0.float())
#     wc = (x - x0.float()) * (y1.float() - y)
#     wd = (x - x0.float()) * (y - y0.float())

#     wa = wa.unsqueeze(1)
#     wb = wb.unsqueeze(1)
#     wc = wc.unsqueeze(1)
#     wd = wd.unsqueeze(1)

#     return wa*Ia + wb*Ib + wc*Ic + wd*Id
#     # return wa + wb + wc + wd

def bilinear_interpolate_torch(im,x,y):
    '''
    im : B,C,H,W
    y : 1,numPoints -- pixel location y float
    x : 1,numPOints -- pixel location y float
    '''

    B, C, H, W = im.size()
    _, n_queries, n_points = x.size() #160,40,36
    x = x.reshape(B,-1) #160,1440
    y = y.reshape(B,-1) #160,1440

    x0 = torch.clamp(x, 0, W - 1)
    x1 = torch.clamp(x+1, 0, W - 1)
    y0 = torch.clamp(y, 0, H - 1)
    y1 = torch.clamp(y+1, 0, H - 1)

    x0 = x0.long()
    x1 = x1.long()
    y0 = y0.long()
    y1 = y1.long()

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    # Instead of clamp
    # x1 = x1 - torch.floor(x1 / im.shape[3]).int()
    # y1 = y1 - torch.floor(y1 / im.shape[2]).int()

    Ia = im[:, :, y0, x0]
    Ib = im[:, :, y1, x0]
    Ic = im[:, :, y0, x1]
    Id = im[:, :, y1, x1]

    out =  Ia  * wa + Ib * wb + Ic * wc + Id * wd
    out = out.squeeze(1) # 160,64,1,40,36 -> 160,64,40,36
    out = out.view(B, C, n_queries, n_points)
    return out


def custom_grid_sample(input, grid, align_corners=True):
    N, C, H, W = input.size()
    x = grid[..., 0] # 160,40,36
    y = grid[..., 1] # 160,40,36

    if align_corners:
        x = ((x + 1) * (W - 1)) / 2
        y = ((y + 1) * (H - 1)) / 2
    else:
        x = ((x + 1) * W - 1) / 2
        y = ((y + 1) * H - 1) / 2
    return bilinear_interpolate_torch(input, x, y)

# def grid_sampler(g, input, grid, mode, padding_mode, align_corners):
#     # mode
#     #   'bilinear'      : onnx::Constant[value={0}]
#     #   'nearest'       : onnx::Constant[value={1}]
#     #   'bicubic'       : onnx::Constant[value={2}]
#     # padding_mode
#     #   'zeros'         : onnx::Constant[value={0}]
#     #   'border'        : onnx::Constant[value={1}]
#     #   'reflection'    : onnx::Constant[value={2}]
#     mode = sym_help._maybe_get_const(mode, "i")
#     padding_mode = sym_help._maybe_get_const(padding_mode, "i")
#     mode_str = ['bilinear', 'nearest', 'bicubic'][mode]
#     padding_mode_str = ['zeros', 'border', 'reflection'][padding_mode]
#     align_corners = int(sym_help._maybe_get_const(align_corners, "b"))

#     # From opset v13 onward, the output shape can be specified with
#     # (N, C, H, W) (N, H_out, W_out, 2) => (N, C, H_out, W_out)
#     # input_shape = input.type().sizes()
#     # grid_shape = grid.type().sizes()
#     # output_shape = input_shape[:2] + grid_shape[1:3]
#     # g.op(...).setType(input.type().with_sizes(output_shape))

#     return g.op("com.microsoft::GridSample", input, grid,
#                 mode_s=mode_str,
#                 padding_mode_s=padding_mode_str,
#                 align_corners_i=align_corners)
    
# register_custom_op_symbolic('::grid_sampler', grid_sampler, 1)

def sampling_each_level(sample_points: torch.Tensor,
                        value: torch.Tensor,
                        weight=None):
    B, n_queries, n_points, _ = sample_points.shape
    _, C, H_feat, W_feat = value.shape

    # # `sampling_points` now has the shape [B*n_groups, n_queries, n_points, 2]
    # out = F.grid_sample(
    #     value, sample_points.float(),
    #     mode="bilinear", padding_mode="zeros", align_corners=True,
    # )
    # out = torch.zeros(B, C, n_queries, n_points, device=sample_points.device) # 160,64,40,36
    
    out = custom_grid_sample(
        value, sample_points.float(), align_corners=True
    )


    if weight is not None:
        weight = weight.view(B, n_queries, n_points).unsqueeze(1)
        out *= weight

    return out.permute(0, 2, 3, 1)

    # if weight is not None:
    #     weight = weight.view(B, 1, n_queries, n_points).permute(0, 2, 3, 1)
    #     weight = weight.expand(B, n_queries, n_points, C)
    #     out = out.permute(0, 2, 3, 1) * weight
    #     out = out.permute(0, 3, 1, 2)

    # return out.permute(0, 2, 3, 1)


def sampling_3d(
        sample_points: torch.Tensor,
        weight: torch.Tensor,
        multi_lvl_values,
):
    B, n_queries, n_points, _ = sample_points.shape #160,40,36,2
    B, C, _, _ = multi_lvl_values[0].shape

    num_levels = len(multi_lvl_values) # multi_lvl_values shapes [0]->60,64,14,25 [1]->160,64,28,50 [2]->160,64,56,100

    sample_points_xy = sample_points * 2.0 - 1.0

    sample_points_lvl_weight_list = weight.unbind(-1) # weight.shape 160,40,36,3

    out = sample_points.new_zeros(
        B, n_queries, n_points, C)

    for i in range(num_levels):
        value = multi_lvl_values[i]
        lvl_weights = sample_points_lvl_weight_list[i]

        out += sampling_each_level(sample_points_xy, value,
                                   weight=lvl_weights)

    return out
