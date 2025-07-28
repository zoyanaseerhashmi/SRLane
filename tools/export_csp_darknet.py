import torch
from torch import nn    
import onnx
from onnxsim import simplify

import os
import argparse

# from mmengine.config import Config
# from srlane.models.registry import build_net
# from srlane.utils.net_utils import load_network
from srlane.models.backbones import CSPDarknet

import onnxruntime as ort
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--work_dirs", type=str, default="/media/data/hamza/lane_understanding/codes/SRLane/work_dirs/sr_mtv/5006_3",
                        help="Dirs for log and saving ckpts")
    parser.add_argument("--load_from", default="ckpt/478.pth",
                        help="The checkpoint file to load from")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = CSPDarknet(        
        deepen_factor=0.33,
        widen_factor=0.375,
        act_cfg=dict(type="LeakyReLU"),
        )
    model = nn.Sequential(*list(model.children()),
                          nn.AdaptiveAvgPool2d((1, 1)),
                          nn.Flatten(),
                          nn.Linear(384,1))
    model.eval()
    
    # load_network(model, os.path.join(args.work_dirs,args.load_from), strict=False)
    print(model)
    # Create a dummy input tensor with the same shape as the input your model expects
    dummy_input = torch.randn(1, 3, 448, 800)  # Example for an image classification model

    # Define the path where the ONNX model will be saved
    onnx_model_dir = os.path.join(args.work_dirs,"exported")
    if not os.path.exists(onnx_model_dir):
        os.makedirs(onnx_model_dir)
    onnx_model_path = os.path.join(onnx_model_dir,"backbone.onnx")

    # Export the model
    torch.onnx.export(
        model,                # The model to be exported
        dummy_input,          # The dummy input tensor
        onnx_model_path,      # The path where the ONNX model will be saved
        export_params=True,   # Store the trained parameter weights inside the model file
        opset_version=16,     # The ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=['input'],     # The model's input names
        output_names= ['output'], ##['preds','cls_preds','meta'],   # The model's output names
        dynamic_axes= None ##{'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic axes
    )

    print(f"Model has been converted to ONNX and saved at {onnx_model_path}")

    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    # Check the model
    print("INTEGRITY CHECK")
    onnx.checker.check_model(model)

    session = ort.InferenceSession(onnx_model_path)

    # Get the input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Create a dummy input tensor with the same shape as the input your model expects
    dummy_input = np.random.randn(1, 3, 448, 800).astype(np.float32)
    # dummy_input = np.random.randn(1, 3, 720, 1280).astype(np.float32)

    # Run inference
    outputs = session.run([output_name], {input_name: dummy_input})

    # Print the output
    print(outputs[0].shape)
if __name__ == "__main__":
    main()
