import torch
from torch import nn    
import onnx
from onnxsim import simplify

import os
import argparse

from mmengine.config import Config
from srlane.models.registry import build_net
from srlane.utils.net_utils import load_network
# from srlane.models.backbones import CSPDarknet

import torch.nn.functional as F

import onnxruntime as ort
import numpy as np
from onnx2tf import convert

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", type=str, default=None,
                        help="Dirs for log and saving ckpts")
    parser.add_argument("--work_dirs", type=str, default="/media/data/hamza/lane_understanding/codes/SRLane/work_dirs/sr_mtv/5006_3",
                        help="Dirs for log and saving ckpts")
    parser.add_argument("--load_from", default="ckpt/478.pth",
                        help="The checkpoint file to load from")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # model = CSPDarknet(        
    #     deepen_factor=0.33,
    #     widen_factor=0.375,
    #     act_cfg=dict(type="LeakyReLU"),
    #     )
    # model = nn.Sequential(*list(model.children()),
    #                       nn.AdaptiveAvgPool2d((1, 1)),
    #                       nn.Flatten(),
    #                       nn.Linear(384,1))
    # model.eval()
    
    cfg = Config.fromfile(args.config)
    model = build_net(cfg)
    # class Model(nn.Module):
    #     def forward(self, x):
    #         angle = F.interpolate(x,
    #                             size=[4, 10],
    #                             mode="bilinear",
    #                             align_corners=True)
    #         if len(angle.size())==4 and angle.size(1) == 1:
    #         #    angle = angle.squeeze(1)
    #             B,_,H,W = angle.size()  
    #             angle = angle.reshape(B,H,W)
    #         return angle
    # model = Model()

    model.eval()
    # load_network(model, os.path.join(args.work_dirs,args.load_from), strict=False)
    print(model)
    # Create a dummy input tensor with the same shape as the input your model expects
    dummy_input = torch.randn(1, 3, 448, 800)  # Example for an image classification model

    # Define the path where the ONNX model will be saved
    onnx_model_dir = os.path.join(args.work_dirs,"exported")
    if not os.path.exists(onnx_model_dir):
        os.makedirs(onnx_model_dir)
    onnx_model_path = os.path.join(onnx_model_dir,"model.onnx")

    # Export the model
    torch.onnx.export(
        model,                # The model to be exported
        dummy_input,          # The dummy input tensor
        onnx_model_path,      # The path where the ONNX model will be saved
        export_params=True,   # Store the trained parameter weights inside the model file
        # keep_initializers_as_inputs=True,
        opset_version=11,     # The ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=['input'],     # The model's input names
        output_names= ['output'], ##['preds','cls_preds','meta'],   # The model's output names
        dynamic_axes= None ##{'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic axes
    )
    print(f"Model has been converted to ONNX and saved at {onnx_model_path}")

    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    model_simplified, check = simplify(model)
    if check:
        print("ONNX model simplification successful!")
    else:
        print("ONNX model simplification failed!")

    # Save the simplified model
    simplified_model_path = os.path.join(onnx_model_dir,"model_simplified.onnx")
    onnx.save(model_simplified, simplified_model_path)
    print(f"Simplified ONNX model saved at {simplified_model_path}")
    # # # Convert the ONNX model to TensorFlow using onnx-tf
    # # from onnx_tf.backend import prepare
    # # tf_rep = prepare(model)
    # # tf_model_path = os.path.join(onnx_model_dir, "model_tf")
    # # tf_rep.export_graph(tf_model_path)

    # # # Load the TensorFlow model
    # # import tensorflow as tf
    # # tf_model = tf.saved_model.load(tf_model_path)

    # # # Run inference using TensorFlow
    # # input_tensor = tf.constant(np.random.randn(1, 3, 448, 800).astype(np.float32))
    # # output_tensor = tf_model.signatures["serving_default"](input_tensor)

    # # # Print the output
    # # print(output_tensor.shape)

    # convert(
    #     input_onnx_file_path=onnx_model_path,
    #     # output_nms_with_dynamic_tensor=False,
    #     output_signaturedefs=True,
    #     output_folder_path=onnx_model_dir,
    #     # output_tfv1_pb=True,
    #     # copy_onnx_input_output_names_to_tflite=True,
    #     non_verbose=False,
    #     # disable_group_convolution=True,  # Disable GroupConvolution optimization
    # )

    # # Load the ONNX model
    # model = onnx.load(onnx_model_path)
    
    # # Check the model
    # print("INTEGRITY CHECK")
    # onnx.checker.check_model(model)

    # # Set the ir_version manually if not set
    # if not model.ir_version:
    #     model.ir_version = 6  # Set to a valid ir_version

    # # Save the model again after setting ir_version
    # onnx.save(model, onnx_model_path)

    # Check the model
    print("INTEGRITY CHECK")
    onnx.checker.check_model(model)

    session = ort.InferenceSession(simplified_model_path)

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
    print(input_name)
    print(output_name)
if __name__ == "__main__":
    main()
