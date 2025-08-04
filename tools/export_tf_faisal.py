# import torch
# import tensorflow as tf
# import os
# import argparse
# from mmengine.config import Config
# from srlane.models.registry import build_net
# from srlane.utils.net_utils import load_network
# from torch2tf import torch2tf


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

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from onnx_tf.backend import prepare

import onnxruntime as ort
import numpy as np
from onnx import numpy_helper
from onnx import helper
from onnx2tf import convert



def parse_args():
    parser = argparse.ArgumentParser(description="Export a model to TensorFlow")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--work_dirs", type=str, default="/media/data/hamza/lane_understanding/codes/SRLane/work_dirs/sr_mtv/5006_3",
                        help="Dirs for log and saving ckpts")
    parser.add_argument("--load_from", default="ckpt/478.pth",
                        help="The checkpoint file to load from")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    ## onnx2tf
    # convert(
    #     # input_onnx_file_path=simplified_model_path,
    #     input_onnx_file_path="/media/data/hamza/lane_understanding/codes/SRLane/work_dirs/sr_mtv/5006_3/exported/model_simplified.onnx",
    #     # output_nms_with_dynamic_tensor=False,
    #     output_signaturedefs=True,
    #     output_folder_path=onnx_model_dir,
    #     output_tfv1_pb=True,
    #     copy_onnx_input_output_names_to_tflite=True,
    #     non_verbose=False,
    #     disable_group_convolution=True,  # Disable GroupConvolution optimization
    #     # replace_gather_with_gathernd=True,  # Replace Gather with GatherND to avoid issues
    #     # replace_argmax_with_max_and_indicies=True,  # Replace ArgMax to avoid INT64 outputs
    # )
    convert(
        input_onnx_file_path="/media/data/hamza/lane_understanding/codes/SRLane/work_dirs/sr_mtv/5006_3/exported/model_simplified.onnx",
        output_nms_with_dynamic_tensor=True,
        output_signaturedefs=True,
        output_folder_path=os.path.join("/media/data/hamza/lane_understanding/codes/SRLane/work_dirs/sr_mtv/5006_3/exported", "tmp_saved_model"),
        copy_onnx_input_output_names_to_tflite=True,
    )

    ## onnx-tf
    # convert_onnx_to_pb(simplified_model_path, onnx_model_dir)

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
