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
    model = build_net(cfg)

    # model_dir = os.path.join(args.work_dirs,args.load_from)
    # weights = torch.load(model_dir, map_location=torch.device('cpu'))["net"]
    # new_weights = {}
    # for k, v in weights.items():
    #     new_k = k.replace("module.", '') if "module" in k else k
    #     new_weights[new_k] = v
    # model.load_state_dict(new_weights, strict=False,)

    model.eval()
    load_network(model, os.path.join(args.work_dirs,args.load_from), strict=False)

    # model.cuda()
    # load_network(model, model_dir, strict=False)

    print(model)
    # model = torch.compile(model)
    # print(model)

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

    # model_simplified = replace_squeeze_with_reshape(model_simplified)
    
    # model_simplified = fix_node_names(model_simplified)

    # # Iterate through nodes and modify Gather indices if necessary
    # for node in model_simplified.graph.node:
    #     if node.op_type == "Gather":
    #         for input in node.input:
    #             # Find the tensor in the initializer
    #             tensor = next((t for t in model_simplified.graph.initializer if t.name == input), None)
    #             if tensor and tensor.data_type == onnx.TensorProto.INT64:
    #                 # Convert the tensor data to int32 while preserving the values
    #                 tensor_array = numpy_helper.to_array(tensor)
    #                 print(f"Original tensor (int64): {tensor_array}")

    #                 # Ensure the data type is converted correctly
    #                 tensor_array = tensor_array.astype(np.int32)
    #                 print(f"Converted tensor (int32): {tensor_array}")

    #                 # Update the tensor in the initializer
    #                 new_tensor = numpy_helper.from_array(tensor_array, name=tensor.name)
    #                 model_simplified.graph.initializer.remove(tensor)
    #                 model_simplified.graph.initializer.append(new_tensor)

    # # Save the modified ONNX model
    # onnx.save(model, simplified_model_path)

    # for initializer in model_simplified.graph.initializer:
    #     if initializer.name == "onnx::Gather_1576":  # Replace with the actual tensor name
    #         tensor_array = numpy_helper.to_array(initializer)
    #         print(f"Original shape: {tensor_array.shape}")

    #         # Reshape the tensor to the expected shape
    #         reshaped_array = tensor_array.reshape((36,))
    #         new_initializer = numpy_helper.from_array(reshaped_array, name=initializer.name)

    #         # Replace the initializer in the model
    #         model_simplified.graph.initializer.remove(initializer)
    #         model_simplified.graph.initializer.append(new_initializer)

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
    
    ## onnx2tf
    convert(
        # input_onnx_file_path=simplified_model_path,
        input_onnx_file_path=simplified_model_path,
        # output_nms_with_dynamic_tensor=False,
        output_signaturedefs=True,
        output_folder_path=onnx_model_dir,
        output_tfv1_pb=True,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,
        disable_group_convolution=True,  # Disable GroupConvolution optimization
        # replace_gather_with_gathernd=True,  # Replace Gather with GatherND to avoid issues
        # replace_argmax_with_max_and_indicies=True,  # Replace ArgMax to avoid INT64 outputs
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
