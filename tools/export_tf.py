import torch
import tensorflow as tf
import os
import argparse
from mmengine.config import Config
from srlane.models.registry import build_net
from srlane.utils.net_utils import load_network
from torch2tf import torch2tf

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
    # load_network(model, os.path.join(args.work_dirs, args.load_from), strict=False)
    # print(model)
    # Create a dummy input tensor with the same shape as the input your model expects
    dummy_input = torch.randn(1, 3, 448, 800)  # Example for an image classification model

    # Define the path where the TensorFlow model will be saved
    tf_model_dir = os.path.join(args.work_dirs, "exported")
    if not os.path.exists(tf_model_dir):
        os.makedirs(tf_model_dir)
    tf_model_path = os.path.join(tf_model_dir, "model_tf")

    # Convert the PyTorch model to TensorFlow
    tf_model = torch2tf(model, dummy_input)

    # Save the TensorFlow model
    tf.saved_model.save(tf_model, tf_model_path)
    print(f"TensorFlow model has been saved at {tf_model_path}")

if __name__ == "__main__":
    main()