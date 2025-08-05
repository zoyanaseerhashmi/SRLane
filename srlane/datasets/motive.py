import os
import os.path as osp
from os.path import join

import json
import numpy as np
import pickle as pkl
from tqdm import tqdm

import srlane.evaluation.motive_culane_metric as motive_metric
from .base_dataset import BaseDataset
from .registry import DATASETS

LIST_FILE = {
    "train": "imageSets/v11/train.txt",
    "val": "imageSets/v11/test.txt",
    "test": "imageSets/v11/test.txt",
}

CATEGORYS = {
    "normal": "list/test_split/test0_normal.txt",
    "crowd": "list/test_split/test1_crowd.txt",
    "hlight": "list/test_split/test2_hlight.txt",
    "shadow": "list/test_split/test3_shadow.txt",
    "noline": "list/test_split/test4_noline.txt",
    "arrow": "list/test_split/test5_arrow.txt",
    "curve": "list/test_split/test6_curve.txt",
    "cross": "list/test_split/test7_cross.txt",
    "night": "list/test_split/test8_night.txt",
}


@DATASETS.register_module
class Motive(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = join(data_root, LIST_FILE[split])
        self.split = split
        self.load_annotations()
        self.h_samples = np.arange(0, 720, 8) / 720

    def load_annotations(self, diff_thr=15):
        self.logger.info("Loading Motive annotations...")
        os.makedirs(".cache", exist_ok=True)
        cache_path = f".cache/motive_{self.split}.pkl"
        if osp.exists(cache_path):
            with open(cache_path, "rb") as cache_file:
                self.data_infos = pkl.load(cache_file)
                self.max_lanes = max(
                    len(anno["lanes"]) for anno in self.data_infos)
                return

        self.data_infos = []
        with open(self.list_path) as list_file:
            for i, line in tqdm(enumerate(list_file)):
                infos = {}
                img_line = line.strip()
                img_path = join(self.data_root, "images_resized", img_line) 

                infos["img_name"] = img_line
                infos["img_path"] = img_path


                infos["mask_path"] = None

                anno_path = join(self.data_root, "annotations_resized", img_line[:-3]+"json")
                if not osp.exists(anno_path):
                    continue
                with open(anno_path, 'r') as anno_file:
                    data = json.load(anno_file)
                    lanes_data = data["lanes"]
                    
                    exist_list = []
                    lanes = []
                    for lane in lanes_data:
                        lane_points = []
                        if len(lane):
                            exist_list.append(1)

                            for kpt in lane["polyline"]:
                                point = tuple([kpt[1], kpt[0]])
                                lane_points.append(point)
                                
                            lanes.append(lane_points)
                        else:
                            exist_list.append(0)
                            
                exist_list = np.array(exist_list)
                        
                lanes = [list(set(lane)) for lane in
                         lanes]  # remove duplicated points
                lanes = [lane for lane in lanes
                         if
                         len(lane) > 2]  # remove lanes with less than 2 points

                lanes = [sorted(lane, key=lambda x: x[1])
                         for lane in lanes]  # sort by y
                infos["lanes"] = lanes
                infos["lane_exist"] = np.array(exist_list)

                self.data_infos.append(infos)

        with open(cache_path, "wb") as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def get_prediction_string(self, pred):
        ys = self.h_samples
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            lane_xs = xs[valid_mask] * self.cfg.ori_img_w
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join([
                f"{x:.5f} {y:.5f}" for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        self.logger.info("Generating CULane prediction output...")
        for idx, pred in enumerate(predictions):
            output_dir = join(
                output_basedir,
                "predictions",
                osp.dirname(self.data_infos[idx]["img_name"]))
            output_filename = osp.basename(
                self.data_infos[idx]["img_name"])[:-3] + "lines.txt"
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)

            with open(join(output_dir, output_filename),
                      'w') as out_file:
                out_file.write(output)
        # if self.split == "test":
        #     for cate, cate_file in CATEGORYS.items():
        #         culane_metric.eval_predictions(output_basedir,
        #                                        self.data_root,
        #                                        join(self.data_root, cate_file),
        #                                        iou_thresholds=[0.5],
        #                                        official=True)

        result = motive_metric.eval_predictions(join(output_basedir, "predictions"),
                                                self.data_root+"/annotations_resized",
                                                self.data_root+"/images_resized",
                                                self.list_path,
                                                iou_thresholds=[0.5],
                                                official=True)

        return result[0.5]["F1"]
