import logging
import os.path as osp

import cv2
import numpy as np
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer as DC

from .registry import DATASETS
from .process import Process
from srlane.utils.visualization import imshow_lanes


@DATASETS.register_module
class BaseDatasetCls(Dataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = "train" in split
        self.processes = Process(processes, cfg)

    def view(self, predictions, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta["img_name"]
            # img = cv2.imread(osp.join(self.data_root, img_name))
            img = cv2.imread(osp.join(self.data_root+"/images_resized", img_name))
            out_file = osp.join(self.cfg.work_dir, "visualization",
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(img_meta["img_size"]) for lane in lanes]
            imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)

    @staticmethod
    def imread(path, rgb=True):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = self.imread(data_info["img_path"])
        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({"img": img})

        if self.training:
            if self.cfg.cut_height != 0:
                new_lanes = []
                for i in sample["lanes"]:
                    lanes = []
                    for p in i:
                        lanes.append((p[0], p[1] - self.cfg.cut_height))
                    new_lanes.append(lanes)
                sample.update({"lanes": new_lanes})

        sample = self.processes(sample)
        meta = {"full_img_path": data_info["img_path"],
                "img_name": data_info["img_name"],
                "img_size": data_info.get("img_size",
                                          (self.cfg.ori_img_h,
                                           self.cfg.ori_img_w)),
                "img_cut_height": self.cfg.cut_height}
        
        gt_line_clr = np.zeros((self.cfg.max_lanes, 4), dtype=np.float32)
        gt_line_typ = np.zeros((self.cfg.max_lanes, 4), dtype=np.float32)
        gt_line_ego = np.zeros((self.cfg.max_lanes, 3), dtype=np.float32)
        gt_line_class = np.zeros((self.cfg.max_lanes, 2), dtype=np.float32)
        gt_line_group = np.zeros((self.cfg.max_lanes, 2), dtype=np.float32)
        gt_line_curvature = np.zeros((self.cfg.max_lanes, 2), dtype=np.float32)
        gt_line_direction = np.zeros((self.cfg.max_lanes, 2), dtype=np.float32)
        gt_curb_position = np.zeros((self.cfg.max_lanes, 3), dtype=np.float32)


        if len(data_info['line_color']) != 0:
            gt_line_clr[:len(data_info['line_color'])] = np.array(data_info['line_color'])
        
        if len(data_info['line_typ']) != 0:
            gt_line_typ[:len(data_info['line_typ'])] = np.array(data_info['line_typ'])
        
        if len(data_info['line_ego']) != 0:
            gt_line_ego[:len(data_info['line_ego'])] = np.array(data_info['line_ego'])
        
        if len(data_info['line_class']) != 0:
            gt_line_class[:len(data_info['line_class'])] = np.array(data_info['line_class'])
        
        if len(data_info['line_group']) != 0:
            gt_line_group[:len(data_info['line_group'])] = np.array(data_info['line_group'])
        
        if len(data_info['line_curvature']) != 0:
            gt_line_curvature[:len(data_info['line_curvature'])] = np.array(data_info['line_curvature'])
        
        if len(data_info['line_direction']) != 0:
            gt_line_direction[:len(data_info['line_direction'])] = np.array(data_info['line_direction'])
        
        if len(data_info['curb_position']) != 0:
            gt_curb_position[:len(data_info['curb_position'])] = np.array(data_info['curb_position'])
        
        sample.update({
            "gt_line_clr": gt_line_clr,
            "gt_line_typ": gt_line_typ,
            "gt_line_ego": gt_line_ego,
            "gt_line_class": gt_line_class,
            "gt_line_group": gt_line_group,
            "gt_line_curvature": gt_line_curvature,
            "gt_line_direction": gt_line_direction,
            "gt_curb_position": gt_curb_position
        })
        meta = DC(meta, cpu_only=True)
        sample.update({"meta": meta})

        return sample
