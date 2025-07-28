import os
import os.path as osp
from os.path import join

import json
import numpy as np
import pickle as pkl
from tqdm import tqdm

# import srlane.evaluation.motive_culane_metric as motive_metric
import srlane.evaluation.motive_updated_culane_metric as motive_metric
from .base_dataset_cls import BaseDatasetCls
from .registry import DATASETS
import cv2
from srlane.utils.visualization import COLORS

LIST_FILE = {
    "train": "image_sets/batch1_batch2/train.txt",
    "val": "image_sets/batch1_batch2/val.txt",
    "test": "image_sets/batch1_batch2/test.txt",
    # "train": "images.txt",
    # "val": "images.txt",
    # "test": "images.txt",
}

# LIST_FILE = {
#     "train": "imageSets/v11/train.txt",
#     "val": "imageSets/v11/test.txt",
#     "test": "imageSets/v11/test.txt",
# }

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

LINE_COLOR_MAP = {
    "yellow": 0,
    "white": 1,
    "unknown": 2,
    "none": 3
}

LINE_TYPE_MAP = {
    "solid": 0,
    "broken": 1,
    "unknown": 2,
    "none": 3
}

LINE_EGO_MAP = {
    "left": 0,
    "right": 1,
    "none": 2
}

LINE_CLASS_MAP = {
    "line": 0,
    "curb": 1,
}

LINE_GROUP_MAP = {
    "single": 0,
    "double": 1,
}

LINE_CURVATURE_MAP = {
    "straight": 0,
    "curve": 1,
}

LINE_DIRECTION_MAP = {
    "upstream": 0,
    "downstream": 1,
}

CURB_POSITION_MAP = {
    "left": 0,
    "right": 1,
    "none": 2
}


def imshow_lanes(img, lanes, line_color, line_type, line_ego, line_cls, line_group, line_curvature, line_direction, curb_position,  show=False, out_file=None, width=4):
    lanes_xys = []

    line_color_id_map = {v: k for k, v in LINE_COLOR_MAP.items()}
    line_type_id_map = {v: k for k, v in LINE_TYPE_MAP.items()}
    line_ego_id_map = {v: k for k, v in LINE_EGO_MAP.items()}
    line_cls_id_map = {v: k for k, v in LINE_CLASS_MAP.items()}
    line_group_id_map = {v: k for k, v in LINE_GROUP_MAP.items()}
    line_curvature_id_map = {v: k for k, v in LINE_CURVATURE_MAP.items()}
    line_direction_id_map = {v: k for k, v in LINE_DIRECTION_MAP.items()}
    curb_position_id_map = {v: k for k, v in CURB_POSITION_MAP.items()}

    for idx, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)
    # lanes_xys.sort(key=lambda xys: xys[0][0])

    # max_curb_position = [[0.48556924 0.5144308 ]
    #  [0.46596462 0.53403544]
    #  [0.6730305  0.32696947]
    #  [0.79286546 0.20713452]
    #  [0.7973991  0.2026009 ]
    #  [0.7782283  0.22177176]]

    # keep max values from each colum and set rest to 0
    # if len(curb_position) > 0:
    #     curb_position = np.array(curb_position)
    #     curb_position = np.where(curb_position == np.max(curb_position, axis=0), curb_position, 0)

    #     # convert it back to list of list
    #     curb_position = curb_position.tolist()

    xy_point = [20, 20]
    if len(lanes_xys) == 0:
        offset = 150
    else:
        offset = 1280 // len(lanes_xys)
    for idx, xys in enumerate(lanes_xys):
        if len(xys) < 2:
            continue
        for i in range(1, len(xys)):
            # plot point
            # cv2.circle(img, xys[i], 5, COLORS[idx], -1)
            cv2.circle(img, xys[i], 5, COLORS[1], -1)
        cv2.putText(img, str(idx), xys[-1], cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 2, cv2.LINE_AA)
            # cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)
        
        max_color = np.argmax(line_color[idx])
        max_type = np.argmax(line_type[idx])
        max_ego = np.argmax(line_ego[idx])
        max_cls = np.argmax(line_cls[idx])
        max_group = np.argmax(line_group[idx])
        max_curvature = np.argmax(line_curvature[idx])
        max_direction = np.argmax(line_direction[idx])
        max_curb_position = np.argmax(curb_position[idx])


        max_color_string = f"color: {line_color_id_map[max_color]}: {line_color[idx][max_color]:.2f}"
        max_type_string = f"type: {line_type_id_map[max_type]}: {line_type[idx][max_type]:.2f}"
        max_ego_string = f"ego: {line_ego_id_map[max_ego]}: {line_ego[idx][max_ego]:.2f}"
        max_cls_string = f"class: {line_cls_id_map[max_cls]}: {line_cls[idx][max_cls]:.2f}"
        max_group_string = f"group: {line_group_id_map[max_group]}: {line_group[idx][max_group]:.2f}"
        max_curvature_string = f"curvature: {line_curvature_id_map[max_curvature]}: {line_curvature[idx][max_curvature]:.2f}"
        max_direction_string = f"direction: {line_direction_id_map[max_direction]}: {line_direction[idx][max_direction]:.2f}"
        max_curb_position_string = f"curb_position: {curb_position_id_map[max_curb_position]}: {curb_position[idx][max_curb_position]:.2f}"

        font_size = 0.5

        cv2.putText(img, max_color_string, xy_point, cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[idx], 1, cv2.LINE_AA)
        cv2.putText(img, max_type_string, (xy_point[0], xy_point[1]+20), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[idx], 1, cv2.LINE_AA)
        cv2.putText(img, max_ego_string, (xy_point[0], xy_point[1]+40), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[idx], 1, cv2.LINE_AA)
        cv2.putText(img, max_cls_string, (xy_point[0], xy_point[1]+60), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[idx], 1, cv2.LINE_AA)
        cv2.putText(img, max_curvature_string, (xy_point[0], xy_point[1]+80), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[idx], 1, cv2.LINE_AA)
        cv2.putText(img, max_direction_string, (xy_point[0], xy_point[1]+100), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[idx], 1, cv2.LINE_AA)
        cv2.putText(img, max_group_string, (xy_point[0], xy_point[1]+120), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[idx], 1, cv2.LINE_AA)
        # cv2.putText(img, max_curb_position_string, (xy_point[0], xy_point[1]+140), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[idx], 1, cv2.LINE_AA)
        
        # if curb_position[idx][max_curb_position] > 0.1:
        if max_cls == 1:
            cv2.putText(img, max_curb_position_string, (xy_point[0], xy_point[1]+140), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[idx], 1, cv2.LINE_AA)
        else:
            max_curb_position_string = f"curb_position: none: {0:.2f}"

        #update xy_point, move towards right of frame
        xy_point = [xy_point[0]+offset, xy_point[1]]

        # xy_point = xys[-1] if len(xys) < 10 else xys[9]

        # cv2.putText(img, max_color_string, xy_point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, max_type_string, (xy_point[0], xy_point[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, max_ego_string, (xy_point[0], xy_point[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, max_cls_string, (xy_point[0], xy_point[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, max_group_string, (xy_point[0], xy_point[1]+80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, max_curvature_string, (xy_point[0], xy_point[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, max_direction_string, (xy_point[0], xy_point[1]+120), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, max_curb_position_string, (xy_point[0], xy_point[1]+140), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # if len(xys) < 10:
        #     cv2.putText(img, max_color_string, xy_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_type_string, (xy_point[0]-10, xy_point[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_ego_string, (xy_point[0]-20, xy_point[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_cls_string, (xy_point[0]-30, xy_point[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_group_string, (xy_point[0]-40, xy_point[1]+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_curvature_string, (xy_point[0]-50, xy_point[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_direction_string, (xy_point[0]-60, xy_point[1]+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_curb_position_string, (xy_point[0]-70, xy_point[1]+140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # else:
        #     cv2.putText(img, max_color_string, xys[8], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_type_string, (xys[7][0], xys[7][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_ego_string, (xys[6][0], xys[6][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_cls_string, (xys[5][0], xys[5][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_group_string, (xys[4][0], xys[4][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_curvature_string, (xys[3][0], xys[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_direction_string, (xys[2][0], xys[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(img, max_curb_position_string, (xys[1][0], xys[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)



    if show:
        cv2.imshow("view", img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)


@DATASETS.register_module
class MotiveUpdatedCls(BaseDatasetCls):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = join(data_root, LIST_FILE[split])
        self.split = split
        self.load_annotations()
        self.h_samples = np.arange(0, 720, 8) / 720
    
    def view(self, predictions, cls_preds, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]

        for lanes, line_color, line_type, line_ego, line_cls, line_group, line_curvature, line_direction, curb_position, img_meta in zip(predictions, cls_preds['color'], cls_preds['type'], cls_preds['ego'], cls_preds['class'], cls_preds['group'], cls_preds['curvature'], cls_preds['direction'], cls_preds['curb_position'], img_metas):
            img_name = img_meta["img_name"]
            # img = cv2.imread(osp.join(self.data_root, img_name))
            img = cv2.imread(osp.join(self.data_root+"/images_resized", img_name))
            out_file = osp.join(self.cfg.work_dir, "visualization",
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(img_meta["img_size"]) for lane in lanes]
            imshow_lanes(img, lanes, line_color, line_type, line_ego, line_cls, line_group, line_curvature, line_direction, curb_position, out_file=out_file)

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

                exist_list = []
                lanes = []
                line_color = [] # white, yellow, unknown, none (4)
                line_type = [] #solid, broken, unknown, none (4)
                line_ego = [] #left, right, none (3)
                line_class = [] #curb, line (2)
                line_group = [] # yes, no (2)
                line_curvature = [] # straight, curve (2)
                line_direction = [] # upstream, downstream (2)
                curb_position = [] # left, right (2)


                infos["mask_path"] = None

                anno_path = join(self.data_root, "annotations_resized", img_line[:-3]+"json")
                if not osp.exists(anno_path):
                    continue
                with open(anno_path, 'r') as anno_file:
                    data = json.load(anno_file)
                    
                    lanes_data = data["lines"]
                    
                    if "lanes" in lanes_data.keys():
                        for k in lanes_data["lanes"]:
                            lane_points = []
                            if len(lanes_data["lanes"][k]["polyline"]):
                                exist_list.append(1)
                                for kpt in lanes_data["lanes"][k]["polyline"]:
                                    point = tuple([kpt[1], kpt[0]])
                                    lane_points.append(point)
                                
                                if lanes_data["lanes"][k]["color"] is None:
                                    line_color.append("none")
                                else:
                                    line_color.append(lanes_data["lanes"][k]["color"])
                                
                                if lanes_data["lanes"][k]["type"] is None:
                                    line_type.append("none")
                                else:
                                    line_type.append(lanes_data["lanes"][k]["type"])
                                
                                if lanes_data["lanes"][k]["is_ego_left"] == 'True':
                                    line_ego.append("left")
                                elif lanes_data["lanes"][k]["is_ego_right"] == 'True':
                                    line_ego.append("right")
                                else:
                                    line_ego.append("none")
                                
                                line_class.append('line')

                                if lanes_data["lanes"][k]["group_id"] is not None:
                                    line_group.append('double')
                                else:
                                    line_group.append('single')
                                
                                if lanes_data["lanes"][k]["is_curved"] == 'True':
                                    line_curvature.append("curve")
                                elif lanes_data["lanes"][k]["is_curved"] == 'False':
                                    line_curvature.append("straight")
                                else:
                                    line_curvature.append("none")
                                
                                line_direction.append(lanes_data["lanes"][k]["direction"])

                                curb_position.append("none")
                            
                                lanes.append(lane_points)
                            else:
                                exist_list.append(0)
                    
                    if "curbs" in lanes_data.keys():
                        # print('here')
                        for k in lanes_data["curbs"]:
                            lane_points = []
                            if len(lanes_data["curbs"][k]["polyline"]):
                                exist_list.append(1)
                                for kpt in lanes_data["curbs"][k]["polyline"]:
                                    point = tuple([kpt[1], kpt[0]])
                                    lane_points.append(point)
                                
                                line_color.append("none")
                                line_type.append("none")
                                
                                if lanes_data["curbs"][k]["is_ego_left"] == 'True':
                                    line_ego.append("left")
                                elif lanes_data["curbs"][k]["is_ego_right"] == 'True':
                                    line_ego.append("right")
                                else:
                                    line_ego.append("none")
                                
                                # print('here1')
                                line_class.append('curb')
                                # if line_class[-1] == 'curb':
                                #     print('curb')

                                if lanes_data["curbs"][k]["is_curved"] == 'True':
                                    line_curvature.append("curve")
                                elif lanes_data["curbs"][k]["is_curved"] == 'False':
                                    line_curvature.append("straight")
                                else:
                                    line_curvature.append("none")
                                
                                curb_position.append(lanes_data["curbs"][k]["position"])
                                line_group.append("none")
                                line_direction.append("none")
                            
                                lanes.append(lane_points)
                            else:
                                exist_list.append(0)


                    # lanes_data = data["lanes"]
                    
                    # exist_list = []
                    # lanes = []
                    # for lane in lanes_data:
                    #     lane_points = []
                    #     if len(lane):
                    #         exist_list.append(1)

                    #         for kpt in lane["polyline"]:
                    #             point = tuple([kpt[1], kpt[0]])
                    #             lane_points.append(point)
                                
                    #         lanes.append(lane_points)
                    #     else:
                    #         exist_list.append(0)
                            
                exist_list = np.array(exist_list)
                        
                lanes = [list(set(lane)) for lane in
                         lanes]  # remove duplicated points
                
                final_lanes = []
                line_clr_one_hot_vec = []
                line_typ_one_hot_vec = []
                line_ego_one_hot_vec = []
                line_cls_one_hot_vec = []
                line_group_one_hot_vec = []
                line_curvature_one_hot_vec = []
                line_direction_one_hot_vec = []
                curb_position_one_hot_vec = []
                # print(line_class)
                # print("lanes:", len(lanes))
                # print("line_color:", len(line_color))
                # print("line_type:", len(line_type))
                # print("line_ego:", len(line_ego))
                # print("line_class:", len(line_class))
                # print("line_group:", len(line_group))
                # print("line_curvature:", len(line_curvature))
                # print("line_direction:", len(line_direction))

                assert len(lanes) == len(line_color) == len(line_type) == len(line_ego) == len(line_class) == len(line_group) == len(line_curvature) == len(line_direction) == len(curb_position)


                for lane, clr, typ, ego, cls, grp, curv, dir, c_pos in zip(lanes, line_color, line_type, line_ego, line_class, line_group, line_curvature, line_direction, curb_position):
                    if len(lane) > 2:
                        final_lanes.append(lane)
                        line_clr_one_hot_vect = [0, 0, 0, 0] # white, yellow, unknown, none
                        line_typ_one_hot_vect = [0, 0, 0, 0] # solid, broken, unknown, none
                        line_ego_one_hot_vect = [0, 0, 0] # left, right, none
                        line_cls_one_hot_vect = [0, 0] # curb, line
                        line_group_one_hot_vect = [0, 0] # single, double
                        line_curvature_one_hot_vect = [0, 0] # straight, curve
                        line_direction_one_hot_vect = [0, 0] # upstream, downstream
                        curb_position_one_hot_vect = [0, 0, 0] # left, right

                        if clr not in LINE_COLOR_MAP:
                            clr = "none"
                        if typ not in LINE_TYPE_MAP:
                            typ = "none"
                        if ego not in LINE_EGO_MAP:
                            ego = "none"

                        line_clr_one_hot_vect[LINE_COLOR_MAP[clr]] = 1
                        line_typ_one_hot_vect[LINE_TYPE_MAP[typ]] = 1
                        line_ego_one_hot_vect[LINE_EGO_MAP[ego]] = 1

                        if cls in LINE_CLASS_MAP:
                            line_cls_one_hot_vect[LINE_CLASS_MAP[cls]] = 1
                        
                        if grp in LINE_GROUP_MAP:
                            line_group_one_hot_vect[LINE_GROUP_MAP[grp]] = 1
                        
                        if curv in LINE_CURVATURE_MAP:
                            line_curvature_one_hot_vect[LINE_CURVATURE_MAP[curv]] = 1
                        
                        if dir in LINE_DIRECTION_MAP:
                            line_direction_one_hot_vect[LINE_DIRECTION_MAP[dir]] = 1
                        
                        if c_pos in CURB_POSITION_MAP:
                            curb_position_one_hot_vect[CURB_POSITION_MAP[c_pos]] = 1

                        line_clr_one_hot_vec.append(line_clr_one_hot_vect)
                        line_typ_one_hot_vec.append(line_typ_one_hot_vect)
                        line_ego_one_hot_vec.append(line_ego_one_hot_vect)
                        line_cls_one_hot_vec.append(line_cls_one_hot_vect)
                        line_group_one_hot_vec.append(line_group_one_hot_vect)
                        line_curvature_one_hot_vec.append(line_curvature_one_hot_vect)
                        line_direction_one_hot_vec.append(line_direction_one_hot_vect)
                        curb_position_one_hot_vec.append(curb_position_one_hot_vect)

                # lanes = [lane for lane in lanes
                #          if
                #          len(lane) > 2]  # remove lanes with less than 2 points

                final_lanes = [sorted(lane, key=lambda x: x[1])
                         for lane in final_lanes]  # sort by y
                infos["lanes"] = final_lanes
                infos["lane_exist"] = np.array(exist_list)
                infos["line_color"] = line_clr_one_hot_vec
                infos["line_typ"] = line_typ_one_hot_vec
                infos["line_ego"] = line_ego_one_hot_vec
                infos["line_class"] = line_cls_one_hot_vec
                infos["line_group"] = line_group_one_hot_vec
                infos["line_curvature"] = line_curvature_one_hot_vec
                infos["line_direction"] = line_direction_one_hot_vec
                infos["curb_position"] = curb_position_one_hot_vec

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
