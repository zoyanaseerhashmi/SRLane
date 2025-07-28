import os.path as osp
import argparse
from functools import partial

import cv2
import json
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
from PIL import Image


def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img,
                 tuple(p1),
                 tuple(p2),
                 color=255,
                 thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640)):
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def continuous_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [
        LineString(lane).buffer(distance=width / 2., cap_style=1,
                                join_style=2).intersection(image)
        for lane in xs
    ]
    ys = [
        LineString(lane).buffer(distance=width / 2., cap_style=1,
                                join_style=2).intersection(image)
        for lane in ys
    ]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))
    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def culane_metric(pred,
                  anno,
                  ori_img_shape,
                  width=30,
                  iou_thresholds=[0.5],
                  official=True,
                  img_shape=(320, 800)):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]

    xy_factors = (img_shape[1] / ori_img_shape[1],
                  img_shape[0] / ori_img_shape[0])

    def resize_lane(lane):
        return [[x * xy_factors[0], y * xy_factors[1]] for x, y in lane]

    pred = list(map(resize_lane, pred))
    anno = list(map(resize_lane, anno))

    interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred],
                           dtype=object)  # (4, 50, 2)
    interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno],
                           dtype=object)  # (4, 50, 2)

    if official:
        ious = discrete_cross_iou(interp_pred,
                                  interp_anno,
                                  width=width,
                                  img_shape=img_shape)
    else:
        ious = continuous_cross_iou(interp_pred,
                                    interp_anno,
                                    width=width,
                                    img_shape=img_shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        _metric[thr] = [tp, fp, fn]
    return _metric


def load_motive_img_data(annotations_dir, file_list_path, images_dir=None, load_size=False):
    
    with open(file_list_path, 'r') as data_file:
        filenames = [x.strip() for x in data_file.readlines()]

    all_lanes = []
    for filename in filenames:
        anno_path = osp.join(annotations_dir, filename[:-3]+"json")
        with open(anno_path, 'r') as anno_file:
            data = json.load(anno_file)

            lanes_data = data["lines"]
            lanes = []
            
            if "lanes" in lanes_data.keys():
                for k in lanes_data["lanes"]:
                    lane_points = []
                    if len(lanes_data["lanes"][k]["polyline"]):
                        for kpt in lanes_data["lanes"][k]["polyline"]:
                            point = tuple([kpt[1], kpt[0]])
                            lane_points.append(point)
                        lanes.append(lane_points)
            
            if "curbs" in lanes_data.keys():
                for k in lanes_data["curbs"]:
                    lane_points = []
                    if len(lanes_data["curbs"][k]["polyline"]):
                        for kpt in lanes_data["curbs"][k]["polyline"]:
                            point = tuple([kpt[1], kpt[0]])
                            lane_points.append(point)
                        lanes.append(lane_points)

            # lanes_data = data["lanes"]

            # lanes = []
            # for lane in lanes_data:
            #     lane_points = []
            #     if len(lane):
            #         for kpt in lane["polyline"]:
            #             # point = tuple(kpt)
            #             point = tuple([kpt[1], kpt[0]])
            #             lane_points.append(point)
            #         lanes.append(lane_points)
        all_lanes.append(lanes)
        
    # img_data = [line.split() for line in img_data]
    # img_data = [list(map(float, lane)) for lane in img_data]
    # img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
    #             for lane in img_data]
    img_data = []
    for image_lanes in all_lanes:
        image_lanes = [list(set(lane)) for lane in image_lanes]  # remove duplicated points
        img_data.append([lane for lane in image_lanes if len(lane) >= 2])

    # img_data = [lane for lane in lanes for lanes in image_data if len(lane) >= 2]

    if load_size:
        with open(file_list_path, 'r') as file_list:
            img_paths = [
                osp.join(images_dir, line.rstrip())
                for line in file_list.readlines()
            ]
            ori_img_shape = [
                Image.open(img_path).size[::-1] for img_path in img_paths
            ]
        return data, ori_img_shape

    return img_data, None


def load_motive_predictions(data_dir, file_list_path, load_size=False):
    with open(file_list_path, 'r') as file_list:
        filepaths = [
            osp.join(data_dir, line[1 if line[0] == '/' else 0:]
                     .rstrip()[:-3]+"lines.txt")
            for line in file_list.readlines()
        ]
    data = []
    for path in filepaths:
        with open(path, 'r') as data_file:
            img_data = data_file.readlines()
        img_data = [line.split() for line in img_data]
        img_data = [list(map(float, lane)) for lane in img_data]
        img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                    for lane in img_data]
        img_data = [lane for lane in img_data if len(lane) >= 2]
        data.append(img_data)

    return data, None


def eval_predictions(pred_dir,
                     anno_dir,
                     imgs_dir,
                     list_path,
                     img_shape=(720, 1280),
                     iou_thresholds=[0.5],
                     width=30,
                     is_curvelanes=False,
                     official=True,
                     sequential=False):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating metric for List: {list_path}")
    predictions, _ = load_motive_predictions(pred_dir, list_path, load_size=False)
    annotations, ori_img_shape = load_motive_img_data(anno_dir, list_path, images_dir=imgs_dir,
                                                  load_size=is_curvelanes)
    if not is_curvelanes:
        ori_img_shape = [img_shape, ] * len(annotations)
    if sequential:
        results = map(
            partial(culane_metric,
                    width=width,
                    official=official,
                    iou_thresholds=iou_thresholds,
                    img_shape=img_shape),
            predictions, annotations, ori_img_shape)
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(culane_metric, zip(predictions, annotations,
                                                   ori_img_shape,
                                                   repeat(width),
                                                   repeat(iou_thresholds),
                                                   repeat(official),
                                                   repeat(img_shape)))

    mean_f1, mean_prec, mean_recall = 0, 0, 0
    total_tp, total_fp, total_fn = 0, 0, 0
    ret = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp != 0 else 0
        logger.info(f"iou thr: {thr:.2f}, tp: {tp}, fp: {fp}, fn: {fn},"
                    f"precision: {precision}, recall: {recall}, f1: {f1}")
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    if len(iou_thresholds) > 2:
        logger.info(f"mean result, total_tp: {total_tp}, "
                    f"total_fp: {total_fp}, total_fn: {total_fn},"
                    f"precision: {mean_prec}, recall: {mean_recall},"
                    f" f1: {mean_f1}")
        ret["mean"] = {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "Precision": mean_prec,
            "Recall": mean_recall,
            "F1": mean_f1
        }
    return ret


def main():
    args = parse_args()
    for list_path in args.list:
        results = eval_predictions(args.pred_dir,
                                   args.anno_dir,
                                   list_path,
                                   img_shape=tuple(args.shape),
                                   width=args.width,
                                   official=args.official,
                                   sequential=args.sequential)

        header = '=' * 20
        header += f" Results ({osp.basename(list_path)})"
        header += '=' * 20
        print(header)
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        print('=' * len(header))


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CULane's metric")
    parser.add_argument(
        "--pred_dir",
        help="Path to directory containing the predicted lanes",
        required=True)
    parser.add_argument(
        "--anno_dir",
        help="Path to directory containing the annotated lanes",
        required=True)
    parser.add_argument("--width",
                        type=int,
                        default=30,
                        help="Width of the lane")
    parser.add_argument("--list",
                        nargs='+',
                        help="Path to txt file containing the list of files",
                        required=True)
    parser.add_argument("--shape", nargs='+', type=int)
    parser.add_argument("--sequential",
                        action="store_true",
                        help="Run sequentially instead of in parallel")
    parser.add_argument("--official",
                        action="store_true",
                        help="Use official way to calculate the metric")

    return parser.parse_args()


if __name__ == "__main__":
    main()
