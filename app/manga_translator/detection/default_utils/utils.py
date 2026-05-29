import cv2
import numpy as np
import pyclipper
import torch
from shapely.geometry import Polygon


def adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys, dtype=np.float32)
        for sub_list in polys:
            sub_list[:, 0] *= ratio_w * ratio_net
            sub_list[:, 1] *= ratio_h * ratio_net
    return polys


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape
    target_size = mag_ratio * square_size
    ratio = target_size / max(height, width)

    target_h, target_w = int(round(height * ratio)), int(round(width * ratio))
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    mult = 256
    target_h32, target_w32 = target_h, target_w
    pad_h = 0
    pad_w = 0
    if target_h % mult != 0:
        pad_h = mult - target_h % mult
        target_h32 = target_h + pad_h
    if target_w % mult != 0:
        pad_w = mult - target_w % mult
        target_w32 = target_w + pad_w
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap, pad_w, pad_h


class SegDetectorRepresenter:
    def __init__(self, thresh=0.6, box_thresh=0.8, max_candidates=1000, unclip_ratio=2.2):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, batch, pred, is_output_polygon=False):
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        batch_size = pred.size(0) if isinstance(pred, torch.Tensor) else pred.shape[0]
        for batch_index in range(batch_size):
            height, width = batch["shape"][batch_index]
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[: self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        assert len(_bitmap.shape) == 2
        if isinstance(pred, torch.Tensor):
            bitmap = _bitmap.cpu().numpy()
            pred = pred.cpu().detach().numpy()
        else:
            bitmap = _bitmap
        height, width = bitmap.shape
        try:
            contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            return [], []
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int64)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - startidx, 0)
            box = np.array(box)
            boxes[index, :, :] = box.astype(np.int64)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.8):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / (poly.length + 1e-8)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]
