from functools import cached_property
from typing import List, Tuple

import numpy as np
from shapely.geometry import MultiPoint, Polygon


class TextBlock(object):
    """
    Core data structure representing a merged block of translated text overlays.
    """

    def __init__(
        self,
        lines: List[Tuple[int, int, int, int]],
        texts: List[str] = None,
        translation: str = "",
        prob: float = 1.0,
        language: str = "unknown",
        font_size: float = -1,
        angle: float = 0,
        **kwargs,
    ) -> None:
        self.lines = np.array(lines, dtype=np.int32)
        self.language = language
        self.font_size = round(font_size)
        self.angle = angle
        self.prob = prob
        self.translation = translation

        self.texts = texts if texts is not None else []
        self.text = self.texts[0] if self.texts else ""
        if self.text and len(self.texts) > 1:
            for txt in self.texts[1:]:
                first_cjk = "\u3000" <= self.text[-1] <= "\u9fff"
                second_cjk = txt and ("\u3000" <= txt[0] <= "\u9fff")
                if first_cjk or second_cjk:
                    self.text += txt
                else:
                    self.text += " " + txt

    @cached_property
    def xyxy(self) -> np.ndarray:
        """Coordinates of the bounding box [x1, y1, x2, y2]."""
        x1 = self.lines[..., 0].min()
        y1 = self.lines[..., 1].min()
        x2 = self.lines[..., 0].max()
        y2 = self.lines[..., 1].max()
        return np.array([x1, y1, x2, y2]).astype(np.int32)

    @cached_property
    def xywh(self) -> np.ndarray:
        """Coordinates of the bounding box [x, y, w, h]."""
        x1, y1, x2, y2 = self.xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1]).astype(np.int32)

    @cached_property
    def center(self) -> np.ndarray:
        """Centroid of the text region [cx, cy]."""
        xyxy = self.xyxy
        return (xyxy[:2] + xyxy[2:]) / 2

    @cached_property
    def unrotated_polygons(self) -> np.ndarray:
        polygons = self.lines.reshape(-1, 8)
        if self.angle != 0:
            polygons = rotate_polygons(self.center, polygons, self.angle)
        return polygons

    @cached_property
    def min_rect(self) -> np.ndarray:
        polygons = self.unrotated_polygons
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        if self.angle != 0:
            min_bbox = rotate_polygons(self.center, min_bbox, -self.angle)
        return min_bbox.clip(0).reshape(-1, 4, 2).astype(np.int64)

    @property
    def polygon_object(self) -> Polygon:
        min_rect = self.min_rect[0]
        return MultiPoint([tuple(min_rect[0]), tuple(min_rect[1]), tuple(min_rect[2]), tuple(min_rect[3])]).convex_hull

    @property
    def area(self) -> float:
        return self.polygon_object.area

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


def rotate_polygons(center, polygons, rotation, new_center=None, to_int=True):
    if rotation == 0:
        return polygons
    if new_center is None:
        new_center = center
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)

    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    if to_int:
        return rotated.astype(np.int64)
    return rotated
