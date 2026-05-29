import hashlib
import os
import re
import sys
import unicodedata
from typing import Callable, List, Optional, Tuple

import cv2
import einops
import numpy as np
import requests
import tqdm
from PIL import Image
from shapely.geometry import MultiPoint, Polygon

# Resolve BASE_PATH
if getattr(sys, "frozen", False):
    # Packaged production mode
    BASE_PATH = os.path.dirname(sys.executable)
else:
    # Development mode
    MODULE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    BASE_PATH = os.path.dirname(MODULE_PATH)


# Context class
class Context(dict):
    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __getattr__(self, item):
        return self.get(item)

    def __delattr__(self, key) -> None:
        return self.__delitem__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def __getstate__(self):
        return self.copy()

    def __setstate__(self, state):
        self.update(state)

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        return dict(self) == dict(other)

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        for name, value in self.items():
            arg_strings.append("%s=%r" % (name, value))
        return "%s(%s)" % (type_name, ", ".join(arg_strings))


# Essential text helpers
def is_punctuation(ch):
    cp = ord(ch)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return True
    return False


def is_whitespace(ch):
    if ch == " " or ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) == 0:
        return True
    cat = unicodedata.category(ch)
    if cat == "Zs":
        return True
    return False


def is_control(ch):
    if ch == "\t" or ch == "\n" or ch == "\r":
        return False
    cat = unicodedata.category(ch)
    if cat in ("Cc", "Cf"):
        return True
    return False


def is_valuable_char(ch):
    return not is_punctuation(ch) and not is_control(ch) and not is_whitespace(ch) and not ch.isdigit()


def is_valuable_text(text):
    for ch in text:
        if is_valuable_char(ch):
            return True
    return False


def count_valuable_text(text: str) -> int:
    return sum([1 for ch in text if is_valuable_char(ch)])


def repeating_sequence(s: str):
    """Extracts repeating sequence from string. Example: 'abcabca' -> 'abc'."""
    for i in range(1, len(s) // 2 + 1):
        seq = s[:i]
        if seq * (len(s) // len(seq)) + seq[: len(s) % len(seq)] == s:
            return seq
    return s


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def __call__(self, val=None):
        if val is not None:
            self.sum += val
            self.count += 1
        if self.count > 0:
            return self.sum / self.count
        else:
            return 0


def replace_prefix(s: str, old: str, new: str):
    if s.startswith(old):
        s = new + s[len(old) :]
    return s


def get_digest(file_path: str) -> str:
    h = hashlib.sha256()
    BUF_SIZE = 65536
    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(BUF_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_filename_from_url(url: str, default: str = "") -> str:
    m = re.search(r"/([^/?]+)[^/]*$", url)
    if m:
        return m.group(1)
    return default


def download_url_with_progressbar(url: str, path: str):
    if os.path.basename(path) in (".", "") or os.path.isdir(path):
        new_filename = get_filename_from_url(url)
        if not new_filename:
            raise Exception("Could not determine filename")
        path = os.path.join(path, new_filename)

    headers = {}
    downloaded_size = 0
    if os.path.isfile(path):
        downloaded_size = os.path.getsize(path)
        headers["Range"] = "bytes=%d-" % downloaded_size
        headers["Accept-Encoding"] = "deflate"

    r = requests.get(url, stream=True, allow_redirects=True, headers=headers)
    if downloaded_size and r.headers.get("Accept-Ranges") != "bytes":
        print("Error: Webserver does not support partial downloads. Restarting from the beginning.")
        r = requests.get(url, stream=True, allow_redirects=True)
        downloaded_size = 0
    total = int(r.headers.get("content-length", 0))
    chunk_size = 1024

    if r.ok:
        with tqdm.tqdm(
            desc=os.path.basename(path),
            initial=downloaded_size,
            total=total + downloaded_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar:
            with open(path, "ab" if downloaded_size else "wb") as f:
                is_tty = sys.stdout.isatty()
                downloaded_chunks = 0
                for data in r.iter_content(chunk_size=chunk_size):
                    size = f.write(data)
                    bar.update(size)
                    downloaded_chunks += 1
                    if not is_tty and downloaded_chunks % 1000 == 0:
                        print(bar)
    else:
        raise Exception(f'Couldn\'t resolve url: "{url}" (Error: {r.status_code})')


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:
        return 0


def load_image(img: Image.Image) -> Tuple[np.ndarray, Optional[Image.Image]]:
    if img.mode == "RGBA":
        img.load()
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha_ch = img.split()[3]
        background.paste(img, mask=alpha_ch)
        return np.array(background), alpha_ch
    elif img.mode == "P":
        img = img.convert("RGBA")
        img.load()
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha_ch = img.split()[3]
        background.paste(img, mask=alpha_ch)
        return np.array(background), alpha_ch
    else:
        return np.array(img.convert("RGB")), None


class BBox(object):
    def __init__(self, x: int, y: int, w: int, h: int, text: str, prob: float, fg_r=0, fg_g=0, fg_b=0, bg_r=0, bg_g=0, bg_b=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.prob = prob
        self.fg_r = fg_r
        self.fg_g = fg_g
        self.fg_b = fg_b
        self.bg_r = bg_r
        self.bg_g = bg_g
        self.bg_b = bg_b

    def width(self):
        return self.w

    def height(self):
        return self.h

    def to_points(self):
        return (
            np.array([self.x, self.y]),
            np.array([self.x + self.w, self.y]),
            np.array([self.x + self.w, self.y + self.h]),
            np.array([self.x, self.y + self.h]),
        )

    @property
    def xywh(self):
        return np.array([self.x, self.y, self.w, self.h], dtype=np.int32)


def sort_pnts(pts: np.ndarray):
    if isinstance(pts, list):
        pts = np.array(pts)
    assert isinstance(pts, np.ndarray) and pts.shape == (4, 2)
    pairwise_vec = (pts[:, None] - pts[None]).reshape((16, -1))
    pairwise_vec_norm = np.linalg.norm(pairwise_vec, axis=1)
    long_side_ids = np.argsort(pairwise_vec_norm)[[8, 10]]
    long_side_vecs = pairwise_vec[long_side_ids]
    inner_prod = (long_side_vecs[0] * long_side_vecs[1]).sum()
    if inner_prod < 0:
        long_side_vecs[0] = -long_side_vecs[0]
    struc_vec = np.abs(long_side_vecs.mean(axis=0))
    is_vertical = struc_vec[0] <= struc_vec[1]

    if is_vertical:
        pts = pts[np.argsort(pts[:, 1])]
        pts = pts[[*np.argsort(pts[:2, 0]), *np.argsort(pts[2:, 0])[::-1] + 2]]
        return pts, is_vertical
    else:
        pts = pts[np.argsort(pts[:, 0])]
        pts_sorted = np.zeros_like(pts)
        pts_sorted[[0, 3]] = sorted(pts[[0, 1]], key=lambda x: x[1])
        pts_sorted[[1, 2]] = sorted(pts[[2, 3]], key=lambda x: x[1])
        return pts_sorted, is_vertical


class Quadrilateral(object):
    def __init__(self, pts: np.ndarray, text: str, prob: float, fg_r=0, fg_g=0, fg_b=0, bg_r=0, bg_g=0, bg_b=0):
        self.pts, is_vertical = sort_pnts(pts)
        self.direction = "v" if is_vertical else "h"
        self.text = text
        self.prob = prob
        self.fg_r = fg_r
        self.fg_g = fg_g
        self.fg_b = fg_b
        self.bg_r = bg_r
        self.bg_g = bg_g
        self.bg_b = bg_b
        self.assigned_direction: str = None
        self.textlines: List[Quadrilateral] = []

    @property
    def fg_colors(self):
        return np.array([self.fg_r, self.fg_g, self.fg_b])

    @property
    def bg_colors(self):
        return np.array([self.bg_r, self.bg_g, self.bg_b])

    @property
    def structure(self) -> List[np.ndarray]:
        return [
            ((self.pts[0] + self.pts[1]) / 2).astype(int),
            ((self.pts[2] + self.pts[3]) / 2).astype(int),
            ((self.pts[1] + self.pts[2]) / 2).astype(int),
            ((self.pts[3] + self.pts[0]) / 2).astype(int),
        ]

    @property
    def aspect_ratio(self) -> float:
        struct = self.structure
        v1 = struct[1] - struct[0]
        v2 = struct[3] - struct[2]
        return np.linalg.norm(v2) / np.linalg.norm(v1)

    @property
    def font_size(self) -> float:
        struct = self.structure
        v1 = struct[1] - struct[0]
        v2 = struct[3] - struct[2]
        return min(np.linalg.norm(v1), np.linalg.norm(v2))

    def width(self) -> int:
        return self.aabb.w

    def height(self) -> int:
        return self.aabb.h

    @property
    def xyxy(self):
        return self.aabb.x, self.aabb.y, self.aabb.x + self.aabb.w, self.aabb.y + self.aabb.h

    def clip(self, width, height):
        self.pts[:, 0] = np.clip(np.round(self.pts[:, 0]), 0, width)
        self.pts[:, 1] = np.clip(np.round(self.pts[:, 1]), 0, height)

    @property
    def aabb(self) -> BBox:
        max_coord = np.max(self.pts, axis=0)
        min_coord = np.min(self.pts, axis=0)
        return BBox(
            int(min_coord[0]),
            int(min_coord[1]),
            int(max_coord[0] - min_coord[0]),
            int(max_coord[1] - min_coord[1]),
            self.text,
            self.prob,
            self.fg_r,
            self.fg_g,
            self.fg_b,
            self.bg_r,
            self.bg_g,
            self.bg_b,
        )

    def get_transformed_region(self, img, direction, textheight) -> np.ndarray:
        struct = self.structure
        v_vec = struct[1] - struct[0]
        h_vec = struct[3] - struct[2]
        ratio = np.linalg.norm(v_vec) / np.linalg.norm(h_vec)

        src_pts = self.pts.astype(np.int64).copy()
        im_h, im_w = img.shape[:2]

        x1, y1, x2, y2 = src_pts[:, 0].min(), src_pts[:, 1].min(), src_pts[:, 0].max(), src_pts[:, 1].max()
        x1 = np.clip(x1, 0, im_w)
        y1 = np.clip(y1, 0, im_h)
        x2 = np.clip(x2, 0, im_w)
        y2 = np.clip(y2, 0, im_h)
        img_cropped = img[y1:y2, x1:x2]

        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1

        self.assigned_direction = direction
        if direction == "h":
            h = max(int(textheight), 2)
            w = max(int(round(textheight / ratio)), 2)
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts.astype(np.float32), dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img_cropped, M, (w, h))
            return region
        elif direction == "v":
            w = max(int(textheight), 2)
            h = max(int(round(textheight * ratio)), 2)
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts.astype(np.float32), dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img_cropped, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return region

    @property
    def is_approximate_axis_aligned(self) -> bool:
        struct = self.structure
        v1 = struct[1] - struct[0]
        v2 = struct[3] - struct[2]
        e1 = np.array([0, 1], dtype=np.float32)
        e2 = np.array([1, 0], dtype=np.float32)
        unit_vector_1 = v1 / (np.linalg.norm(v1) + 1e-8)
        unit_vector_2 = v2 / (np.linalg.norm(v2) + 1e-8)
        if (
            abs(np.dot(unit_vector_1, e1)) < 0.05
            or abs(np.dot(unit_vector_1, e2)) < 0.05
            or abs(np.dot(unit_vector_2, e1)) < 0.05
            or abs(np.dot(unit_vector_2, e2)) < 0.05
        ):
            return True
        return False

    @property
    def cosangle(self) -> float:
        struct = self.structure
        v1 = struct[1] - struct[0]
        e2 = np.array([1, 0], dtype=np.float32)
        unit_vector_1 = v1 / (np.linalg.norm(v1) + 1e-8)
        return float(np.dot(unit_vector_1, e2))

    @property
    def angle(self) -> float:
        return float(np.fmod(np.arccos(np.clip(self.cosangle, -1.0, 1.0)) + np.pi, np.pi))

    @property
    def centroid(self) -> np.ndarray:
        return np.average(self.pts, axis=0)

    def distance_to_point(self, p: np.ndarray) -> float:
        d = 1.0e20
        for i in range(4):
            d = min(d, distance_point_point(p, self.pts[i]))
            d = min(d, distance_point_lineseg(p, self.pts[i], self.pts[(i + 1) % 4]))
        return d

    @property
    def polygon(self) -> Polygon:
        return MultiPoint([tuple(self.pts[0]), tuple(self.pts[1]), tuple(self.pts[2]), tuple(self.pts[3])]).convex_hull

    @property
    def area(self) -> float:
        return self.polygon.area

    def poly_distance(self, other) -> float:
        return self.polygon.distance(other.polygon)

    def distance(self, other, rho=0.5) -> float:
        return self.distance_impl(other, rho)

    def distance_impl(self, other, rho=0.5) -> float:
        pattern = ""
        if self.assigned_direction == "h":
            pattern = "h_left"
        else:
            pattern = "v_top"
        fs = max(self.font_size, other.font_size)
        if self.assigned_direction == "h":
            poly1 = MultiPoint([tuple(self.pts[0]), tuple(self.pts[3]), tuple(other.pts[0]), tuple(other.pts[3])]).convex_hull
            poly2 = MultiPoint([tuple(self.pts[2]), tuple(self.pts[1]), tuple(other.pts[2]), tuple(other.pts[1])]).convex_hull
            struct = self.structure
            other_struct = other.structure
            poly3 = MultiPoint(
                [
                    tuple(struct[0]),
                    tuple(struct[1]),
                    tuple(other_struct[0]),
                    tuple(other_struct[1]),
                ]
            ).convex_hull
            dist1 = poly1.area / (fs + 1e-8)
            dist2 = poly2.area / (fs + 1e-8)
            dist3 = poly3.area / (fs + 1e-8)
            if dist1 < fs * rho:
                pattern = "h_left"
            if dist2 < fs * rho and dist2 < dist1:
                pattern = "h_right"
            if dist3 < fs * rho and dist3 < dist1 and dist3 < dist2:
                pattern = "h_middle"
            if pattern == "h_left":
                return dist(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
            elif pattern == "h_right":
                return dist(self.pts[1][0], self.pts[1][1], other.pts[1][0], other.pts[1][1])
            else:
                return dist(struct[0][0], struct[0][1], other_struct[0][0], other_struct[0][1])
        else:
            poly1 = MultiPoint([tuple(self.pts[0]), tuple(self.pts[1]), tuple(other.pts[0]), tuple(other.pts[1])]).convex_hull
            poly2 = MultiPoint([tuple(self.pts[2]), tuple(self.pts[3]), tuple(other.pts[2]), tuple(other.pts[3])]).convex_hull
            dist1 = poly1.area / (fs + 1e-8)
            dist2 = poly2.area / (fs + 1e-8)
            if dist1 < fs * rho:
                pattern = "v_top"
            if dist2 < fs * rho and dist2 < dist1:
                pattern = "v_bottom"
            if pattern == "v_top":
                return dist(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
            else:
                return dist(self.pts[2][0], self.pts[2][1], other.pts[2][0], other.pts[2][1])

    def copy(self, new_pts: np.ndarray):
        return Quadrilateral(new_pts, self.text, self.prob, *self.fg_colors, *self.bg_colors)


def distance_point_point(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def distance_point_lineseg(p: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    x, y = p[0], p[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return np.sqrt(dx * dx + dy * dy)


def quadrilateral_can_merge_region(
    a: Quadrilateral,
    b: Quadrilateral,
    ratio=1.9,
    discard_connection_gap=2,
    char_gap_tolerance=0.6,
    char_gap_tolerance2=1.5,
    font_size_ratio_tol=1.5,
    aspect_ratio_tol=2,
) -> bool:
    b1 = a.aabb
    b2 = b.aabb
    char_size = min(a.font_size, b.font_size)
    x1, y1, w1, h1 = b1.x, b1.y, b1.w, b1.h
    x2, y2, w2, h2 = b2.x, b2.y, b2.w, b2.h
    p1 = Polygon(a.pts)
    p2 = Polygon(b.pts)
    dist = p1.distance(p2)
    if dist > discard_connection_gap * char_size:
        return False
    if max(a.font_size, b.font_size) / (char_size + 1e-8) > font_size_ratio_tol:
        return False
    if a.aspect_ratio > aspect_ratio_tol and b.aspect_ratio < 1.0 / aspect_ratio_tol:
        return False
    if b.aspect_ratio > aspect_ratio_tol and a.aspect_ratio < 1.0 / aspect_ratio_tol:
        return False
    a_aa = a.is_approximate_axis_aligned
    b_aa = b.is_approximate_axis_aligned
    if a_aa and b_aa:
        if dist < char_size * char_gap_tolerance:
            if abs(x1 + w1 // 2 - (x2 + w2 // 2)) < char_gap_tolerance2:
                return True
            if w1 > h1 * ratio and h2 > w2 * ratio:
                return False
            if w2 > h2 * ratio and h1 > w1 * ratio:
                return False
            if w1 > h1 * ratio or w2 > h2 * ratio:
                return (
                    abs(x1 - x2) < char_size * char_gap_tolerance2 or abs(x1 + w1 - (x2 + w2)) < char_size * char_gap_tolerance2
                )
            elif h1 > w1 * ratio or h2 > w2 * ratio:
                return (
                    abs(y1 - y2) < char_size * char_gap_tolerance2 or abs(y1 + h1 - (y2 + h2)) < char_size * char_gap_tolerance2
                )
            return False
        else:
            return False
    if True:
        if abs(a.angle - b.angle) < 15 * np.pi / 180:
            fs_a = a.font_size
            fs_b = b.font_size
            fs = min(fs_a, fs_b)
            if a.poly_distance(b) > fs * char_gap_tolerance2:
                return False
            if abs(fs_a - fs_b) / (fs + 1e-8) > 0.25:
                return False
            return True
    return False


def square_pad_resize(img: np.ndarray, tgt_size: int):
    h, w = img.shape[:2]
    pad_h, pad_w = 0, 0
    if w < h:
        pad_w = h - w
        w += pad_w
    elif h < w:
        pad_h = w - h
        h += pad_h
    pad_size = tgt_size - h
    if pad_size > 0:
        pad_h += pad_size
        pad_w += pad_size
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)
    down_scale_ratio = tgt_size / img.shape[0]
    assert down_scale_ratio <= 1
    if down_scale_ratio < 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_LINEAR)
    return img, down_scale_ratio, pad_h, pad_w


def det_rearrange_forward(
    img: np.ndarray,
    dbnet_batch_forward: Callable[[np.ndarray, str], Tuple[np.ndarray, np.ndarray]],
    tgt_size: int = 1280,
    max_batch_size: int = 4,
    device="cuda",
    verbose=False,
):
    def _unrearrange(patch_lst: List[np.ndarray], transpose: bool, channel=1, pad_num=0):
        _psize = _h = patch_lst[0].shape[-1]
        _step = int(ph_step * _psize / patch_size)
        _pw = int(_psize / pw_num)
        _h = int(_pw / w * h)
        tgtmap = np.zeros((channel, _h, _pw), dtype=np.float32)
        num_patches = len(patch_lst) * pw_num - pad_num
        for ii, p in enumerate(patch_lst):
            if transpose:
                p = einops.rearrange(p, "c h w -> c w h")
            for jj in range(pw_num):
                pidx = ii * pw_num + jj
                rel_t = rel_step_list[pidx]
                t = int(round(rel_t * _h))
                b = min(t + _psize, _h)
                l = jj * _pw
                r = l + _pw
                tgtmap[..., t:b, :] += p[..., : b - t, l:r]
                if pidx > 0:
                    interleave = _psize - _step
                    tgtmap[..., t : t + interleave, :] /= 2.0
                if pidx >= num_patches - 1:
                    break
        if transpose:
            tgtmap = einops.rearrange(tgtmap, "c h w -> c w h")
        return tgtmap[None, ...]

    def _patch2batches(patch_lst: List[np.ndarray], p_num: int, transpose: bool):
        if transpose:
            patch_lst = einops.rearrange(patch_lst, "(p_num pw_num) ph pw c -> p_num (pw_num pw) ph c", p_num=p_num)
        else:
            patch_lst = einops.rearrange(patch_lst, "(p_num pw_num) ph pw c -> p_num ph (pw_num pw) c", p_num=p_num)
        batches = [[]]
        for ii, patch in enumerate(patch_lst):
            if len(batches[-1]) >= max_batch_size:
                batches.append([])
            p, down_scale_ratio, pad_h, pad_w = square_pad_resize(patch, tgt_size=tgt_size)
            assert pad_h == pad_w
            pad_size = pad_h
            batches[-1].append(p)
        return batches, down_scale_ratio, pad_size

    h, w = img.shape[:2]
    transpose = False
    if h < w:
        transpose = True
        h, w = img.shape[1], img.shape[0]

    asp_ratio = h / w
    down_scale_ratio = h / tgt_size
    require_rearrange = down_scale_ratio > 2.5 and asp_ratio > 3
    if not require_rearrange:
        return None, None

    if transpose:
        img = einops.rearrange(img, "h w c -> w h c")

    pw_num = max(int(np.floor(2 * tgt_size / w)), 2)
    patch_size = ph = pw_num * w

    ph_num = int(np.ceil(h / ph))
    ph_step = int((h - ph) / (ph_num - 1)) if ph_num > 1 else 0
    rel_step_list = []
    patch_list = []
    for ii in range(ph_num):
        t = ii * ph_step
        b = t + ph
        rel_step_list.append(t / h)
        patch_list.append(img[t:b])

    p_num = int(np.ceil(ph_num / pw_num))
    pad_num = p_num * pw_num - ph_num
    for ii in range(pad_num):
        patch_list.append(np.zeros_like(patch_list[0]))

    batches, down_scale_ratio, pad_size = _patch2batches(patch_list, p_num, transpose)

    db_lst, mask_lst = [], []
    for batch in batches:
        batch = np.array(batch)
        db, mask = dbnet_batch_forward(batch, device=device)
        for d, m in zip(db, mask):
            if pad_size > 0:
                paddb = int(db.shape[-1] / tgt_size * pad_size)
                padmsk = int(mask.shape[-1] / tgt_size * pad_size)
                d = d[..., :-paddb, :-paddb]
                m = m[..., :-padmsk, :-padmsk]
            db_lst.append(d)
            mask_lst.append(m)

    db = _unrearrange(db_lst, transpose, channel=2, pad_num=pad_num)
    mask = _unrearrange(mask_lst, transpose, channel=1, pad_num=pad_num)
    return db, mask
