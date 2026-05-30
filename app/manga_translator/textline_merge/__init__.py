import itertools
from collections import Counter
from typing import List, Set

import networkx as nx
import numpy as np

from ..utils import Quadrilateral, TextBlock, quadrilateral_can_merge_region


def split_text_region(
    bboxes: List[Quadrilateral],
    connected_region_indices: Set[int],
    width,
    height,
    gamma=0.5,
    sigma=2,
) -> List[Set[int]]:
    connected_region_indices = list(connected_region_indices)

    # case 1
    if len(connected_region_indices) == 1:
        return [set(connected_region_indices)]

    # case 2
    if len(connected_region_indices) == 2:
        fs1 = bboxes[connected_region_indices[0]].font_size
        fs2 = bboxes[connected_region_indices[1]].font_size
        fs = max(fs1, fs2)

        if (
            bboxes[connected_region_indices[0]].distance(bboxes[connected_region_indices[1]]) < (1 + gamma) * fs
            and abs(bboxes[connected_region_indices[0]].angle - bboxes[connected_region_indices[1]].angle) < 0.2 * np.pi
        ):
            return [set(connected_region_indices)]
        else:
            return [set([connected_region_indices[0]]), set([connected_region_indices[1]])]

    # case 3
    G = nx.Graph()
    for idx in connected_region_indices:
        G.add_node(idx)
    for u, v in itertools.combinations(connected_region_indices, 2):
        G.add_edge(u, v, weight=bboxes[u].distance(bboxes[v]))
    # Get distances from neighboring bboxes
    edges = nx.algorithms.tree.minimum_spanning_edges(G, algorithm="kruskal", data=True)
    edges = sorted(edges, key=lambda a: a[2]["weight"], reverse=True)
    distances_sorted = [a[2]["weight"] for a in edges]
    fontsize = np.mean([bboxes[idx].font_size for idx in connected_region_indices])
    distances_std = np.std(distances_sorted)
    distances_mean = np.mean(distances_sorted)
    std_threshold = max(0.3 * fontsize + 5, 5)

    b1, b2 = bboxes[edges[0][0]], bboxes[edges[0][1]]
    max_poly_distance = b1.poly_distance(b2)
    max_centroid_alignment = min(abs(b1.centroid[0] - b2.centroid[0]), abs(b1.centroid[1] - b2.centroid[1]))

    if (distances_sorted[0] <= distances_mean + distances_std * sigma or distances_sorted[0] <= fontsize * (1 + gamma)) and (
        distances_std < std_threshold or max_poly_distance == 0 and max_centroid_alignment < 5
    ):
        return [set(connected_region_indices)]
    else:
        G = nx.Graph()
        for idx in connected_region_indices:
            G.add_node(idx)
        # Split out the most deviating bbox
        for edge in edges[1:]:
            G.add_edge(edge[0], edge[1])
        ans = []
        for node_set in nx.algorithms.components.connected_components(G):
            ans.extend(split_text_region(bboxes, node_set, width, height))
        return ans


def merge_bboxes_text_region(bboxes: List[Quadrilateral], width, height):
    # step 1: divide into multiple text region candidates
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, box=box)

    for (u, ubox), (v, vbox) in itertools.combinations(enumerate(bboxes), 2):
        if quadrilateral_can_merge_region(
            ubox,
            vbox,
            aspect_ratio_tol=1.3,
            font_size_ratio_tol=2,
            char_gap_tolerance=1,
            char_gap_tolerance2=3,
        ):
            G.add_edge(u, v)

    # step 2: postprocess - further split each region
    region_indices: List[Set[int]] = []
    for node_set in nx.algorithms.components.connected_components(G):
        region_indices.extend(split_text_region(bboxes, node_set, width, height))

    # step 3: return regions
    for node_set in region_indices:
        nodes = list(node_set)
        txtlns: List[Quadrilateral] = [bboxes[node] for node in nodes]

        # calculate average fg and bg color in a single pass
        fg_sum_r = fg_sum_g = fg_sum_b = 0
        bg_sum_r = bg_sum_g = bg_sum_b = 0
        for box in txtlns:
            fg_sum_r += box.fg_r
            fg_sum_g += box.fg_g
            fg_sum_b += box.fg_b
            bg_sum_r += box.bg_r
            bg_sum_g += box.bg_g
            bg_sum_b += box.bg_b
        num_boxes = len(txtlns)
        fg_r = round(fg_sum_r / num_boxes)
        fg_g = round(fg_sum_g / num_boxes)
        fg_b = round(fg_sum_b / num_boxes)
        bg_r = round(bg_sum_r / num_boxes)
        bg_g = round(bg_sum_g / num_boxes)
        bg_b = round(bg_sum_b / num_boxes)

        # majority vote for direction
        dirs = [box.direction for box in txtlns]
        majority_dir_top_2 = Counter(dirs).most_common(2)
        if len(majority_dir_top_2) == 1:
            majority_dir = majority_dir_top_2[0][0]
        elif majority_dir_top_2[0][1] == majority_dir_top_2[1][1]:  # if top 2 have the same counts
            max_ratio = -1.0
            for box in txtlns:
                ratio = max(box.aspect_ratio, 1.0 / (box.aspect_ratio + 1e-8))
                if ratio > max_ratio:
                    max_ratio = ratio
                    majority_dir = box.direction
        else:
            majority_dir = majority_dir_top_2[0][0]

        # sort textlines
        if majority_dir == "h":
            nodes = sorted(nodes, key=lambda x: bboxes[x].centroid[1])
        elif majority_dir == "v":
            nodes = sorted(nodes, key=lambda x: -bboxes[x].centroid[0])
        txtlns = [bboxes[node] for node in nodes]

        # yield overall bbox and sorted indices
        yield txtlns, (fg_r, fg_g, fg_b), (bg_r, bg_g, bg_b)


async def dispatch(textlines: List[Quadrilateral], width: int, height: int, verbose: bool = False) -> List[TextBlock]:
    text_regions: List[TextBlock] = []
    for txtlns, fg_color, bg_color in merge_bboxes_text_region(textlines, width, height):
        total_logprobs = 0
        for txtln in txtlns:
            total_logprobs += np.log(txtln.prob) * txtln.area
        total_logprobs /= sum([txtln.area for txtln in txtlns])

        font_size = int(min([txtln.font_size for txtln in txtlns]))
        angle = np.rad2deg(np.mean([txtln.angle for txtln in txtlns])) - 90
        if abs(angle) < 3:
            angle = 0
        lines = [txtln.pts for txtln in txtlns]
        texts = [txtln.text for txtln in txtlns]
        region = TextBlock(
            lines,
            texts,
            font_size=font_size,
            angle=angle,
            prob=np.exp(total_logprobs),
            fg_color=fg_color,
            bg_color=bg_color,
        )
        text_regions.append(region)
    return text_regions
