from typing import List

import numpy as np

from .textblock import TextBlock


def sort_regions(
    regions: List[TextBlock],
    right_to_left: bool = True,
    img: np.ndarray = None,
    force_simple_sort: bool = False,
) -> List[TextBlock]:
    """
    Sort list of text blocks.
    Purged of panel-detection and visualization.
    Uses smart dispersion standard-deviation sorting or a fallback simple coordinate sort.
    """
    if not regions:
        return []

    if force_simple_sort:
        return _simple_sort(regions, right_to_left)

    xs = [r.center[0] for r in regions]
    ys = [r.center[1] for r in regions]

    if len(regions) > 1:
        x_std = np.std(xs) if len(xs) > 1 else 0
        y_std = np.std(ys) if len(ys) > 1 else 0
        is_horizontal = x_std > y_std
    else:
        is_horizontal = False

    sorted_regions = []
    if is_horizontal:
        # Horizontal dispersion: sort by x primarily, then by y
        primary = sorted(regions, key=lambda r: -r.center[0] if right_to_left else r.center[0])
        group = []
        prev = None
        for r in primary:
            cx = r.center[0]
            if prev is not None and abs(cx - prev) > 20:
                group.sort(key=lambda r: r.center[1])
                sorted_regions += group
                group = []
            group.append(r)
            prev = cx
        if group:
            group.sort(key=lambda r: r.center[1])
            sorted_regions += group
    else:
        # Vertical dispersion: sort by y primarily, then by x
        primary = sorted(regions, key=lambda r: r.center[1])
        group = []
        prev = None
        for r in primary:
            cy = r.center[1]
            if prev is not None and abs(cy - prev) > 15:
                group.sort(key=lambda r: -r.center[0] if right_to_left else r.center[0])
                sorted_regions += group
                group = []
            group.append(r)
            prev = cy
        if group:
            group.sort(key=lambda r: -r.center[0] if right_to_left else r.center[0])
            sorted_regions += group

    return sorted_regions


def _simple_sort(regions: List[TextBlock], right_to_left: bool) -> List[TextBlock]:
    """
    A simple fallback sorting logic. Sorts regions from top to bottom,
    then by x-coordinate based on reading direction.
    """
    sorted_regions = []
    # Sort primarily by the y-coordinate of the center
    for region in sorted(regions, key=lambda r: r.center[1]):
        for i, sorted_region in enumerate(sorted_regions):
            # If the current region is clearly below a sorted region, continue
            if region.center[1] > sorted_region.xyxy[3]:
                continue
            # If the current region is clearly above a sorted region, it means we went too far
            if region.center[1] < sorted_region.xyxy[1]:
                sorted_regions.insert(i, region)
                break

            # y-center of the region is within the y-range of the sorted_region, so sort by x instead
            if right_to_left and region.center[0] > sorted_region.center[0]:
                sorted_regions.insert(i, region)
                break
            if not right_to_left and region.center[0] < sorted_region.center[0]:
                sorted_regions.insert(i, region)
                break
        else:
            # If the loop finishes without breaking, append the region to the end
            sorted_regions.append(region)

    return sorted_regions
