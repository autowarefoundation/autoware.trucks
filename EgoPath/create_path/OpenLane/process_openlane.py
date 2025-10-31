#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import warnings
import numpy as np
from tqdm import tqdm
from typing import Any
from PIL import Image, ImageDraw, ImageFont


# ============================= Format functions ============================= #


PointCoords = tuple[float, float]
ImagePointCoords = tuple[int, int]
Line = list[PointCoords] | list[ImagePointCoords]

def round_line_floats(
    line: Line, 
    ndigits: int = 3
):
    """
    Round the coordinates of a line to a specified number of decimal places.
    """

    line = list(line)
    for i in range(len(line)):
        line[i] = [
            round(line[i][0], ndigits),
            round(line[i][1], ndigits)
        ]
    line = tuple(line)

    return line


# Custom warning format
def custom_warning_format(
    message, 
    category, filename, 
    lineno, line = None
):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format


# Log skipped images
def log_skipped_image(
    log_json: dict,
    reason: str,
    image_path: str
):
    if (reason not in log_json):
        log_json[reason] = []
    log_json[reason].append(image_path)


# Annotate skipped images
def annotate_skipped_image(
    image: Image,
    reason: str,
    save_path: str,
    lanes: list[Line],
    egoleft: Line = None,
    egoright: Line = None,
    egopath: Line = None
):
    draw = ImageDraw.Draw(image)
    draw.text(
        (10, 10), 
        reason, 
        fill = (255, 0, 0),
        font = ImageFont.truetype("arial.ttf", 24)
    )
    for line in lanes:
        draw.line(line, fill = (255, 0, 0), width = 2)
    if (egoleft):
        draw.line(egoleft, fill = (0, 128, 0), width = 2)       # Green
    if (egoright):
        draw.line(egoright, fill = (0, 255, 255), width = 2)    # Cyan
    if (egopath):
        draw.line(egopath, fill = (255, 255, 0), width = 2)     # Yellow

    image.save(save_path)


# ============================== Helper functions ============================== #


def polyfitLine(
    line: Line, 
    num_points: int = 10,
    deg: int = 3
):
    """
    Polynomial fit a line so the algorithm knows the line shape.
    Should be cubic fit with 10 points by default.
    Outputted line should be sorted by descending y-coords.
    """
    if (len(line) < deg + 1):
        return line

    x_coords = [point[0] for point in line]
    y_coords = [point[1] for point in line]

    poly_coeffs = np.polyfit(
        y_coords, 
        x_coords, 
        deg = deg
    )
    poly_func = np.poly1d(poly_coeffs)

    y_min = min(y_coords)
    y_max = max(y_coords)
    y_new = np.linspace(
        y_min, 
        y_max, 
        num_points
    )
    x_new = poly_func(y_new)

    fitted_line = sorted(
        [
            (
                float(x_new[i]), 
                float(y_new[i])
            ) 
            for i in range(len(y_new))
        ], 
        key = lambda point: point[1], 
        reverse = True
    )

    return fitted_line


def normalizeCoords(
    line: Line, 
    width: int, 
    height: int
):
    """
    Normalize the coords of line points.
    """
    return [
        (x / width, y / height) 
        for x, y in line
    ]


def getLineAnchor(
    line: Line,
    verbose: bool = False
):
    """
    Determine "anchor" point of a line.
    """
    (x2, y2) = line[0]
    (x1, y1) = line[
        # int(len(line) / 5) 
        # if (
        #     len(line) > 5 and
        #     line[0][1] >= H * 0.8
        # ) else 1
        int(len(line) / 2)
    ]
    if (verbose):
        print(f"Anchor points chosen: ({x1}, {y1}), ({x2}, {y2})")

    if (x1 == x2) or (y1 == y2):
        return (x1, None, None)
    
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (H - b) / a
    if (verbose):
        print(f"Anchor point computed: (x0 = {x0}, a = {a}, b = {b})")

    return (x0, a, b)


def getDrivablePath(
    left_ego        : Line, 
    right_ego       : Line, 
    y_coords_interp : bool = False
):
    """
    Computes drivable path as midpoint between 2 ego lanes.
    """

    drivable_path = []

    # Interpolation among non-uniform y-coords
    if (y_coords_interp):

        left_ego = np.array(left_ego)
        right_ego = np.array(right_ego)
        y_coords_ASSEMBLE = np.unique(
            np.concatenate((
                left_ego[:, 1],
                right_ego[:, 1]
            ))
        )[::-1]
        left_x_interp = np.interp(
            y_coords_ASSEMBLE, 
            left_ego[:, 1][::-1], 
            left_ego[:, 0][::-1]
        )
        right_x_interp = np.interp(
            y_coords_ASSEMBLE, 
            right_ego[:, 1][::-1], 
            right_ego[:, 0][::-1]
        )
        mid_x = (left_x_interp + right_x_interp) / 2
        # Filter out those points that are not in the common vertical zone between 2 egos
        drivable_path = [
            (x, y) for x, y in list(zip(mid_x, y_coords_ASSEMBLE))
            if y <= min(left_ego[0][1], right_ego[0][1])
        ]

    else:
        # Get the normal drivable path from the longest common y-coords
        while (i <= len(left_ego) - 1 and j <= len(right_ego) - 1):
            if (left_ego[i][1] == right_ego[j][1]):
                drivable_path.append((
                    (left_ego[i][0] + right_ego[j][0]) / 2,     # Midpoint along x axis
                    left_ego[i][1]
                ))
                i += 1
                j += 1
            elif (left_ego[i][1] > right_ego[j][1]):
                i += 1
            else:
                j += 1

    # Extend drivable path to bottom edge of the frame
    if ((len(drivable_path) >= 2) and (drivable_path[0][1] < H - 1)):
        x1, y1 = drivable_path[
            int(len(drivable_path) / 5)
            if (
                len(drivable_path) > 2 and
                drivable_path[0][1] >= H * 4/5
            ) else -1
        ]
        x2, y2 = drivable_path[0]
        if (x2 == x1):
            x_bottom = x2
        else:
            a = (y2 - y1) / (x2 - x1)
            x_bottom = x2 + (H - 1 - y2) / a
        drivable_path.insert(0, (x_bottom, H - 1))

    # Drivable path only extends to the shortest ego line
    drivable_path = [
        (x, y) for x, y in drivable_path
        if y >= max(left_ego[-1][1], right_ego[-1][1])
    ]

    # Extend drivable path to be on par with longest ego line
    # # By making it parallel with longer ego line
    # y_top = min(
    #     left_ego[-1][1], 
    #     right_ego[-1][1]
    # )

    # if (
    #     (len(drivable_path) >= 2) and 
    #     (drivable_path[-1][1] > y_top)
    # ):
    #     sign_left_ego = left_ego[-1][0] - left_ego[-2][0]
    #     sign_right_ego = right_ego[-1][0] - right_ego[-2][0]
    #     sign_val = sign_left_ego * sign_right_ego

    #     # 2 egos going the same direction
    #     if (sign_val > 0):
    #         longer_ego = left_ego if left_ego[-1][1] < right_ego[-1][1] else right_ego
    #         if (
    #             (len(longer_ego) >= 2) and 
    #             (len(drivable_path) >= 2)
    #         ):
    #             x1, y1 = longer_ego[-1]
    #             x2, y2 = longer_ego[-2]
    #             if (x2 == x1):
    #                 x_top = drivable_path[-1][0]
    #             else:
    #                 a = (y2 - y1) / (x2 - x1)
    #                 x_top = drivable_path[-1][0] + (y_top - drivable_path[-1][1]) / a

    #             drivable_path.append((x_top, y_top))
        
    #     # 2 egos going opposite directions
    #     else:
    #         if (len(drivable_path) >= 2):
    #             x1, y1 = drivable_path[-1]
    #             x2, y2 = drivable_path[-2]

    #             if (x2 == x1):
    #                 x_top = x1
    #             else:
    #                 a = (y2 - y1) / (x2 - x1)
    #                 x_top = x1 + (y_top - y1) / a

    #             drivable_path.append((x_top, y_top))

    return drivable_path


# ============================== Core functions ============================== #


def parseData(
    json_data: dict[str: Any],
    lane_point_threshold: int = 20,
    verbose: bool = False
):
    """
    Parse raw JSON data from OpenLane dataset, then return a dict with:
        - "img_path"        : str, path to the image file.
        - "other_lanes"     : list of lanes [ (xi, yi) ] that are NOT ego lanes.
        - "egoleft_lane"    : egoleft lane in [ (xi, yi) ].
        - "egoright_lane"   : egoright lane in [ (xi, yi) ].
        - "drivable_path"   : drivable path in [ (xi, yi) ].

    Since each line in OpenLane has too many points, I implement `lane_point_threshold` 
    to determine approximately the maximum number of points allowed per lane.

    All coords are rounded to 2 decimal places (honestly we won't need more than that).
    All coords are NOT NORMALIZED (will do it right before saving to JSON).
    """

    img_path = json_data["file_path"]
    lane_lines = json_data["lane_lines"]
    all_lanes = []
    egoleft_lane = None
    egoright_lane = None
    other_lanes = []
    logs = []

    # Walk thru each lane
    for i, lane in enumerate(lane_lines):

        if not len(lane["uv"][0]) == len(lane["uv"][1]):
            if (verbose):
                warnings.warn(
                    f"Inconsistent number of U and V coords:\n \
                        - file_path  : {img_path}\n \
                        - lane_index : {i}\n \
                        - u          : {len(lane['uv'][0])}\n \
                        - v          : {len(lane['uv'][1])}"
                )
            logs.append(f"{i} : Inconsistent number of U and V coords |")
            continue

        if not (len(lane["uv"][0]) >= 10):
            if (verbose):
                warnings.warn(
                    f"Lane with insufficient points detected (less than 10 points). Ignored.\n \
                        - file_path  : {img_path}\n \
                        - lane_index : {i}\n \
                        - num_points : {len(lane['uv'][0])}"
                )
            logs.append(f"{i} : Lane with insufficient points detected |")
            continue

        # There are adjacent points with the same y-coords. Filtering em out.
        raw_lane = sorted(
            [
                (
                    int(lane["uv"][0][j]), 
                    int(lane["uv"][1][j])
                )
                for j in range(
                    0, 
                    len(lane["uv"][0]), 
                    (
                        1 if (len(lane['uv'][0]) < lane_point_threshold) 
                        else len(lane['uv'][0]) // lane_point_threshold
                    )
                )
            ],
            key = lambda x: x[1],
            reverse = True
        )
        this_lane = [raw_lane[0]] if raw_lane else []
        for point in raw_lane[1:]:
            if (point[1] != this_lane[-1][1]):
                this_lane.append(point)

        if (len(this_lane) < 2):
            if (verbose):
                warnings.warn(
                    f"Lane with insufficient unique y-coords detected (less than 2 points). Ignored.\n \
                        - file_path  : {img_path}\n \
                        - lane_index : {i}\n \
                        - num_points : {len(this_lane)}"
                )
            logs.append(f"{i} : Lane with insufficient unique y-coords detected |")
            continue

        # Polyfit line before adding anchor
        this_lane = polyfitLine(this_lane)
        
        # Add anchor to line, if needed
        if (this_lane and (this_lane[0][1] < H - 1)):
            this_lane.insert(0, (
                getLineAnchor(this_lane, verbose)[0],
                H - 1
            ))

        # Append to all lanes, with top-cropped y-coords
        all_lanes.append(this_lane)

        # this_attribute = lane["attribute"]

        # """
        # "attribute":    <int>: left-right attribute of the lane
        #                     1: left-left
        #                     2: left (exactly egoleft)
        #                     3: right (exactly egoright)
        #                     4: right-right
        # """
        # if (this_attribute == 2):
        #     if (egoleft_lane and verbose):
        #         warnings.warn(
        #             f"Multiple egoleft lanes detected. Please check! \n\
        #                 - file_path: {img_path}"
        #         )
        #     else:
        #         egoleft_lane = this_lane
        # elif (this_attribute == 3):
        #     if (egoright_lane and verbose):
        #         warnings.warn(
        #             f"Multiple egoright lanes detected. Please check! \n\
        #                 - file_path: {img_path}"
        #         )
        #     else:
        #         egoright_lane = this_lane
        # else:
        #     other_lanes.append(this_lane)

    # Sanity check if we have at least 2 lanes
    if (len(all_lanes) < 2):
        if (verbose):
            warnings.warn(f"Insufficient lanes detected. Ignored.\n \
                - file_path   : {img_path}\n \
                - total lanes : {len(all_lanes)}"
            )
        
        # Log skipped image
        reason = f"Insufficient lanes detected"
        true_img_path = os.path.join(dataset_dir, IMG_DIR, img_path)
        log_skipped_image(
            log_json = log_skipped_json,
            reason = reason,
            image_path = true_img_path
        )
        annotate_skipped_image(
            image = Image.open(true_img_path).convert("RGB"),
            lanes = all_lanes,
            reason = f"{reason} : only {len(all_lanes)} lanes",
            save_path = os.path.join(skipped_path, os.path.basename(true_img_path))
        )

        return None
    
    # Sort all lanes by their anchor x-coord
    all_lanes = sorted(
        all_lanes, 
        key = lambda lane: lane[0][0],
        reverse = False
    )
    
    # Determine 2 ego lanes by anchors instead
    for i, lane in enumerate(all_lanes):
        this_anchor = lane[0]
        if (this_anchor[0] >= W / 2):
            if (i == 0):
                egoleft_lane = all_lanes[0]
                egoright_lane = all_lanes[1]
                other_lanes = [
                    line for j, line in enumerate(all_lanes) 
                    if j != 0 and j != 1
                ]
            else:
                egoleft_lane = all_lanes[i - 1]
                egoright_lane = all_lanes[i]
                other_lanes = [
                    line for j, line in enumerate(all_lanes) 
                    if j != i - 1 and j != i
                ]
            break
        else:
            # Traversed all lanes but none is on the right half
            if (i == len(all_lanes) - 1):
                egoleft_lane = None
                egoright_lane = None

    if (egoleft_lane and egoright_lane):
        # Cut off longer ego line to match the shorter one
        if (egoleft_lane[-1][1] < egoright_lane[-1][1]):    # Left longer
            egoleft_lane = [
                point for point in egoleft_lane
                if point[1] >= egoright_lane[-1][1]
            ]
        elif (egoright_lane[-1][1] < egoleft_lane[-1][1]):  # Right longer
            egoright_lane = [
                point for point in egoright_lane
                if point[1] >= egoleft_lane[-1][1]
            ]

        drivable_path = getDrivablePath(
            left_ego = egoleft_lane,
            right_ego = egoright_lane,
            y_coords_interp = True
        )
    else:
        if (verbose):
            warnings.warn(f"Missing egolines detected: \n\
            - file_path: {img_path}")
        
        # Log skipped image
        if (not egoleft_lane and not egoright_lane):
            missing_line = "both"
        elif (not egoleft_lane):
            missing_line = "left"
        elif (not egoright_lane):
            missing_line = "right"
        reason = f"Missing egolines detected: {missing_line}"
        true_img_path = os.path.join(dataset_dir, IMG_DIR, img_path)
        log_skipped_image(
            log_json = log_skipped_json,
            reason = reason,
            image_path = true_img_path
        )
        annotate_skipped_image(
            image = Image.open(true_img_path).convert("RGB"),
            lanes = all_lanes,
            egoleft = egoleft_lane,
            egoright = egoright_lane,
            reason = reason,
            save_path = os.path.join(skipped_path, os.path.basename(true_img_path))
        )

        return None
    
    # Check drivable path validity
    THRESHOLD_EGOPATH_ANCHOR = 0.25
    THRESHOLD_LANE_WIDTH = 0.2

    if (len(drivable_path) < 2):
        if (verbose):
            warnings.warn(f"Drivable path with insufficient points detected (less than 2 points). Ignored.\n \
                - file_path  : {img_path}\n \
                - num_points : {len(drivable_path)}"
            )
        
        # Log skipped image
        reason = f"Drivable path with insufficient points"
        true_img_path = os.path.join(dataset_dir, IMG_DIR, img_path)
        log_skipped_image(
            log_json = log_skipped_json,
            reason = reason,
            image_path = true_img_path
        )
        annotate_skipped_image(
            image = Image.open(true_img_path).convert("RGB"),
            lanes = all_lanes,
            egoleft = egoleft_lane,
            egoright = egoright_lane,
            egopath = drivable_path,
            reason = f"{reason} : only {len(drivable_path)} points",
            save_path = os.path.join(skipped_path, os.path.basename(true_img_path))
        )

        return None
    
    elif not (
        THRESHOLD_EGOPATH_ANCHOR * W <= drivable_path[0][0] <= (1 - THRESHOLD_EGOPATH_ANCHOR) * W
    ):
        if (verbose):
            warnings.warn(f"Drivable path anchor too close to edge of frame. Ignored.\n \
                - file_path  : {img_path}\n \
                - anchor_x   : {drivable_path[0][0]}\n \
                - anchor_y   : {drivable_path[0][1]}"
            )
        
        # Log skipped image
        reason = f"Drivable path not in middle"
        true_img_path = os.path.join(dataset_dir, IMG_DIR, img_path)
        log_skipped_image(
            log_json = log_skipped_json,
            reason = reason,
            image_path = true_img_path
        )
        annotate_skipped_image(
            image = Image.open(true_img_path).convert("RGB"),
            lanes = all_lanes,
            egoleft = egoleft_lane,
            egoright = egoright_lane,
            egopath = drivable_path,
            reason = f"{reason} : drivable_path[0][0] = {drivable_path[0][0]}",
            save_path = os.path.join(skipped_path, os.path.basename(true_img_path))
        )

        return None
    
    elif not (
        (egoright_lane[0][0] - egoleft_lane[0][0]) >= THRESHOLD_LANE_WIDTH * W
    ):
        if (verbose):
            warnings.warn(f"Ego lane width too small. Ignored.\n \
                - file_path      : {img_path}\n \
                - lane_width    : {egoright_lane[0][0] - egoleft_lane[0][0]}"
            )
        
        # Log skipped image
        reason = f"Ego lane width too small"
        true_img_path = os.path.join(dataset_dir, IMG_DIR, img_path)
        log_skipped_image(
            log_json = log_skipped_json,
            reason = reason,
            image_path = true_img_path
        )
        annotate_skipped_image(
            image = Image.open(true_img_path).convert("RGB"),
            lanes = all_lanes,
            egoleft = egoleft_lane,
            egoright = egoright_lane,
            egopath = drivable_path,
            reason = f"{reason} : lane_width_bottom = {int(egoright_lane[0][0] - egoleft_lane[0][0])} < {int(THRESHOLD_LANE_WIDTH * W)}",
            save_path = os.path.join(skipped_path, os.path.basename(true_img_path))
        )

        return None

    elif not (
        (egoleft_lane[0][0] < drivable_path[0][0] < egoright_lane[0][0]) and
        (egoleft_lane[-1][0] < drivable_path[-1][0] < egoright_lane[-1][0])
    ):
        if (verbose):
            warnings.warn(f"Drivable path not between 2 egolanes. Ignored.\n \
                - file_path      : {img_path}\n \
                - drivable_path  : {drivable_path}\n \
                - egoleft_lane   : {egoleft_lane}\n \
                - egoright_lane  : {egoright_lane}"
            )
        
        # Log skipped image
        reason = f"Paths not in correct order"
        true_img_path = os.path.join(dataset_dir, IMG_DIR, img_path)
        log_skipped_image(
            log_json = log_skipped_json,
            reason = reason,
            image_path = true_img_path
        )
        annotate_skipped_image(
            image = Image.open(true_img_path).convert("RGB"),
            lanes = all_lanes,
            egoleft = egoleft_lane,
            egoright = egoright_lane,
            egopath = drivable_path,
            reason = reason,
            save_path = os.path.join(skipped_path, os.path.basename(true_img_path))
        )

        return None
    
    elif not (egoright_lane[0][0] - egoleft_lane[0][0] >= egoright_lane[-1][0] - egoleft_lane[-1][0]):
        if (verbose):
            warnings.warn(f"Ego lanes are not parallel logically. Ignored.\n \
                - file_path      : {img_path}\n \
                - egoleft_lane   : {egoleft_lane}\n \
                - egoright_lane  : {egoright_lane}"
            )

        # Log skipped image
        reason = f"Lane width illogical, bottom bigger than top"
        true_img_path = os.path.join(dataset_dir, IMG_DIR, img_path)
        log_skipped_image(
            log_json = log_skipped_json,
            reason = reason,
            image_path = true_img_path
        )
        annotate_skipped_image(
            image = Image.open(true_img_path).convert("RGB"),
            lanes = all_lanes,
            egoleft = egoleft_lane,
            egoright = egoright_lane,
            egopath = drivable_path,
            reason = f"{reason} : width_bottom = {egoright_lane[0][0] - egoleft_lane[0][0]}, width_top = {egoright_lane[-1][0] - egoleft_lane[-1][0]}",
            save_path = os.path.join(skipped_path, os.path.basename(true_img_path))
        )

        return None
    
    # FROM HERE, TOP CROPPING IS NOW IN EFFECT
    
    # Top-crop all lines
    egoleft_lane = [
        (x, y - CROP_TOP)
        for x, y in egoleft_lane
    ]
    egoright_lane = [
        (x, y - CROP_TOP)
        for x, y in egoright_lane
    ]
    other_lanes = [
        [
            (x, y - CROP_TOP)
            for x, y in lane
        ]
        for lane in other_lanes
    ]
    
    # Create segmentation masks:
    # Channel 1: egoleft lane
    # Channel 2: egoright lane
    # Channel 3: other lanes
    mask = np.zeros(
        (NEW_H, W, 3), 
        dtype = np.uint8
    )
    mask[:, :, 0] = calcLaneSegMask(
        [egoleft_lane], 
        W, NEW_H,
        normalized = False
    )
    mask[:, :, 1] = calcLaneSegMask(
        [egoright_lane], 
        W, NEW_H,
        normalized = False
    )
    mask[:, :, 2] = calcLaneSegMask(
        other_lanes, 
        W, NEW_H,
        normalized = False
    )

    # Assemble all data
    anno_entry = {
        "img_path"        : img_path,
        "other_lanes"     : other_lanes,
        "egoleft_lane"    : egoleft_lane,
        "egoright_lane"   : egoright_lane,
        "mask"            : mask,
        # "drivable_path"   : drivable_path
    }

    return anno_entry


def calcLaneSegMask(
    lanes, 
    width, height,
    normalized: bool = True
):
    """
    Calculates binary segmentation mask for some lane lines.
    """

    # Create blank mask as new Image
    bin_seg = np.zeros(
        (height, width), 
        dtype = np.uint8
    )
    bin_seg_img = Image.fromarray(bin_seg)

    # Draw lines on mask
    draw = ImageDraw.Draw(bin_seg_img)
    for lane in lanes:
        if (normalized):
            lane = [
                (
                    x * width, 
                    y * height
                ) 
                for x, y in lane
            ]
        draw.line(
            lane, 
            fill = 255, 
            width = 4
        )

    # Convert back to numpy array
    bin_seg = np.array(
        bin_seg_img, 
        dtype = np.uint8
    )

    return bin_seg


def annotateGT(
    anno_entry: dict,
    img_dir: str,
    mask_dir: str,
    visualization_dir: str
):
    """
    Annotates and saves an image with:
        - Annotated image with all lanes, in "output_dir/visualization".
    """

    # Define save name, now saving everything in JPG
    # to preserve my remaining disk space
    save_name = str(img_id_counter).zfill(6)

    # Prepping canvas
    raw_img = Image.open(
        os.path.join(
            img_dir, 
            anno_entry["img_path"]
        )
    ).convert("RGB")

    # Crop top
    raw_img = raw_img.crop((
        0, 
        CROP_TOP, 
        W, 
        H
    ))
    
    # draw = ImageDraw.Draw(raw_img)
    
    # lane_colors = {
    #     "outer_red": (255, 0, 0), 
    #     "ego_green": (0, 255, 0), 
    #     "drive_path_yellow": (255, 255, 0)
    # }
    # lane_w = 5

    # # Draw other lanes, in red
    # for line in anno_entry["other_lanes"]:
    #     draw.line(
    #         line, 
    #         fill = lane_colors["outer_red"], 
    #         width = lane_w
    #     )
    
    # # Draw drivable path, in yellow
    # draw.line(
    #     anno_entry["drivable_path"],
    #     fill = lane_colors["drive_path_yellow"], 
    #     width = lane_w
    # )

    # # Draw ego lanes, in green
    # if (anno_entry["egoleft_lane"]):
    #     draw.line(
    #         anno_entry["egoleft_lane"],
    #         fill = lane_colors["ego_green"],
    #         width = lane_w
    #     )
    # if (anno_entry["egoright_lane"]):
    #     draw.line(
    #         anno_entry["egoright_lane"],
    #         fill = lane_colors["ego_green"],
    #         width = lane_w
    #     )

    # Fetch seg mask and save as RGB PNG
    mask_img = Image.fromarray(anno_entry["mask"]).convert("RGB")
    mask_img.save(os.path.join(mask_dir, save_name + ".png"))

    # Overlay mask on raw image, ratio 1:1
    overlayed_img = Image.blend(
        raw_img, 
        mask_img, 
        alpha = 0.5
    )

    # Save visualization img, JPG for lighter weight, just different dir
    overlayed_img.save(os.path.join(visualization_dir, save_name + ".jpg"))


if __name__ == "__main__":

    # ============================== Dataset structure ============================== #

    # FYI: https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md

    IMAGE_SPLITS = [
        "training", 
        "validation"
    ]
    IMG_DIR = "images"
    LABEL_SPLITS = {
        "lane3d_1000_training" : [
            "training",
        ],
        "lane3d_1000_validation_test" : [
            "validation",
            # "test" not included
        ]
    }

    # All 200k scenes have reso 1920 x 1280. I checked it manually.
    # Now I will cut the top 320 pixels to make it 1920 x 960, exactly 2:1 ratio.
    W = 1920
    H = 1280
    CROP_TOP = 320
    NEW_H = H - CROP_TOP

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process OpenLane dataset - groundtruth generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        "-i",
        type = str, 
        help = "OpenLane raw directory",
        required = True
    )
    parser.add_argument(
        "--output_dir", 
        "-o",
        help = "Output directory",
        required = True
    )
    # For debugging only
    parser.add_argument(
        "--early_stopping",
        "-e",
        type = int,
        help = "Num. files you wanna limit, instead of whole set.",
        required = False
    )

    args = parser.parse_args()

    # Parse dirs
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, stopping after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # Generate output structure
    """
    Due to the huge dataset size, and since we don't have to edit the raw images,
    I have decided to not outputing the raw image files, but instead only the
    visualizations and groundtruth JSON.

    --output_dir
        |----visualization
        |----drivable_path.json

    """

    list_subdirs = ["visualization", "mask"]

    if (os.path.exists(output_dir)):
        warnings.warn(f"Output directory {output_dir} already exists. Purged")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for subdir in list_subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if (not os.path.exists(subdir_path)):
            os.makedirs(subdir_path, exist_ok = True)

    # Logging skipped images for auditing
    log_skipped_json = {}
    skipped_path = os.path.join(output_dir, "skipped")
    if (not os.path.exists(skipped_path)):
        os.makedirs(skipped_path, exist_ok = True)

    # ============================== Parsing annotations ============================== #

    data_master = {}
    img_id_counter = 0
    flag_continue = True

    for label_split, list_label_subdirs in LABEL_SPLITS.items():
        if (not flag_continue):
            break
        print(f"\nPROCESSING LABEL SPLIT : {label_split}")
        
        for subsplit in list_label_subdirs:
            if (not flag_continue):
                break
            print(f"PROCESSING SUBSPLIT : {subsplit}")

            subsplit_path = os.path.join(
                dataset_dir,
                label_split,
                subsplit
            )

            for segment in tqdm(
                sorted(os.listdir(subsplit_path)), 
                desc = "Processing segments : ",
                colour = "green"
            ):
                if (not flag_continue):
                    break
                segment_path = os.path.join(subsplit_path, segment)

                for label_file in sorted(os.listdir(segment_path)):
                    if (not flag_continue):
                        break
                                    
                    label_file_path = os.path.join(segment_path, label_file)

                    with open(label_file_path, "r") as f:
                        this_label_data = json.load(f)

                    this_label_data = parseData(
                        json_data = this_label_data,
                        verbose = False
                    )
                    if (this_label_data):

                        annotateGT(
                            anno_entry = this_label_data,
                            img_dir = os.path.join(
                                dataset_dir,
                                IMG_DIR
                            ),
                            mask_dir = os.path.join(
                                output_dir, 
                                "mask"
                            ),
                            visualization_dir = os.path.join(
                                output_dir, 
                                "visualization"
                            )
                        )

                        img_index = str(str(img_id_counter).zfill(6))
                        data_master[img_index] = {
                            "img_path"      : this_label_data["img_path"],
                            "egoleft_lane"  : round_line_floats(
                                normalizeCoords(
                                    this_label_data["egoleft_lane"],
                                    W, NEW_H
                                )
                            ),
                            "egoright_lane" : round_line_floats(
                                normalizeCoords(
                                    this_label_data["egoright_lane"],
                                    W, NEW_H
                                )
                            ),
                            "other_lanes"   : [
                                round_line_floats(
                                    normalizeCoords(
                                        lane,
                                        W, NEW_H
                                    )
                                )
                                for lane in this_label_data["other_lanes"]
                            ],
                            # "drivable_path" : round_line_floats(
                            #     normalizeCoords(
                            #         this_label_data["drivable_path"],
                            #         W, NEW_H
                            #     )
                            # )
                        }

                        img_id_counter += 1

                    # Early stopping check
                    if (
                        early_stopping and 
                        (img_id_counter >= early_stopping)
                    ):
                        flag_continue = False
                        print(f"Early stopping reached at {early_stopping} samples.")
                        break

                # print(f"Segment {segment} done, with {len(os.listdir(segment_path))} samples collected.")

    # Save master data
    with open(
        os.path.join(output_dir, "drivable_path.json"), 
        "w"
    ) as f:
        json.dump(
            data_master, f, 
            indent = 4
        )

    # Save log of skipped images
    with open(
        os.path.join(output_dir, "log_skipped.json"), 
        "w"
    ) as f:
        json.dump(
            log_skipped_json, f, 
            indent = 4
        )