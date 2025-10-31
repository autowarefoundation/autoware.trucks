import argparse
import os
import json
import cv2
import warnings
from PIL import Image, ImageDraw
import copy
import sys
import math


# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"


warnings.formatwarning = custom_warning_format


def annotateGT(classified_lanes, gt_images_path, output_dir="visualization", crop=None):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)

    # Define lane colors
    lane_colors = {
        "egoleft_lane": (0, 255, 0),  # Green
        "egoright_lane": (0, 0, 255),  # Blue
        "other_lanes": (255, 255, 0),  # Yellow
    }
    lane_width = 5

    # To stop loop with required images.
    img_id_counter = 0

    for image_id, lanes in classified_lanes.items():
        image_path = os.path.join(gt_images_path, image_id + ".png")
        img_id_counter += 1

        if img_id_counter == STOP_AT:
            break

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_id} not found in {gt_images_path}. Skipping.")
            continue

        # Load image
        img = Image.open(image_path).convert("RGB")

        # Apply cropping if crop is provided
        if crop:
            CROP_TOP = crop["TOP"]
            CROP_RIGHT = crop["RIGHT"]
            CROP_BOTTOM = crop["BOTTOM"]
            CROP_LEFT = crop["LEFT"]

            img = img.crop(
                (
                    CROP_LEFT,
                    CROP_TOP,
                    ORIGINAL_IMG_WIDTH - CROP_RIGHT,
                    ORIGINAL_IMG_HEIGHT - CROP_BOTTOM,
                )
            )

        draw = ImageDraw.Draw(img)
        # Draw each lane
        for lane_type, lane_data in lanes.items():
            if lane_type in ["img_height", "img_width"]:
                continue  # Skip metadata keys

            color = lane_colors.get(
                lane_type, (255, 255, 255)
            )  # Default white if unknown

            if lane_type in ["egoleft_lane", "egoright_lane"]:
                # Handle ego lanes which contain direct keypoints
                lane_points = []  # Convert keypoints
                for x, y in lane_data:
                    new_x = x * IMG_WIDTH
                    new_y = y * IMG_HEIGHT
                    lane_points.append((new_x, new_y))
                if len(lane_points) > 1:
                    # Ensure we have enough points to draw
                    draw.line(lane_points, fill=color, width=lane_width)
            else:
                # Handle other lanes which are in a list
                for lane_group in lane_data:
                    lane_points = []  # Convert keypoints
                    for x, y in lane_group:
                        new_x = x * IMG_WIDTH
                        new_y = y * IMG_HEIGHT
                        lane_points.append((new_x, new_y))
                    if len(lane_points) > 1:
                        # Ensure we have enough points to draw
                        draw.line(lane_points, fill=color, width=lane_width)

        # Ensure we save as PNG by changing the file extension
        image_name_base = image_id
        # image_name_base = os.path.splitext(image_name)[0]  # Remove the extension
        image_name_png = f"{image_name_base}.png"  # Add .png extension

        save_path = os.path.join(output_dir, "visualization", image_name_png)
        img.save(save_path, format="PNG")
        print(f"Saved annotated image: {save_path} with dimensions {img.size}")

    print("Annotation complete. All images saved in", output_dir)


def classify_lanes(data, gt_images_path, output_dir):
    result = {}
    # Define threshold for slope comparison
    SLOPE_THRESHOLD = 0.6
    DISTANCE_THRESHOLD = 0.25 * IMG_WIDTH  # For parallel lanes that shouldn't be merged

    img_id_counter = 0
    for entry in data:
        image_id = entry["name"]
        img_id_counter += 1
        if img_id_counter == STOP_AT:
            break

        # Initialize result structure
        result[image_id] = {
            "egoleft_lane": [],
            "egoright_lane": [],
            "other_lanes": [],
            "img_height": IMG_HEIGHT,
            "img_width": IMG_WIDTH,
        }

        if "labels" not in entry or not entry["labels"]:
            KeyError(f"labels not present for Image {image_id}")
            continue

        # First collect all valid lanes(lanes with parallel lane direction)
        valid_lanes = []
        for lane in entry["labels"]:
            if "poly2d" not in lane:
                KeyError(f"poly2d not present for Image {image_id}")
                continue

            # Skip lanes with vertical direction
            if (
                "attributes" in lane
                and "laneDirection" in lane["attributes"]
                and lane["attributes"]["laneDirection"] == "vertical"
            ) or len(lane["poly2d"][0]["vertices"]) <= 1:
                continue

            vertices = lane["poly2d"][0]["vertices"]
            anchor = getLaneAnchor(vertices)

            if anchor[0] is not None and anchor[1] is not None:
                valid_lanes.append(
                    {
                        "id": lane["id"],
                        "vertices": vertices,
                        "anchor_x": anchor[0],
                        "slope": anchor[1],
                        "intercept": anchor[2],
                    }
                )

        # First merge similar lanes
        merged_lanes = []

        # Keep an array to track which lanes have been merged with current lane.
        processed = [False] * len(valid_lanes)

        # Sort lanes by anchor x position
        valid_lanes.sort(key=lambda lane: lane["anchor_x"])

        i = 0
        # Check if current lane i can be merged with any of the next lanes in valid_lanes list.

        while i < len(valid_lanes):
            if processed[i]:
                i += 1
                continue

            current_lane = valid_lanes[i]

            # This group contains all the lanes that could be merged.
            merged_group = [current_lane]
            processed[i] = True

            # Check subsequent lanes for merging
            j = i + 1
            while j < len(valid_lanes):
                if not processed[j]:
                    next_lane = valid_lanes[j]

                    # Check if slopes are valid (not None) and not perpendicular
                    slopes_valid = (
                        current_lane["slope"] is not None
                        and next_lane["slope"] is not None
                        and (current_lane["slope"] * next_lane["slope"] != -1)
                    )

                    # Check if anchors are close enough
                    anchor_distance_valid = (
                        abs(current_lane["anchor_x"] - next_lane["anchor_x"])
                        <= DISTANCE_THRESHOLD
                    )

                    # Check if slopes are similar
                    if slopes_valid:
                        angle_radians = math.atan(
                            abs(
                                (current_lane["slope"] - next_lane["slope"])
                                / (1 + current_lane["slope"] * next_lane["slope"])
                            )
                        )
                        slopes_similar = angle_radians <= SLOPE_THRESHOLD

                    if slopes_valid and slopes_similar and anchor_distance_valid:
                        merged_group.append(next_lane)
                        processed[j] = True

                j += 1

            # Merge all lanes in the group
            if len(merged_group) > 1:
                merged_vertices = merged_group[0]["vertices"]
                for k in range(1, len(merged_group)):
                    merged_vertices = merge_lane_lines(
                        merged_vertices, merged_group[k]["vertices"]
                    )
                if len(merged_vertices) > 1:
                    # Recalculate anchor for the merged lane
                    merged_anchor = getLaneAnchor(merged_vertices)

                    merged_lanes.append(
                        {"vertices": merged_vertices, "anchor_x": merged_anchor[0]}
                    )
            else:
                # Add unmerged lane as is
                merged_lanes.append(
                    {
                        "vertices": current_lane["vertices"],
                        "anchor_x": current_lane["anchor_x"],
                    }
                )

            i += 1

        # Sort merged lanes by anchor x position
        merged_lanes.sort(key=lambda lane: lane["anchor_x"])

        # Now classify the merged lanes using anchor points logic
        if len(merged_lanes) > 0:
            # Prepare anchor points in the format expected by getEgoIndexes
            # getEgoIndexes expects a list of tuples (x_position, lane_id, slope)
            # We'll use the index as lane_id since we don't need it for merging anymore
            anchor_points = []
            for idx, lane in enumerate(merged_lanes):
                # Create a dummy tuple with anchor_x, index as id, and None for slope (not used in getEgoIndexes)
                anchor_points.append((lane["anchor_x"], idx, None))

            # Get left and right ego lanes using getEgoIndexes
            ego_indexes = getEgoIndexes(anchor_points)

            # Skip if ego_indexes indicates an error
            if isinstance(ego_indexes, str):
                continue

            left_idx, right_idx = ego_indexes

            # Add lanes to appropriate categories
            for i, lane in enumerate(merged_lanes):
                if left_idx is not None and i == left_idx:
                    result[image_id]["egoleft_lane"] = lane[
                        "vertices"
                    ]  # Store directly as vertices
                elif right_idx is not None and i == right_idx:
                    result[image_id]["egoright_lane"] = lane[
                        "vertices"
                    ]  # Store directly as vertices
                else:
                    result[image_id]["other_lanes"].append(lane["vertices"])

    return result


def process_lane_keypoints(lane, CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM):
    formatted_lane = []

    # Interpolate for very few points
    if len(lane) < 10:
        lane = interpolated_list(lane, 5)  # 5 points between each pair of points

    for point in lane:
        # Normalize x by width and y by height
        if (CROP_LEFT <= point[0] <= (ORIGINAL_IMG_WIDTH - CROP_RIGHT)) and (
            CROP_TOP <= point[1] <= (ORIGINAL_IMG_HEIGHT - CROP_BOTTOM)
        ):
            new_x = point[0] - CROP_LEFT
            new_y = point[1] - CROP_TOP
            # Normalize and crop
            formatted_lane.append([new_x / IMG_WIDTH, new_y / IMG_HEIGHT])

    return formatted_lane


def format_data(data, crop):
    formatted_data = {}

    if crop:
        CROP_TOP = crop["TOP"]
        CROP_RIGHT = crop["RIGHT"]
        CROP_BOTTOM = crop["BOTTOM"]
        CROP_LEFT = crop["LEFT"]

    for idx, (image_key, image_data) in enumerate(data.items()):
        # Create a copy of the image data
        formatted_image = image_data.copy()

        # Normalize lane keypoints
        for lane_type in ["egoleft_lane", "egoright_lane", "other_lanes"]:
            if lane_type in image_data:
                if lane_type in ["egoleft_lane", "egoright_lane"]:
                    # Handle ego lanes which contain direct keypoints
                    lane = image_data[lane_type]
                    formatted_image[lane_type] = process_lane_keypoints(
                        lane, CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM
                    )
                else:
                    # Handle other lanes which are in a list
                    formatted_lanes = []
                    for lane in image_data[lane_type]:
                        formatted_lane = process_lane_keypoints(
                            lane, CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM
                        )
                        formatted_lanes.append(formatted_lane)
                    formatted_image[lane_type] = formatted_lanes

        formatted_data[image_key] = formatted_image
    return formatted_data


def getLaneAnchor(lane):
    # Sort lane keypoints in decreasing order of y coordinates.

    lane_copy = lane.copy()
    lane_copy.sort(key=lambda point: point[1], reverse=True)

    (x2, y2) = lane_copy[0]
    (x1, y1) = lane_copy[1]

    num_vertical = 0
    num_horizontal = 0

    for i in range(1, len(lane_copy) - 1, 1):
        if lane_copy[i][0] != x2:
            (x1, y1) = lane_copy[i]
            break
    if x1 == x2:
        num_vertical += 1
        return (x1, None, None)
    # Get slope and intercept
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    # Handle horizontal lanes
    if a == 0:
        num_horizontal += 1
        return (x1, None, None)
    # Get projected point on the image's bottom
    x0 = (ORIGINAL_IMG_HEIGHT - b) / a

    return (x0, a, b)


def getEgoIndexes(anchors):
    tolerance = 0.03  # 3% tolerance
    for i in range(len(anchors)):
        if anchors[i][0] >= (ORIGINAL_IMG_WIDTH - (tolerance * ORIGINAL_IMG_WIDTH)) / 2:
            if i == 0:
                # First lane is already past the center
                if len(anchors) >= 2:
                    # Use first and second lanes as ego lanes
                    return (i, i + 1)
                return (None, 0)
            # Normal case: use the lane before and at the center
            return (i - 1, i)

    # If we get here, no lanes are past the center
    if len(anchors) >= 2:
        return (len(anchors) - 1, None)
    elif len(anchors) == 1:
        return (0, None)
    return "Unhandled Edge Case"


def process_binary_mask(args, json_data_path):
    CROP_TOP, CROP_RIGHT, CROP_BOTTOM, CROP_LEFT = args.crop

    # Check output directory exists or not
    os.makedirs(os.path.join(args.output_dir, "segmentation"), exist_ok=True)

    # Read JSON data
    with open(json_data_path, "r") as f:
        data = json.load(f)

    binary_lane_mask_path = os.path.join(args.labels_dir, "lane", "masks", "train")

    img_id_counter = 0
    # Process each image entry from the JSON file
    for entry in data:
        image_id = str(str(img_id_counter).zfill(6))
        img_id_counter += 1
        if img_id_counter == STOP_AT:
            break

        if "name" not in entry:
            continue

        # Construct input path using the image name from JSON
        input_path = os.path.join(binary_lane_mask_path, entry["name"].replace(".jpg", ".png"))
        output_path = os.path.join(args.output_dir, "segmentation", f"{image_id}.png")

        # Read the binary mask image in grayscale mode
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # **Invert the binary mask** (assumes lane is black and background is white)
            img = 255 - img  # Invert colors

            # Crop the image
            height, width = img.shape
            cropped_img = img[
                CROP_TOP : height - CROP_BOTTOM, CROP_LEFT : width - CROP_RIGHT
            ]

            # Save the processed image
            cv2.imwrite(output_path, cropped_img)
        else:
            pass

    print(
        f"Processed {img_id_counter} binary mask images to {os.path.join(args.output_dir, 'segmentation')}"
    )


def saveGT(json_data_path, gt_images_path, args):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "image"), exist_ok=True)

    with open(json_data_path, "r") as f:
        data = json.load(f)

    print(f"Found {len(data)} entries in lane_train.json")

    img_id_counter = 0
    skipped_img_counter = 0

    # Process each image entry from the JSON file
    for entry in data:
        image_id = str(str(img_id_counter).zfill(6))  # Format as 000000, 000001, etc.
        img_id_counter += 1
        # Remove, Early stopping
        if img_id_counter == STOP_AT:
            break
        if "name" not in entry:
            skipped_img_counter += 1
            continue

        input_path = os.path.join(gt_images_path, entry["name"])  # Original image path
        entry["name"] = image_id

        # Skip if file doesn't exist
        if not os.path.exists(input_path):
            print(
                f"Warning: Ground truth image {image_id} not found in {gt_images_path}. Skipping."
            )
            continue

        # Save gt images with serialized names
        output_path = os.path.join(args.output_dir, "image", f"{image_id}.png")

        # Read the image
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)  # OpenCV loads as BGR

        if img is not None:
            # Save the image
            cv2.imwrite(output_path, img)
        else:
            skipped_img_counter += 1
            continue
    print(
        f"Copied {img_id_counter} images to {os.path.join(args.output_dir, 'images')} and skipped {skipped_img_counter} images."
    )

    return data


# Create interpolation functions for x-coordinates
def interpolate_x(y, vertices):
    for i in range(len(vertices) - 1):
        y1, y2 = vertices[i][1], vertices[i + 1][1]
        if y1 <= y <= y2:
            # use current best ground truth values for interpolation.
            x1, x2 = vertices[i][0], vertices[i + 1][0]
            # Linear interpolation
            if y2 - y1 == 0:
                return x1
            return x1 + (x2 - x1) * (y - y1) / (y2 - y1)
    return None


def interpolated_list(point_list, req_points):
    if not point_list or len(point_list) < 2:
        return point_list.copy()

    result = []

    # Process each pair of consecutive points in original ordering
    for i in range(len(point_list) - 1):
        p1 = point_list[i]
        p2 = point_list[i + 1]

        # Add the first point of this segment
        result.append(p1.copy())

        # Get the linear equation between p1 and p2
        x1, y1 = p1
        x2, y2 = p2

        # Add interpolated points between p1 and p2
        for j in range(1, req_points + 1):
            t = j / (
                req_points + 1
            )  # Choose t to be 1/req_points, 2/req_points, ..., req_points/req_points + 1

            # Calculate y using linear interpolation
            y = y1 + t * (y2 - y1)

            # Calculate x using the linear equation between p1 and p2
            if y2 - y1 == 0:  # Horizontal line
                x = x1 + t * (x2 - x1)
            else:
                # Linear interpolation for x based on y
                x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)

            result.append([x, y])

    # Add the last point
    result.append(point_list[-1].copy())

    return result


def merge_lane_lines(vertices1, vertices2):
    # Check for empty inputs
    if not vertices1 or not vertices2:
        return vertices1 if vertices1 else vertices2

    vertices1_copy = vertices1.copy()
    vertices2_copy = vertices2.copy()

    # Sort vertices by y-coordinate
    v1_sorted = sorted(vertices1_copy, key=lambda p: p[1])
    v2_sorted = sorted(vertices2_copy, key=lambda p: p[1])

    # Verify that vertices have data before accessing
    if not v1_sorted or not v2_sorted:
        return vertices1 if vertices1 else vertices2

    # Get y-coordinate range
    y_min = min(v1_sorted[0][1], v2_sorted[0][1])
    y_max = max(v1_sorted[-1][1], v2_sorted[-1][1])

    # If the lanes don't overlap vertically, return the one with more points
    if y_min >= y_max:
        return vertices1 if len(vertices1) >= len(vertices2) else vertices2

    # Generate merged points
    merged = []
    num_points = 50  # Number of points in merged lane

    for i in range(num_points):
        y = y_min + (y_max - y_min) * i / (num_points - 1)
        x1 = interpolate_x(y, v1_sorted)
        x2 = interpolate_x(y, v2_sorted)

        if x1 is not None and x2 is not None:
            x = (x1 + x2) / 2  # Average x coordinates
            merged.append([x, y])

    # If merged result is empty, return original with more points
    return (
        merged
        if merged
        else (
            vertices1
            if math.sqrt(
                (vertices1[-1][0] - vertices1[0][0]) ** 2
                + (vertices1[-1][1] - vertices1[0][1]) ** 2
            )
            >= math.sqrt(
                (vertices2[-1][0] - vertices2[0][0]) ** 2
                + (vertices2[-1][1] - vertices2[0][1]) ** 2
            )
            else vertices2
        )
        # If merging failed, return the lane with the longer length
        # Calculate the Euclidean distance between the first and last points of each lane
        # This helps determine which lane is longer and likely more representative
        # Choose vertices1 if it's longer than vertices2, otherwise choose vertices2
    )


if __name__ == "__main__":
    # ============================== Constants ============================== #

    # Define Dimension
    ORIGINAL_IMG_WIDTH = 1280
    ORIGINAL_IMG_HEIGHT = 720

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description="Process BDD100k dataset - EgoLanes groundtruth generation"
    )

    # bdd100k/images/100k/
    parser.add_argument(
        "--image_dir",
        type=str,
        help="BDD100k ground truth image parent directory (right after extraction)",
    )

    # bdd100k/labels/
    parser.add_argument(
        "--labels_dir",
        type=str,
        help="BDD100k labels directory (right after extraction)",
        required=True,
    )

    # ./output
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Desired output directory",
        default="./processed_BDD100K",
        required=False,
    )

    parser.add_argument(
        "--num_images",
        type=int,
        help="Number of images to process",
        default=sys.maxsize,
        required=False,
    )

    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        help="Crop image: [TOP, RIGHT, BOTTOM, LEFT]. Must always be 4 ints. Non-cropped sizes are 0.",
        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"),
        default=[0, 140, 220, 140],
        required=False,
    )

    args = parser.parse_args()

    # Store the number of images to process
    STOP_AT = args.num_images

    json_data_path = os.path.join(
        args.labels_dir, "lane", "polygons", "lane_train.json"
    )

    # ============================== Save binary mask ============================== #
    process_binary_mask(args, json_data_path)

    # ============================== Save ground truth images ============================== #

    # Save ground truth images.
    gt_images_path = os.path.join(args.image_dir, "train")
    data = saveGT(
        json_data_path, gt_images_path, args
    )  # Do not enable if not required.

    # ============================== Get crop image params ============================== #

    # Updated dimensions
    IMG_WIDTH = ORIGINAL_IMG_WIDTH - args.crop[1] - args.crop[3]
    IMG_HEIGHT = ORIGINAL_IMG_HEIGHT - args.crop[0] - args.crop[2]

    crop = {
        "TOP": args.crop[0],
        "RIGHT": args.crop[1],
        "BOTTOM": args.crop[2],
        "LEFT": args.crop[3],
    }

    # ============================== Merge and classify lane lines ============================== #
    gt_images_path = os.path.join(args.output_dir, "image")
    classified_lanes = classify_lanes(data, gt_images_path, output_dir=args.output_dir)

    # # # ============================== Normalise, crop Image data  ============================== #
    formatted_data = format_data(classified_lanes, crop)

    # # # ============================== AnnotateGT ================================ # # #

    annotateGT(formatted_data, gt_images_path, output_dir=args.output_dir, crop=crop)

    # # # ============================== Save result JSON ============================== #

    # Save classified lanes as new JSON file
    output_file = os.path.join(args.output_dir, "drivable_path.json")

    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=4)
