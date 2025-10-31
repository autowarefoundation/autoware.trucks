# EgoLanes Dataset - BDD100K

BDD100K provides comprehensive lane marking annotations across 100,000 diverse driving images (1280×720). The lane annotations include:

- Detailed polygonal representations of lane boundaries
- Lane direction attributes (parallel, vertical)
- Lane type classifications (solid, dashed, double)
- Binary lane segmentation masks
- Annotations across diverse conditions (day/night, clear/rainy, city/highway)

Format of lane_train.json
```json
[
  {
    "name": "0000f77c-6257be58.jpg",
    "labels": [
      {
        "id": "0",
        "attributes": {
          "laneDirection": "parallel",
          "laneStyle": "solid",
          "laneTypes": "road curb"
        },
        "category": "road curb",
        "poly2d": [
          {
            "vertices": [[503.67, 373.13], [357.79, 374.67]],
            "types": "LL",
            "closed": false
          }
        ]
      },
      ...  // Additional lane markings
    ]
  },
  {
    "name": "0000f77c-62c2a288.jpg",
    "labels": [
      ...  // Multiple lane markings with different attributes
      {
        "id": "6",
        "attributes": {
          "laneDirection": "parallel",
          "laneStyle": "dashed",
          "laneTypes": "single white"
        },
        "category": "single white",
        "poly2d": [
          {
            "vertices": [
              [795.68, 345.23], 
              [628.16, 370.58], 
              [504.73, 482.99], 
              [370.28, 571.15]
            ],
            "types": "LCCC",  // Line followed by curves
            "closed": false
          }
        ]
      },
      ...  // More lane markings
    ]
  },
  ...  // Thousands more images with similar annotation structure
]
```
</div>

Download image dataset - [Ground Truth Images](https://dl.cv.ethz.ch/bdd100k/data/100k_images_train.zip)<br>
Download lane_train.json, binary lane masks - [Lane_labels](https://dl.cv.ethz.ch/bdd100k/data/bdd100k_lane_labels_trainval.zip)<br>
Download the dataset - [All links](https://dl.cv.ethz.ch/bdd100k/data/)<br>
More about the dataset - [Academic Paper](https://arxiv.org/abs/1805.04687)<br>

# `process_bdd100k.py`

## Overview
This script processes BDD100K's `lane_train.json` data to generate standardized lane annotations for autonomous driving. It automatically classifies lanes into egoleft_lane, egoright_lane, and other_lanes categories based on their position relative to the vehicle. Adjacent lanes with similar slopes and proximity are intelligently merged to create continuous lane markings. The tool also handles sparse points through interpolation, normalizes coordinates, and generates annotated images for visual verification. The output `bdd100k_egolanes_train.json` provides a consistent, ready-to-use format for training lane detection models.

## Requirements
- Python 3.6+
- OpenCV
- PIL (Pillow)
- NumPy

## Usage

```bash
python process_bdd100k.py --image_dir /path/to/bdd100k/images/100k/ \
                          --labels_dir /path/to/lane_labels
```

## Arguments
- `--image_dir`: BDD100K ground truth image parent directory (right after extraction)
- `--labels_dir`: BDD100K lane labels directory (contains lane polygons and masks)
- `--output_dir`: Desired output directory (optional, default directory is ./processed_BDD100K/)
- `--crop`: (optional)[TOP, RIGHT, BOTTOM, LEFT] cropping values (default: [0, 140, 220, 140])
- `--num_images`: Number of images to process from BDD100K dataset. (default is sys.maxsize(2^63 - 1))

## Directory Structure

```bash
├── README.md
├── process_bdd100k.py
├── processed_BDD100K
│   ├── drivable_path.json # Classified, merged lane keypoints data
│   ├── image # Groundtruth images renamed
│   │   ├── 000000.png
│   │   └── 000001.png
│   ├── segmentation # Binary mask for lane lines
│   │   ├── 000000.png
│   │   └── 000001.png
│   ├── visualization # Annotated images output
│   │   ├── 000000.png
│   │   └── 000001.png
```

## Output JSON format
```JSON
{
    <6-digit frame id> : {
        "egoleft_lane": [
            [<x_1>, <y_1>],
            [<x_2>, <y_2>],
            .....................
            [<x_n>, <y_n>],
        ],
        "egoright_lane": [
            [<x_1>, <y_1>],
            [<x_2>, <y_2>],
            .....................
            [<x_n>, <y_n>],
        ],
        "other_lanes": [
          [
            [<x_1>, <y_1>],
            [<x_2>, <y_2>],
            .....................
            [<x_n>, <y_n>],
          ], 
          [
            [<x_1>, <y_1>],
            [<x_2>, <y_2>],
            .....................
            [<x_n>, <y_n>],
          ]
            
        ],
        "img_height": <h>,
        "img_width": <w>
    },
    .................................
}
```

## Core functions

### `classify_lanes()`
Classify lanes from BDD100K dataset, filtering out vertical lanes and processing
the remaining lanes to identify their positions relative to the ego vehicle.
First merges similar lanes, then classifies them as ego left, ego right, or other.

- Args:
    data: List of dictionaries containing lane annotations from BDD100K
    gt_images_path: Path to the ground truth images
    output_dir: Directory to save the processed results

- Returns:
    Dictionary mapping image IDs to classified lane data

### `getLaneAnchor()`
Determine the anchor point of a lane by finding where it intersects with the bottom of the image.

This function calculates where a lane would intersect with the bottom of the image by:
1. Sorting lane points by y-coordinate (from bottom to top)
2. Finding the slope and y-intercept using the lowest two valid points
3. Extrapolating to find the x-coordinate where y = image height

- Args:
    lane (list): List of [x, y] coordinate pairs representing the lane points

- Returns:
    tuple: (x0, a, b) where:
        - x0: x-coordinate where lane intersects bottom of image
        - a: slope of the lane
        - b: y-intercept of the lane
    OR (x1, None, None) if the lane is vertical or horizontal

- Note:
    - For vertical lanes (x1 = x2), returns (x1, None, None)
    - For horizontal lanes (a = 0), returns (x1, None, None)

### `getEgoIndexes()`
Identifies the two ego lanes (left and right) from sorted anchor points.

  This function determines which lanes are the ego-vehicle lanes by finding
  the lanes closest to the center of the image (ORIGINAL_IMG_WIDTH/2).

  - Args:
      anchors: List of tuples (x_position, lane_id, slope), sorted by x_position.
              Each tuple represents a lane with its anchor point x-coordinate,
              unique ID, and slope.

  - Returns:
      tuple: (left_idx, right_idx) - Indices of the left and right ego lanes
              in the anchors list.
      str: Error message if proper ego lanes cannot be determined.

  - Logic:
      1. Iterate through sorted anchors to find the first lane with x-position
          greater than or equal to the image center.
      2. This lane and the one before it are considered the ego lanes.
      3. Special cases:
          - If the first lane is already past center (i=0), use it and the next lane
            if available, otherwise return (None, 0)
          - If no lanes are past center, use the last lane as the left ego lane
            and return right as None
          - If there's only one lane, it's considered the left ego lane
          - If no lanes are available, return "Unhandled Edge Case"

### `interpolated_list()`
Takes original point list and inserts 3 interpolated points between
  each consecutive pair of points in the original list.
  Uses the linear equation between each specific pair of points.
  - Args:
      point_list: List of [x, y] coordinates
      req_points: Number of interpolated points to insert between each pair of points
  - Returns:
      List of original points with interpolated points inserted

### `format_data()`
Normalize all keypoints in the data by dividing x-coordinates by img_width
and y-coordinates by img_height to scale them between 0 and 1.

- Args:
    data (dict): Dictionary containing image data with keypoints and dimensions
    crop (dict): Dictionary specifying cropping boundaries (TOP, RIGHT, BOTTOM, LEFT)

- Returns:
    dict: Dictionary with the same structure but normalized keypoints and sequential image keys

### `merge_lane_lines()`
Merge two lane lines into a single lane by interpolating and averaging points.
- Args:
    vertices1: List of [x, y] coordinates for first lane
    vertices2: List of [x, y] coordinates for second lane
- Returns:
    List of merged [x, y] coordinates


### `interpolate_x()`
Interpolate x-coordinate for a given y-coordinate using linear interpolation between vertices.

This function takes a y-coordinate and a list of vertices, and returns the corresponding
x-coordinate by linearly interpolating between the two vertices that bound the given y-coordinate.
If the y-coordinate is outside the range of vertices or if the vertices form a horizontal line,
it returns None.

- Args:
    y (float): The y-coordinate for which to find the corresponding x-coordinate
    vertices (list): List of [x, y] coordinate pairs representing the vertices of a line

- Returns:
    float or None: The interpolated x-coordinate if successful, None if interpolation is not possible

### `saveGT()`
Copies images from gt_images_path to the output directory with serialized filenames.
Uses image names from lane_train.json instead of reading directory contents.

- Args:
    gt_images_path (str): Path to the directory containing ground truth images.
    args: Command-line arguments containing output_dir and labels_dir.


### `process_binary_mask()`
Processes binary lane mask images from BDD100K dataset.

This function reads binary lane mask images, inverts them (assuming lanes are black
and background is white in the original), crops them according to specified dimensions,
and saves them with serialized filenames in the output directory.

Args:
    args: Command-line arguments containing:
        - labels_dir: Directory containing the BDD100K lane mask images
        - output_dir: Directory where processed images will be saved
        - crop: Tuple of (top, right, bottom, left) crop values in pixels

Output:
    Processed binary masks saved to {args.output_dir}/segmentation/ with
    serialized filenames (000000.png, 000001.png, etc.)

### `annotateGT()`

Annotates ground truth images with lane lines and saves them to the specified output directory.

This function takes classified lane data and creates visual annotations on the corresponding
ground truth images. Each lane type (ego left, ego right, and other lanes) is drawn with
a distinct color for easy visualization. The images can be optionally cropped before
annotation.

- Args:
    classified_lanes (dict): Dictionary mapping image IDs to lane data, where each lane
        is classified as 'egoleft_lane', 'egoright_lane', or 'other_lanes'
    gt_images_path (str): Path to the directory containing the ground truth images
    output_dir (str, optional): Directory to save annotated images. Defaults to "visualization"
    crop (dict, optional): Dictionary specifying cropping boundaries with keys:
        - TOP: Pixels to crop from top
        - RIGHT: Pixels to crop from right
        - BOTTOM: Pixels to crop from bottom
        - LEFT: Pixels to crop from left

- Returns:
    None

- Note:
    - Ego left lanes are drawn in green
    - Ego right lanes are drawn in blue
    - Other lanes are drawn in yellow
    - Each lane is drawn with a width of 5 pixels

### `process_lane_keypoints()`
Crop and normalize lane keypoints.
Returns a list of normalized keypoints.



