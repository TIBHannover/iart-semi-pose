import os
import sys
import re
import argparse
import json
import imageio

sys.path.append(".")
from utils.plot_utils import plot_prediction_images


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--image_path", help="")
    parser.add_argument("-p", "--prediction_path", help="")
    parser.add_argument("-o", "--output_path", help="")

    parser.add_argument("--skip_no_boxes", action="store_true", help="")
    parser.add_argument("--skip_score_below", type=float, help="")
    parser.add_argument("--id_filter", help="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    image_to_draw = []
    with open(args.prediction_path, "r") as f:
        for line in f:
            d = json.loads(line)
            if args.id_filter:
                # print(d["id"])
                # print(args.id_filter)
                if d["id"] == args.id_filter:
                    image_to_draw.append(d)
            else:
                image_to_draw.append(d)
    os.makedirs(args.output_path, exist_ok=True)
    for i, image in enumerate(image_to_draw):
        # print(image)
        boxes = image["boxes"]
        image_id = image["id"]
        if args.skip_no_boxes:
            if len(boxes) == 0:
                continue
        if args.skip_score_below:
            scores = image["scores"]
            if len(boxes) == 0:
                boxes = [x for y, x in zip(boxes, scores) if y > args.skip_score_below]

        image = imageio.imread(os.path.join(args.image_path, image["path"]))
        if len(image.shape) != 3:
            continue
        if image.shape[-1] != 3:
            continue

        plot_prediction_images(image=image, output_path=os.path.join(args.output_path, f"{image_id}.jpg"), boxes=boxes)
        # print(image.shape)
        print(i)
    return 0


if __name__ == "__main__":
    sys.exit(main())