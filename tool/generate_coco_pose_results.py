import os
import sys
import re
import argparse
import json

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate COCO results from box and keypoints predictions")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-b", "--box_prediction", help="verbose output")
    parser.add_argument("-k", "--keypoint_prediction", help="verbose output")
    parser.add_argument("-t", "--threshold", default=0.9, type=float, help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    parser.add_argument("--keypoint_scores", action="store_true", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    box_lut = {}
    if args.box_prediction:
        with open(args.box_prediction, "r") as f_in:
            for line in f_in:
                data = json.loads(line)
                box_lut[str(data["id"])] = {}
                for i, x in enumerate(zip(data["boxes"], data["scores"])):
                    box_lut[str(data["id"])][str(i)] = {"box": x[0], "score": x[1]}

    result = []
    with open(args.keypoint_prediction, "r") as f_in:
        for line in f_in:
            data = json.loads(line)
            for i, x in enumerate(zip(data["keypoints"], data["scores"], data["boxes_id"], data["labels"])):
                # print(data)
                keypoints = np.zeros(17 * 3)
                keypoint_scores = []
                for k, s, l in zip(x[0], x[1], x[3]):
                    # print(k, s, l)
                    if s > args.threshold:
                        keypoints[3 * l : 3 * l + 2] = k
                        keypoints[3 * l + 2] = 2
                        keypoint_scores.append(s)
                # print(keypoints)
                # exit()

                box_score = 1.0
                if str(data["id"]) in box_lut:
                    if str(x[2][0]) in box_lut[str(data["id"])]:
                        box_score = box_lut[str(data["id"])][str(x[2][0])]["score"]

                score = box_score
                if args.keypoint_scores:
                    score = np.mean(keypoint_scores)
                result.append(
                    {"image_id": data["id"], "category_id": 1, "keypoints": keypoints.tolist(), "score": score}
                )
    # print(json.dumps(result, indent=2))
    with open(args.output_path, "w") as f_out:
        f_out.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())