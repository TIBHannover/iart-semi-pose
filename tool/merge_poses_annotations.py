import os
import sys
import re
import argparse
import json
import copy

import torch
import numpy as np

sys.path.append(".")
from utils.box_ops import box_iou, points_in_box


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-p", "--prediction_path", help="verbose output")
    parser.add_argument("-a", "--annotation_path", help="verbose output")
    parser.add_argument("-s", "--skip_missing", action="store_true", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    prediction_dict = {}
    merge_fields = ["boxes", "keypoints", "selected", "labels", "scores", "boxes_id"]
    with open(args.prediction_path, "r") as f:
        for line in f:
            # print(line)
            d = json.loads(line)
            if d["id"] not in prediction_dict:
                prediction_dict[d["id"]] = d
            else:
                existing_dict = prediction_dict[d["id"]]

                for field in merge_fields:
                    if field in existing_dict and field in d:
                        existing_dict[field].extend(d[field])
                    elif field in d:
                        existing_dict[field] = d[field]

    with open(args.annotation_path, "r") as f:
        with open(args.output_path, "w") as f_out:
            for line in f:
                d = json.loads(line)

                if d["id"] not in prediction_dict:
                    if args.skip_missing:
                        continue
                    else:
                        f_out.write(json.dumps(d) + "\n")
                else:
                    p = prediction_dict[d["id"]]
                    # print("#########################")
                    # print("#########################")
                    # print("#########################")
                    # print(p)
                    sorted_poses = list(
                        zip(
                            *sorted(
                                zip(
                                    p["boxes_id"],
                                    p["keypoints"],
                                    p["labels"],
                                    p["scores"],
                                    p["selected"],
                                    p["origin_size"],
                                ),
                                key=lambda x: x[0][0],
                            )
                        )
                    )
                    predictions = {
                        "boxes_id": list(sorted_poses[0]),
                        "keypoints": list(sorted_poses[1]),
                        "labels": list(sorted_poses[2]),
                        "scores": list(sorted_poses[3]),
                        "selected": list(sorted_poses[4]),
                        "origin_size": list(sorted_poses[5]),
                    }
                    # print("#########################")
                    # print(predictions)
                    # exit()

                    if "boxes" in d:
                        d["boxes_scores"] = d["scores"]
                        d["boxes_labels"] = d["labels"]
                    results = {**d, **predictions}
                    f_out.write(json.dumps(results) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())