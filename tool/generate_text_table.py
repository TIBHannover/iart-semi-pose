import os
import sys
import re
import argparse
import json

# coco_rc_199999.json
# coco_semi_09_negative_09_loss_05_rc_199999.json
# coco_style_rc_199999.json
# coco_style_semi_09_negative_09_loss_05_rc_199999.json
# style_rc_199999.json
# style_semi_09_negative_09_loss_05_rc_199999.json
# peopleart_rc_199999.json
# peopleart_semi_09_negative_09_loss_05_rc_v3_199999.json
# /nfs/home/springsteinm/output/iart/pose/results/peopleart/coco_rc_199999.json /nfs/home/springsteinm/output/iart/pose/results/peopleart/coco_semi_09_negative_09_loss_05_rc_199999.json /nfs/home/springsteinm/output/iart/pose/results/peopleart/coco_style_rc_199999.json /nfs/home/springsteinm/output/iart/pose/results/peopleart/coco_style_semi_09_negative_09_loss_05_rc_199999.json /nfs/home/springsteinm/output/iart/pose/results/peopleart/style_rc_199999.json /nfs/home/springsteinm/output/iart/pose/results/peopleart/style_semi_09_negative_09_loss_05_rc_199999.json /nfs/home/springsteinm/output/iart/pose/results/peopleart/peopleart_rc_199999.json /nfs/home/springsteinm/output/iart/pose/results/peopleart/peopleart_semi_09_negative_09_loss_05_rc_v3_199999.json


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    parser.add_argument("-i", "--inputs", nargs="+", help="verbose output")
    parser.add_argument("-k", "--keys", nargs="+", help="verbose output")
    parser.add_argument("-t", "--type", choices=("keypoints", "boxes"))
    parser.add_argument("--siunitx", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.type == "boxes":
        keys = [
            "val/bbox/Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
            "val/bbox/Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
            "val/bbox/Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
            "val/bbox/Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
            "val/bbox/Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
            "val/bbox/Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
            "val/bbox/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        ]

    #   "val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]": 0.5257687647846357,
    #   "val/kpt/Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ]": 0.6392049994796355,
    #   "val/kpt/Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ]": 0.5735161413429696,
    #   "val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]": 0.030798403342050588,
    #   "val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]": 0.5349510475859972,
    #   "val/kpt/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]": 0.7464237516869097,
    #   "val/kpt/Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ]": 0.8556005398110661,
    #   "val/kpt/Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ]": 0.796221322537112,
    #   "val/kpt/Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]": 0.27,
    #   "val/kpt/Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]": 0.7529411764705882

    elif args.type == "keypoints":
        keys = [
            # "val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]"
            "val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]",
            "val/kpt/Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ]",
            "val/kpt/Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ]",
            "val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]",
            "val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]",
            "val/kpt/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]",
        ]

    data = []
    for i, x in enumerate(args.inputs):
        exp_results = []
        with open(x) as f:
            d = json.load(f)
            for k in keys:
                exp_results.append(float(d[k]))

        if args.siunitx:
            exp_results = [f"\\num{{{round(x,4)}}}" for x in exp_results]

        data.append(exp_results)

    for line in data:
        print(" & ".join(line))
    # print(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
