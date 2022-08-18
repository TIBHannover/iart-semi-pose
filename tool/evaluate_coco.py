import os
import sys
import re
import argparse
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-g", "--gt_annotation", help="verbose output")
    parser.add_argument("-p", "--prediction", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    annType = ["segm", "bbox", "keypoints"]
    annType = annType[2]  # specify type here

    cocoGt = COCO(args.gt_annotation)

    # initialize COCO detections api

    cocoDt = cocoGt.loadRes(args.prediction)

    imgIds = sorted(cocoGt.getImgIds())
    # imgIds = imgIds[0:100]
    # imgId = imgIds[np.random.randint(100)]

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if args.output_path:
        resultDict = {
            "val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]": cocoEval.stats[0],
            "val/kpt/Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ]": cocoEval.stats[1],
            "val/kpt/Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ]": cocoEval.stats[2],
            "val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]": cocoEval.stats[3],
            "val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]": cocoEval.stats[4],
            "val/kpt/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]": cocoEval.stats[5],
            "val/kpt/Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ]": cocoEval.stats[6],
            "val/kpt/Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ]": cocoEval.stats[7],
            "val/kpt/Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]": cocoEval.stats[8],
            "val/kpt/Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]": cocoEval.stats[9],
        }
        with open(args.output_path, "w") as f:
            f.write(json.dumps(resultDict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())