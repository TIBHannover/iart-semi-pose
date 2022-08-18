import os
import sys
import re
import argparse

from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-a", "--annotation_path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    coco = COCO(args.annotation_path)
    ids = list(sorted(coco.imgs.keys()))

    images = 0
    keypoints = 0
    boxes = 0
    for id in ids:
        annotation = coco.loadAnns(coco.getAnnIds(id))
        images += 1
        boxes += len(annotation)
        for anno in annotation:
            if "num_keypoints" in anno:
                keypoints += anno["num_keypoints"]

    print(f"images {images}")
    print(f"boxes {boxes}")
    print(f"keypoints {keypoints}")
    # keypoints += annotation
    # exit()

    return 0


if __name__ == "__main__":
    sys.exit(main())