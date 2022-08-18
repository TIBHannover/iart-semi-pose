import os
import sys
import re
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--input_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data = json.load(open(args.input_path))

    cat_lut = {0: 1}

    data["categories"][0]["id"] = 0
    new_annotations = []
    for anno in data["annotations"]:
        anno["category_id"] = cat_lut[anno["category_id"]]
        new_annotations.append(anno)

    data["annotations"] = new_annotations
    with open(args.output_path, "w") as f:
        f.write(json.dumps(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())