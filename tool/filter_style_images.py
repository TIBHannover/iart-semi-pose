import os
import sys
import re
import argparse
import shutil
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--image_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    n = 2
    args = parse_args()

    paths = {}
    files_to_delete = []
    # print("####1")
    for root, dirs, files in os.walk(args.image_path):
        for f in files:
            # print("####")
            file_path = os.path.join(root, f)
            m = re.match(r"^(\d+)_.*?$", f)
            if m:
                coco_id = m.group(1)
                if coco_id not in paths:
                    paths[coco_id] = []
                if len(paths[coco_id]) >= n:
                    files_to_delete.append(file_path)
                else:
                    paths[coco_id].append(file_path)
    # print(len(files_to_delete))
    # exit()
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

        count = 0
        for k, v in tqdm(paths.items()):
            count += len(v)
            for file_path in v:
                shutil.copyfile(file_path, os.path.join(args.output_path, os.path.basename(file_path)))
    else:
        for x in tqdm(files_to_delete):
            print(x)
            os.remove(x)

    return 0


if __name__ == "__main__":
    sys.exit(main())