import os
import sys
import re
import argparse
import json
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--image_path")
    parser.add_argument("-a", "--annotation_path")
    parser.add_argument("-o", "--output_path")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # read annotation

    with open(args.annotation_path, "r") as f:
        annotation = json.loads(f.read())
    # print(annotation.keys())
    # exit()
    annotation_images = {}
    for x in annotation["annotations"]:
        image_id = x["image_id"]
        if image_id not in annotation_images:
            annotation_images[image_id] = []
        annotation_images[image_id].append(x)

    images = {}
    for x in annotation["images"]:
        images[x["id"]] = x

    # parse image folder
    paths = {}
    # print("####1")
    for root, dirs, files in os.walk(args.image_path):
        for f in files:
            # print("####")
            file_path = os.path.join(root, f)
            m = re.match(r"^(\d+)_.*?$", f)
            if m:

                image_id = int(m.group(1))
                if image_id not in paths:
                    paths[image_id] = []

                paths[image_id].append(file_path)
    result_images = []
    result_annotations = []
    new_image_id = 0
    new_annotation_id = 0
    for image_id in paths.keys():
        for path in paths[image_id]:
            print(image_id)
            if image_id not in images:
                print(f"Image not found {image_id}")
                continue

            image = copy.deepcopy(images[image_id])
            image["file_name"] = os.path.basename(path)
            image["id"] = new_image_id
            result_images.append(image)

            if image_id not in annotation_images:
                print(f"No annotation found for {image_id}")
            else:
                for anno in annotation_images[image_id]:

                    anno = copy.deepcopy(anno)
                    anno["image_id"] = new_image_id
                    anno["id"] = new_annotation_id
                    result_annotations.append(anno)
                    new_annotation_id += 1

            new_image_id += 1

    annotation["images"] = result_images
    annotation["annotations"] = result_annotations

    with open(args.output_path, "w") as f:
        f.write(json.dumps(annotation))

    return 0


if __name__ == "__main__":
    sys.exit(main())