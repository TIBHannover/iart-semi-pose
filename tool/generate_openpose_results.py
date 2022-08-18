# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import

        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append("/openpose/build/python/")
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print(
            "Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?"
        )
        raise e
except Exception as e:
    print(e)
    sys.exit(-1)


import os
import sys
import re
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument(
        "--image_dir",
        default="../../../examples/media/",
        help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).",
    )
    parser.add_argument(
        "--annotation",
        help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).",
    )
    parser.add_argument("-o", "--output")
    parser.add_argument("--image_output")
    parser.add_argument("-t", "--threshold", default=0.1, type=float)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/openpose/models/"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    start = time.time()
    a = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
    }
    coco_mapping = {
        0: 0,
        1: 16,
        2: 15,
        3: 18,
        4: 17,
        5: 5,
        6: 2,
        7: 6,
        8: 3,
        9: 7,
        10: 4,
        11: 12,
        12: 9,
        13: 13,
        14: 10,
        15: 14,
        16: 11,
    }

    # coco_mapping = {
    #     0: 0,
    #     1: 15,
    #     2: 16,
    #     3: 17,
    #     4: 18,
    #     5: 2,
    #     6: 5,
    #     7: 3,
    #     8: 6,
    #     9: 4,
    #     10: 7,
    #     11: 9,
    #     12: 12,
    #     13: 10,
    #     14: 13,
    #     15: 11,
    #     16: 14,
    # }

    coco = COCO(args.annotation)
    ids = list(sorted(coco.imgs.keys()))

    # Process and display images
    results = []
    for id in ids:
        # print(f"###### {id}")
        path = coco.loadImgs(id)[0]["file_name"]
        datum = op.Datum()
        imageToProcess = cv2.imread(os.path.join(args.image_dir, path))
        # print(imageToProcess.shape)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        # print(datum)
        keypoints = datum.poseKeypoints
        if keypoints is None:
            continue

        for person in range(keypoints.shape[0]):

            coco_keypoints = np.zeros([17, 3])
            scores = []
            for c, o in coco_mapping.items():
                coco_keypoints[c] = keypoints[person, o]
                if coco_keypoints[c, 2] > args.threshold:
                    scores.append(coco_keypoints[c, 2])
                    coco_keypoints[c, 2] = 2
                else:
                    coco_keypoints[c, 2] = 0
            results.append(
                {
                    "image_id": id,
                    "category_id": 1,
                    "keypoints": coco_keypoints.flatten().tolist(),
                    "score": np.mean(scores),
                }
            )
        if args.image_output is not None:
            os.makedirs(args.image_output, exist_ok=True)
            cv2.imwrite(os.path.join(args.image_output, f"{id}.jpg"), datum.cvOutputData)

        # print("Body keypoints: \n" + str(keypoints) + str(coco_keypoints))
    # print(results)
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    if args.output:
        with open(args.output, "w") as f:
            f.write(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


#     # Construct it from system arguments
#     # op.init_argv(args[1])
#     # oppython = op.OpenposePython()

#     # Starting OpenPose
#     opWrapper = op.WrapperPython()
#     opWrapper.configure(params)
#     opWrapper.start()

#     # Read frames on directory
#     imagePaths = op.get_images_on_directory(args[0].image_dir)
#     start = time.time()

#     # Process and display images
#     for imagePath in imagePaths:
#         datum = op.Datum()
#         imageToProcess = cv2.imread(imagePath)
#         datum.cvInputData = imageToProcess
#         opWrapper.emplaceAndPop(op.VectorDatum([datum]))

#         print("Body keypoints: \n" + str(datum.poseKeypoints))

#     end = time.time()
#     print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
# except Exception as e:
#     print(e)
#     sys.exit(-1)
