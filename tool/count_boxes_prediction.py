import os
import sys
import re
import argparse
import json
import torch
from PIL import Image

sys.path.append(".")
from utils.box_ops import box_iou, boxes_aspect_ratio, boxes_scale, box_xyxy_to_cxcywh


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--input_path", help="verbose output")
    parser.add_argument("--input_image_path", help="verbose output")
    parser.add_argument("--min_area", type=float)
    parser.add_argument("--max_iou", type=float)
    parser.add_argument("--min_score", type=float)
    args = parser.parse_args()
    return args


def _load_image(root, sample) -> Image.Image:
    return Image.open(os.path.join(root, sample.get("path"))).convert("RGB")


def main():
    args = parse_args()

    target_height = 384
    target_width = 288
    ar = target_width / target_height
    count = 0
    skiped = 0
    with open(args.input_path) as f:
        for line in f:
            sample = json.loads(line)
            try:
                image = _load_image(args.input_image_path, sample)
            except:
                continue
            target = {"image_id": sample.get("id")}

            w, h = image.size
            target["size"] = torch.tensor([h, w])
            target["origin_size"] = torch.tensor([h, w])
            boxes = sample.get("boxes")
            scores = sample.get("scores")

            if boxes is None or len(boxes) == 0:
                continue

            if scores is None:
                scores = [1.0 for x in boxes]

            delete_boxes = None
            if args.max_iou:
                # print(torch.as_tensor(boxes))
                iou, _ = box_iou(torch.as_tensor(boxes), torch.as_tensor(boxes))
                diag = torch.diagonal(iou)
                diag[:] = 0
                # print(f"iou {iou}")
                max_iou = torch.max(iou, 0).values
                # print(f"max_iou {max_iou}")
                delete_boxes = max_iou > args.max_iou
                # print(f"delete_boxes {delete_boxes}")

            for i, (box, score) in enumerate(zip(boxes, scores)):
                if delete_boxes is not None:
                    if delete_boxes[i]:
                        # print("BOX iou to large")
                        skiped += 1
                        continue
                if args.min_score:
                    if score < args.min_score:
                        # print(f"BOX score to small {score}")
                        skiped += 1
                        continue
                box = torch.squeeze(torch.as_tensor(box))
                box_target = {"boxes": torch.unsqueeze(box, dim=0), "flipped": False, "boxes_id": [i]}

                # w, h = image.size
                # target["size"] = torch.tensor([h, w])
                ar_box = boxes_aspect_ratio(box, ar)
                scaled_box = boxes_scale(ar_box, 1.25, size=target["size"])
                scaled_box_cxcywh = box_xyxy_to_cxcywh(scaled_box)
                scaled_box = scaled_box.numpy().tolist()
                scaled_box_wh = scaled_box_cxcywh.numpy().tolist()
                # print(f"## {args.min_area} {scaled_box_wh[3] * scaled_box_wh[2]}")
                if args.min_area:
                    if scaled_box_wh[3] * scaled_box_wh[2] < args.min_area:
                        # print("BOX to small")
                        skiped += 1
                        continue
                count += 1
            print(f"{count} {skiped}")
            # exit()

    return 0


if __name__ == "__main__":
    sys.exit(main())