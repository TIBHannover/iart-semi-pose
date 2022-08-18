import os
import sys
import re
import argparse
import uuid
import hashlib
import json


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--input_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    file_list = []
    for root, dirs, files in os.walk(args.input_path):
        for f in files:
            file_path = os.path.join(root, f)

            hash = md5(file_path)
            data = {"id": uuid.uuid4().hex, "path": f, "hash": hash}
            file_list.append(data)
    with open(args.output_path, "w") as f:
        for d in file_list:
            f.write(json.dumps(d) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())