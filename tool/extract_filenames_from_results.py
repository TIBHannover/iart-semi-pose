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
    filenames = []
    for root, dirs, files in os.walk(args.input_path):
        for f in files:
            file_path = os.path.join(root, f)
            if not re.match(r"^.*?\.json$", f):
                print(file_path)
                continue
            with open(file_path, "r") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    print(file_path)
                    continue
                # print(data)

                for _, entry in data.get("retrievalfiles", {}).items():
                    filename = entry.get("filename")
                    if filename is not None:
                        filenames.append(filename)
    print(len(filenames))
    print(len(list(set(filenames))))
    os.makedirs((os.path.dirname(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        for x in list(set(filenames)):
            f.write(json.dumps({"filename": x}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
