import os
import sys
import re
import argparse
import json
import random
import hashlib


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument("-i", "--input_path", help="verbose output")
    parser.add_argument("--image_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    parser.add_argument("-s", "--skip", help="verbose output")

    args = parser.parse_args()
    return args

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    args = parse_args()


    random.seed(42)

    dirs = [x[0] for x in os.walk(args.image_path)]
    print(dirs)
    print(len(dirs))

    # exit()

    with open(args.output_path, 'w') as f_out:
        with open(args.input_path, 'r') as f:
            for line in f:
                found = False
                data = json.loads(line)
                if args.skip:
                    filter = re.match(args.skip, data['filename'])
                    if filter:
                        print(data['filename'])
                        continue
                path_split = data['filename'].split(os.sep)
                try_path = None
                for x in range(len(path_split)):
                    # print(x)
                    split = os.sep.join(path_split[x:])

                    for dir in dirs:
                        try_path = os.path.join(dir, split)
                        # print(try_path)
                        if os.path.exists(try_path):
                            found = True
                            # print(f"found {try_path}")
                            break
                    if found:
                        break

                if found:
                    hash = md5(try_path)
                    rel_path = os.path.relpath(os.path.abspath(try_path), os.path.abspath(args.image_path))
                    f_out.write(json.dumps({'path': rel_path,  'hash':hash})+'\n')

                if not found:
                    print(path_split)

    return 0

if __name__ == '__main__':
    sys.exit(main())