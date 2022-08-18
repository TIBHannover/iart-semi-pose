import os
import sys
import re
import argparse
import hashlib
import json
import uuid

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument("-i", "--input_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
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

    single = {}
    hashs = {}
    with open(args.output_path, 'w') as f_out:
        for root, dirs, files in os.walk(args.input_path):
            for f in files:
                file_path = os.path.join(root, f)
                if os.path.splitext(file_path)[1].lower() not in [".jpg", ".png"]:
                    print(file_path)
                    continue
                try:
                    hash = md5(file_path)
                except KeyboardInterrupt:
                    exit()
                except:
                    continue
                norm_path = os.path.normpath(file_path)
                split_path = norm_path.split(os.sep)
                data = {'id':uuid.uuid4().hex, 'path': norm_path, 'hash': hash}
                # print(split_path)
                single[split_path[-1]] = data
                    # exit()
                print(len(single))
                if hash not in hashs:
                    hashs[hash] = data
                    f_out.write(json.dumps(data)+'\n')
            
    
    return 0

if __name__ == '__main__':
    sys.exit(main())