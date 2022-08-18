import os
import sys
import re
import argparse
import json
import random

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument("-i", "--input_path", help="verbose output")
    parser.add_argument("--image_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")

    parser.add_argument("--first_split", help="verbose output")
    parser.add_argument("-f", "--fractions", type=float, nargs='+', help="verbose output")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    random.seed(42)

    existing_files = {}
    if args.first_split:
        with open(args.first_split, 'r') as f:
            for line in f:
                data = json.loads(line)
                hash = data['hash']
                existing_files[hash] = data
    
    data_split = [[] for x in args.fractions]
    
    available = []
    count = 0
    with open(args.input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            rel_path = data.get("path","")
            hash = data.get("hash","")
            if args.image_path:
                rel_path = os.path.relpath(os.path.abspath(data.get("path","")), os.path.abspath(args.image_path))
            
            if hash in existing_files:
                data_split[0].append({**data, 'path':rel_path})
            else:
                available.append({**data, 'path':rel_path})

            count+= 1

    counts = [int(count*f) for f in args.fractions]
    counts[0] -= len(data_split[0])
    assert counts[0] > 0, "Fraction is not possible"
    random.shuffle(available)
    start = 0
    for x in range(len(counts)):
        y = available[start:start+counts[x]]
        data_split[x].extend(y)
        start+= counts[x]
    
    for i, split in enumerate(data_split):
        with open(os.path.join(args.output_path, f"{i}.jsonl"), 'w') as f:
            for entry in split:
                f.write(json.dumps(entry)+'\n')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())