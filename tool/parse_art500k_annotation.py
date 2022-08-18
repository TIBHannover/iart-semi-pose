import os
import sys
import re
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument("-i", "--input_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args

def main():
    key_index = 2
    args = parse_args()
    index = {}
    count = {}
    with open(args.input_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='|')
        for row in spamreader:
            for i in range(len(row)):
                if i not in count:
                    count[i] = 0
                if row[i]:
                    count[i] += 1
            try:
                index[row[key_index]] = row
            except:
                print('#######')
                print(row)


    print(len(index))
    print(count)    
    return 0

if __name__ == '__main__':
    sys.exit(main())