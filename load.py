import lzma
import sys
from itertools import islice

def print_xz_ngram_file(file_path):
    with lzma.open(file_path, "rt") as f:
        for line in islice(f.readlines(), 100):
            print(line.strip())

if __name__ == '__main__':
    argv = sys.argv[1:]
    if argv:
        print_xz_ngram_file(argv[0])