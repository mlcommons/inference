import os.path as osp
import os
import csv
import argparse
import sys
import sox
from random import shuffle
from model.utils import AverageMeter, make_file, write_line, make_folder


def format_entry(entry, root):
    base = osp.basename(entry[0])
    folder = osp.basename(osp.dirname(entry[0]))
    base = base.split('.')[0] + ".txt"
    new_file = osp.join(root, folder, base)
    new_entry = entry[2].upper()
    return new_file, new_entry


def make_manifest(inputfile, root, idx):
    if idx == -1:
        idx = ""
    else:
        idx = '_held{}'.format(idx)
    base = osp.basename(inputfile)
    base = base + idx
    manifest_file = osp.join(root, base)
    make_folder(manifest_file)
    make_file(manifest_file)
    return manifest_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='some manifest file with the first col containing the wav file path')
    parser.add_argument('--hold_idx', default=-1, type=int)
    parser.add_argument('--stats', dest='stats', action='store_true')
    parser.add_argument('--scramble_repeat', default=-1, type=int)
    args = parser.parse_args()

    root = os.getcwd()  # the root is the current working directory
    filepath = osp.join(os.getcwd(), args.file)
    print("\n\nOpening: {}".format(filepath))
    print("Root: {}".format(root))
    if args.stats:
        manifest_file = filepath + "_stats"
        make_folder(manifest_file)
        make_file(manifest_file)
        audio_dur = AverageMeter()
    elif args.scramble_repeat > 1:
        manifest_file = filepath + "_scram_rep"
        make_folder(manifest_file)
        make_file(manifest_file)
    else:
        manifest_file = make_manifest(filepath, root, args.hold_idx)
    print("Manifest made: {}".format(manifest_file))

    with open(filepath) as f:
        summary = csv.reader(f, delimiter=',')
        tot = 0
        hold_file = ""
        hold_entry = ""
        repeat_store = []
        for i, row in enumerate(summary):
            tot += 1
            if args.hold_idx == i:
                hold_file, hold_entry = row
            elif args.scramble_repeat > 1:
                repeat_store.append(row[0] + "," + row[1])
        if args.scramble_repeat > 1:
            for i in range(args.scramble_repeat + 1):
                shuffle(repeat_store)
                if i == 0:
                    # First is the warmup pad
                    for j, row in enumerate(repeat_store):
                        if j >= 50:
                            break
                        write_line(manifest_file, row + "\n")
                else:
                    for j, row in enumerate(repeat_store):
                        write_line(manifest_file, row + "\n")
                print("Rep {}".format(i))
            exit(0)
        cur = 0
        f.seek(0)
        new_file = hold_entry
        for row in summary:
            if cur == 0:
                cur += 1
                continue
            if not args.stats:
                if args.hold_idx != -1:
                    write_line(manifest_file, hold_file + "," + hold_entry + "\n")
                else:
                    exit(1)
                    (new_file, new_entry) = format_entry(row, root)
                    make_folder(new_file)
                    make_file(new_file, new_entry)
            else:
                seconds = sox.file_info.duration(row[0])
                audio_dur.update(seconds)
                new_file = "{},{}".format(seconds, audio_dur.avg)
                write_line(manifest_file, row[0] + "," + new_file + "\n")
            sys.stdout.write("\r[{}/{}] {}         ".format(cur, tot, new_file))
            sys.stdout.flush()
            cur += 1
        sys.stdout.write("\r[{}/{}] {}         ".format(cur, tot, new_file))
        sys.stdout.flush()
        print("\n")
