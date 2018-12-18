import os
from shutil import copyfile


INPUT_DIR = '/tmp/dataset/lfw'
OUTPUT_DIR = '/tmp/dataset/lfw_set1'
PAIR_LIST = '/tmp/dataset/pairs.txt'

# get lfw pair list
lfw_pairs = []
with open(PAIR_LIST, 'r') as f:
    for line in f.readlines()[1:]:
        pair = line.strip().split()
        lfw_pairs.append(pair)

# choose the first 600 pairs for test set1
for idx in range(600):
    pair = lfw_pairs[idx]
    if len(pair) == 3:
        name0 = pair[0]
        name1 = pair[0]
        number0 = pair[1]
        number1 = pair[2]
    elif len(pair) == 4:
        name0 = pair[0]
        name1 = pair[2]
        number0 = pair[1]
        number1 = pair[3]
    else:
        raise ValueError('Error')
    src_img0 = os.path.join(INPUT_DIR, name0, name0 + '_' + '%04d' % int(number0) + '.' + 'jpg')
    src_img1 = os.path.join(INPUT_DIR, name1, name1 + '_' + '%04d' % int(number1) + '.' + 'jpg')
    dst_dir0 = os.path.join(OUTPUT_DIR, name0)
    dst_dir1 = os.path.join(OUTPUT_DIR, name1)
    dst_img0 = os.path.join(dst_dir0, name0 + '_' + '%04d' % int(number0) + '.' + 'jpg')
    dst_img1 = os.path.join(dst_dir1, name1 + '_' + '%04d' % int(number1) + '.' + 'jpg')
    if not os.path.exists(dst_dir0):
        os.makedirs(dst_dir0)
    if not os.path.exists(dst_dir1):
        os.makedirs(dst_dir1)
    copyfile(src_img0, dst_img0)
    copyfile(src_img1, dst_img1)
