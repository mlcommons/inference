# coding=utf-8
# Copyright 2018 Changan Wang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

def count_split_examples(split_path, file_prefix='.tfrecord'):
    '''
    func: count number of samples
      Args:
        split_path: split path
        file_prefix: prefix of file path
      return:
        num_samples: number of samples
    '''
    num_samples = 0
    tfrecords_to_count = tf.gfile.Glob(os.path.join(split_path, file_prefix))
    for tfrecord_file in tfrecords_to_count:
        num_samples += len(tf.python_io.tf_record_iterator(tfrecord_file))
    return num_samples

if __name__ == '__main__':
    print('train:', count_split_examples('../tfrecords', 'voc_0712_train-?????-of-?????'))
    print('val:', count_split_examples('../tfrecords', 'voc_0712_val-?????-of-?????'))
