  
# Copyright 2018 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import json
import os
import sys
import tensorflow as tf
import time
import argparse
import PIL
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

sys.path.append("..")
import ops as utils_ops

_WARMUP_NUM_LOOPS = 3
PATH_TO_TEST_IMAGES_DIR = ""


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  image_np = np.array(image.getdata())
  return image_np.reshape(im_height, im_width, 3).astype(np.uint8)


def batch_from_image(file_name, batch_data):
  """Produce a batch of data from the passed image file.

  Args:
    file_name: string, path to file containing a JPEG image
    batch_data: batch to append this image to

  Returns:
    Float array representing copies of the image with shape
      [current_size, output_height, output_width, num_channels]
  """
  image_np = np.array(PIL.Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR,file_name)).convert('RGB').resize((300, 300))).astype(np.uint8)
    # BGR2RGB
  image_np = image_np[:, :, [2, 1, 0]]
  batch_data.append(image_np)
  return batch_data

def log_stats(latency_type,graph_name, log_buffer, timings, batch_size):
  """Write stats to the passed log_buffer.

  Args:
    graph_name: string, name of the graph to be used for reporting.
    log_buffer: filehandle, log file opened for appending.
    timings: list of floats, times produced for multiple runs that will be
      used for statistic calculation
    batch_size: int, number of examples per batch
  """
  times = np.array(timings)
  steps = len(times)
  speeds = batch_size / times
  time_mean = np.mean(times)
  time_med = np.median(times)

  speed_mean = np.mean(speeds)
  speed_med = np.median(speeds)

  msg = ("\n============%s==============\n"
         "\t batchsize %d\n"
         "  fps \tmedian: %.1f, \tmean: %.1f"  # pylint: disable=line-too-long
         "  latency \tmedian: %.5f, \tmean: %.5f\n"  # pylint: disable=line-too-long
        ) % ( latency_type, batch_size,
             speed_med, speed_mean,
             time_med, time_mean)

  log_buffer.write(msg)

def run_inference(image, graph):
  timings = []
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      tf.logging.info("MlPerf:::Starting execution")
      tf.logging.info("Starting Warmup cycle")
      for _ in range(_WARMUP_NUM_LOOPS):
        output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})
      tf.logging.info(":::MLPv0.5.0 Starting prediction timing.")
      predict_start = time.time()
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})
      timings.append(time.time() - predict_start)
      print("Time:",timings)
      tf.logging.info(":::MLPv0.5.0Prediction Timing loop done!")

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]


  return output_dict, timings

def main(_):
    print (FLAGS.mode)
    _LOG_FILE = "MlPerf_log_mobilenet_ssd.txt"
    global PATH_TO_TEST_IMAGES_DIR
    PATH_TO_TEST_IMAGES_DIR = os.path.join(FLAGS.data, "val2017")
    print (PATH_TO_TEST_IMAGES_DIR)
    log_buffer = open(os.path.join('output/', _LOG_FILE), "a")
    log_buffer.write("\n==========:::MLPv0.5.0 SSD-Mobilenetv1 Inference Performance=========\n")
    log_buffer.write(("\n:::MLPv0.5.0 Batch size %d:") % (FLAGS.batch_size))
    log_buffer.write(("\n:::MLPv0.5.0 Image size: 300"))
    log_buffer.write(("\n:::MLPv0.5.0 Box score threshold: 0.01"))
    files_dir = PATH_TO_TEST_IMAGES_DIR
    TEST_IMAGE_PATHS = files = os.listdir(files_dir)
    if(FLAGS.mode =="performance"):
        TEST_IMAGE_PATHS = TEST_IMAGE_PATHS[0:FLAGS.run_size]
    # What model to download.
    MODEL_DIRNAME = FLAGS.model

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = os.path.join(FLAGS.model, "ssd_mobilenet_v1_coco_2017_11_17","frozen_inference_graph.pb")

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    predicted_latencies = []
    overall_latencies = []
    detections = []
    batch_data=[]
    step = 1
    batch_imagenames = []
    batch_num=1
    batch_start_flag = True
    batch_size = FLAGS.batch_size
    if(FLAGS.mode =="evaluation"):
        batch_size=1
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_path)).convert('RGB')
      batch_imagenames.append(image_path)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      cols = image_np.shape[1]
      rows = image_np.shape[0]
      if(batch_start_flag):
        overall_start = time.time()
        batch_start_flag = False
      batch_data = batch_from_image(image_path, batch_data)
      if ((step%batch_size)==0 and (len(files)-step>=0)):
          # Actual detection.
          #MLPerf Predicted Latencies
          print("Images in this batch:", batch_imagenames)
          output_dict, predicted_latency_for_batch = run_inference( batch_data, detection_graph)
          predicted_latencies.append(predicted_latency_for_batch)
          overall_latencies.append(time.time() - overall_start)

          #COCO Evaluation
          if(FLAGS.mode =="evaluation"):
              num_detections = output_dict['num_detections']

              #print(num_detections)
              for i in range(num_detections):
                    classId = int(output_dict['detection_classes'][i])
                    score = float(output_dict['detection_scores'][i])
                    bbox = [float(v) for v in output_dict['detection_boxes'][i]]
                    if score > 0.01:
                        x = bbox[1] * cols
                        y = bbox[0] * rows
                        w = bbox[3] * cols - x
                        h = bbox[2] * rows - y
                        detections.append({
                          "image_id": int(image_path.rstrip('0')[:image_path.rfind('.')]),
                          "category_id": classId,
                          "bbox": [x, y, w, h],
                          "score": score
                        })
          batch_data=[]
          batch_imagenames = []
          batch_num+=1
          batch_start_flag = True
      step = step + 1
    if(FLAGS.mode =="performance"):
        log_stats(":::MLPv0.5.0 Predict Latency",od_graph_def, log_buffer, predicted_latencies, FLAGS.batch_size)
        log_stats(":::MLPv0.5.0 Overall Latency",od_graph_def, log_buffer, overall_latencies, FLAGS.batch_size)

    ### Evaluation part ############################################################
    old_stdout = sys.stdout
    sys.stdout = log_buffer
    if(FLAGS.mode =="evaluation"):
        # %matplotlib inline
        log_buffer.write("\n==========:::MLPv0.5.0 SSD-Mobilenetv1 Accuracy=========\n")
        with open('tf_result.json', 'w') as f:
            json.dump(detections, f)

        annType = ['segm','bbox','keypoints']
        annType = annType[1]      #specify type here
        prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
        print('Running demo for *%s* results.'%(annType))

        #initialize COCO ground truth api
        val_annotate =  os.path.join(FLAGS.data, "annotations/instances_val2017.json")
        print (val_annotate)
        cocoGt = COCO(annotation_file=val_annotate)
        #cocoGt=COCO(args.annotations)

        #initialize COCO detections api
        for resFile in ['tf_result.json']:
            print(resFile)
            cocoDt=cocoGt.loadRes(resFile)

            cocoEval = COCOeval(cocoGt,cocoDt,annType)
            cocoEval.evaluate()
            cocoEval.accumulate()
            log_buffer.write("\n==========:::MLPv0.5.0 COCO Eval Summary=========\n")
            cocoEval.summarize()
    sys.stdout= old_stdout



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--mode', "-o",
      type=str,
      default='performance',
      help='Set to performance or evaluation.'
  )
  parser.add_argument(
    "--batch_size", "-bs", type=int, default=1,
    help="[default: %(default)s] Batch size for inference. For evaluation mode the batch size is fixed to 1.",
    metavar="<BS>"
    )
  parser.add_argument(
    "--run_size", "-rs", type=int, default=5000,
    help="[default: %(default)s] Total images for inference. For evaluation mode the run size is fixed to all validation images.",
    metavar="<RS>"
  )
  parser.add_argument('--data', '-d', type=str, default='coco',
    help='path to test and training data files')
  parser.add_argument('--model', '-m', type=str, default='/ssd_model',
    help=' Path to graph file (frozen_graph)')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

