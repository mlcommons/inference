import os
import json
import argparse
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# COCO category mapping (YOLO index -> COCO category_id)
COCO_MAP = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19,
    18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28,
    26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38,
    34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47,
    42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63,
    58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75,
    66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84,
    74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
}


def run_ultralytics_val(model_path, data_yaml):
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    precision, recall, map50, map5095 = results.box.mean_results()
    print("Ultralytics Evaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map5095:.4f}")


def run_pycoco_eval(model_path, image_dir, annotation_file, output_json):
    model = YOLO(model_path)
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    predictions = []

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(image_dir, img_info['file_name'])
        results = model(image_path)
        result = results[0]

        for box, cls, conf in zip(
                result.boxes.xywhn, result.boxes.cls, result.boxes.conf):
            x_center, y_center, width, height = box.tolist()
            x = x_center - width / 2
            y = y_center - height / 2
            prediction = {
                "image_id": img_id,
                "category_id": COCO_MAP[int(cls.item())],
                "bbox": [x * img_info['width'], y * img_info['height'],
                         width * img_info['width'], height * img_info['height']],
                "score": float(conf.item())
            }
            predictions.append(prediction)

    with open(output_json, 'w') as f:
        json.dump(predictions, f)

    coco_dt = coco.loadRes(output_json)
    coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11x COCO Evaluation")
    parser.add_argument('--option', type=int, choices=[1, 2], required=True,
                        help="Evaluation option: 1 for Ultralytics, 2 for pycocotools")
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11x.pt',
        help="Path to YOLOv11x model")
    parser.add_argument(
        '--data',
        type=str,
        default='coco.yaml',
        help="Path to coco.yaml file")
    parser.add_argument(
        '--images',
        type=str,
        default='val2017',
        help="Path to COCO val2017 images")
    parser.add_argument('--annotations', type=str, default='annotations/instances_val2017.json',
                        help="Path to COCO annotations JSON file")
    parser.add_argument('--output_json', type=str, default='predictions.json',
                        help="Path to save predictions in COCO format")

    args = parser.parse_args()

    if args.option == 1:
        run_ultralytics_val(args.model, args.data)
    elif args.option == 2:
        run_pycoco_eval(
            args.model,
            args.images,
            args.annotations,
            args.output_json)
