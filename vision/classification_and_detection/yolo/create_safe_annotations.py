import json
import os
import shutil

# Paths
ann_file = "/mnt/data/yolo/coco/annotations/instances_val2017.json"
images_dir = "/mnt/data/yolo/coco/val2017"
output_images_dir = "/mnt/data/yolo/coco/val2017_safe"
output_ann_file = "/mnt/data/yolo/coco/annotations/instances_val2017_safe.json"

os.makedirs(output_images_dir, exist_ok=True)

# Load COCO annotations
with open(ann_file, "r") as f:
    coco = json.load(f)

licenses = {l["id"]: l for l in coco["licenses"]}

# Define safe license keywords
safe_keywords = [
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
    "creativecommons.org/licenses/by-nd",
    "flickr.com/commons/usage",
    "www.usa.gov"
]

# Filter safe images
safe_images = []
for img in coco["images"]:
    lic = licenses.get(img["license"], {})
    url = lic.get("url", "").lower()
    if any(k in url for k in safe_keywords):
        safe_images.append(img)

print(f"Total images: {len(coco['images'])}")
print(f"Safe images: {len(safe_images)}")

# Copy safe images
for img in safe_images:
    src = os.path.join(images_dir, img["file_name"])
    dst = os.path.join(output_images_dir, img["file_name"])
    if os.path.exists(src):
        shutil.copy2(src, dst)

print(f"Copied {len(safe_images)} images to {output_images_dir}")

# Filter annotations for safe images
safe_image_ids = {img["id"] for img in safe_images}
safe_annotations = [ann for ann in coco["annotations"]
                    if ann["image_id"] in safe_image_ids]

# Build new COCO annotation structure
safe_coco = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "images": safe_images,
    "annotations": safe_annotations,
    "categories": coco.get("categories", [])
}

# Save new annotation file
with open(output_ann_file, "w") as f:
    json.dump(safe_coco, f)

print(f"Saved filtered annotations to {output_ann_file}")
print(f"Total annotations: {len(safe_annotations)}")
