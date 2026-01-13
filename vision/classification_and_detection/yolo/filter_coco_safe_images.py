import json
import os
import shutil

# Paths (adjust these)
ann_file = "/mnt/data/yolo/annotations/instances_val2017.json"
images_dir = "/mnt/data/yolo/coco/val2017"
output_dir = "/mnt/data/yolo/coco/val2017_safe"

os.makedirs(output_dir, exist_ok=True)

# Load COCO annotations
with open(ann_file, "r") as f:
    coco = json.load(f)

# Map license id to name/url
licenses = {l["id"]: l for l in coco["licenses"]}

# Define safe licenses: CC-BY and CC-BY-SA
# safe_keywords = ["creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/"]
# safe_keywords = ["creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/", "creativecommons.org/licenses/by-nc" , "creativecommons.org/licenses/by-nc-nd" , "creativecommons.org/licenses/by-nd", "creativecommons.org/licenses/by-nc-sa" ]
safe_keywords = [
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
    "creativecommons.org/licenses/by-nd",
    "flickr.com/commons/usage",
    "www.usa.gov"]


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
    dst = os.path.join(output_dir, img["file_name"])
    if os.path.exists(src):
        shutil.copy2(src, dst)

print(f"Copied {len(safe_images)} images to {output_dir}")
