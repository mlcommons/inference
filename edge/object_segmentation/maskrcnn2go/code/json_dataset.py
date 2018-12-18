from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import os

import matplotlib
# Use a non-interactive backend
matplotlib.use('Agg')
from pycocotools.coco import COCO


logger = logging.getLogger(__name__)


IM_DIR = "image_directory"
ANN_FN = "annotation_file"
IM_PREFIX = "image_prefix"


class JsonDataset(object):
    def __init__(self, name, ds_im_dir=None, ds_ann=None):
        full_datasets = {}
        if ds_im_dir is not None and ds_ann is not None:
            full_datasets[name] = {
                IM_DIR: ds_im_dir,
                ANN_FN: ds_ann,
            }
        assert name in full_datasets.keys(), "Unknown dataset name {}".format(name)
        logger.debug("Creating: {}".format(name))

        dataset = full_datasets[name]
        logger.info("Loading dataset {}:\n{}".format(name, dataset))

        self.name = name
        self.image_directory = dataset[IM_DIR]
        self.image_prefix = dataset.get(IM_PREFIX, "")

        # general dataset
        self.COCO = COCO(dataset[ANN_FN])
        logger.info(
            "Dataset={}, Number of images={}".format(name, len(self.COCO.getImgIds()))
        )

        category_ids = self.COCO.getCatIds()
        categories = [c["name"] for c in self.COCO.loadCats(category_ids)]
        self.category_ids = category_ids
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ["__background__"] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def get_roidb(self):
        coco = self.COCO
        image_ids = self.COCO.getImgIds()
        image_ids.sort()

        roidb = copy.deepcopy(coco.loadImgs(image_ids))
        for entry in roidb:
            self._prep_roidb_entry(entry)

        return roidb

    def _prep_roidb_entry(self, entry):
        # Reference back to the parent dataset
        entry["dataset"] = self
        # Make file_name an abs path
        entry["image"] = os.path.join(
            self.image_directory, self.image_prefix + entry["file_name"]
        )

        # Remove unwanted fields if they exist
        for k in ["date_captured", "license", "file_name"]:
            if k in entry:
                del entry[k]
