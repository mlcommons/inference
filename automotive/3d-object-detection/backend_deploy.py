from typing import Optional, List, Union
import os
import torch
import logging
import backend

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend-pytorch")


class BackendDeploy(backend.Backend):
    def __init__(
        self,
        # TODO: pass model and inference params
    ):
        super(BackendDeploy, self).__init__()
        # TODO: init model class, model and inference params

    def version(self):
        return torch.__version__

    def name(self):
        return "python-SUT"


    def load(self):
        # TODO: Load model
        return self

    

    def predict(self, inputs):
        # TODO: implement predict
        Boxes, Classes, Scores = [], [], []
        return Boxes, Classes, Scores # Change to desired output

