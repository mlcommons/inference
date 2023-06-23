from absl import app
from absl import flags
from absl import logging

import status
import sax
import time

_SAX_CELL = flags.DEFINE_string(
    'sax_cell',
    '/sax/test',
    'SAX cell of the admin server.',
)
_MODEL_NAME = flags.DEFINE_enum(
  'model_name',
  'lmcloudspmd2b4test',
  ['gpt3175b32','gpt3175b64','gpt3175b128','lmcloudspmd2b4test','lmcloudspmd175b32test','lmcloudspmd175b64test','lmcloudspmd175b128test'],
  'model name to publish.'
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    'None',
    'Checkpoint path, this is required if the model is not in test mode',
)
_REPLICA = flags.DEFINE_integer(
    'replica',
    1,
    'Replica',
)


def main():

  sax_cell = _SAX_CELL.value
  model_name = _MODEL_NAME.value
  model_path = f"{sax_cell}/{model_name}"
  num_replicas = _REPLICA.value
  checkpoint_path = _CHECKPOINT_PATH.value

  if model_name == 'gpt3175b32':
    model_config_path = "saxml.server.pax.lm.params.c4.C4SpmdGpt3AdamOrgHP32"
    assert checkpoint_path is not None
  elif model_name == 'gpt3175b64':
    model_config_path = "saxml.server.pax.lm.params.c4.C4SpmdGpt3AdamOrgHP"
    assert checkpoint_path is not None
  elif model_name == 'gpt3175b128':
    model_config_path = "saxml.server.pax.lm.params.c4.C4SpmdGpt3AdamOrgHP128"
    assert checkpoint_path is not None

  elif model_name == 'lmcloudspmd2b4test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B4Test"
    assert checkpoint_path is None
  elif model_name == 'lmcloudspmd175b32test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B32Test"
    assert checkpoint_path is None
  elif model_name == 'lmcloudspmd175b64test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B64Test"
    assert checkpoint_path is None
  elif model_name == 'lmcloudspmd175b128test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B128Test"
    assert checkpoint_path is None

  logging.info("Publishing models to {} ... ".format(model_path))

  try:
    sax.Publish(
      model_path,
      model_config_path,
      checkpoint_path,
      num_replica,
    )
  except status.StatusNotOk as e:
    logging.info(e)
  else:
    logging.info("Published models in {}: {}".format(sax_cell, published_models_result))

if __name__ == '__main__':
  app.run(main)
