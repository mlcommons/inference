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
  [
    'gpt3175b8',
    'gpt3175b16',
    'gpt3175b32',
    'gpt3175b64',
    'gpt3175b128',
    'gpt3175b256',
    'lmcloudspmd2b4test',
    'lmcloudspmd175b8test',
    'lmcloudspmd175b16test',
    'lmcloudspmd175b32test',
    'lmcloudspmd175b64test',
    'lmcloudspmd175b128test',
    'lmcloudspmd175b256test',
  ],
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
_WAIT = flags.DEFINE_integer(
    'wait',
    1200,
    'Model loading wait time.',
)
_UNPUBLISH = flags.DEFINE_boolean(
    'unpublish',
    False,
    'Unpublish Model.',
)


def main(argv):

  sax_cell = _SAX_CELL.value
  model_name = _MODEL_NAME.value
  model_path = f"{sax_cell}/{model_name}"
  num_replicas = _REPLICA.value
  checkpoint_path = _CHECKPOINT_PATH.value

  if model_name == 'gpt3175b8':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP8"
    assert (checkpoint_path != 'None')
  if model_name == 'gpt3175b16':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP16"
    assert (checkpoint_path != 'None')
  elif model_name == 'gpt3175b32':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP32"
    assert (checkpoint_path != 'None')
  elif model_name == 'gpt3175b64':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP64"
    assert (checkpoint_path != 'None')
  elif model_name == 'gpt3175b128':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP128"
    assert (checkpoint_path != 'None')
  elif model_name == 'gpt3175b256':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP256"
    assert (checkpoint_path != 'None')

  elif model_name == 'lmcloudspmd2b4test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B4Test"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b8test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B8Test"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b16test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B16Test"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b32test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B32Test"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b64test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B64Test"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b128test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B128Test"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b256test':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B256Test"
    assert (checkpoint_path == 'None')

  logging.info("Publishing models to {} ... ".format(model_path))

  try:
    sax.Publish(
      model_path,
      model_config_path,
      checkpoint_path,
      num_replicas,
    )
  except status.StatusNotOk as e:
    logging.info(e)
  else:
    logging.info("Published model: {} with config {} from checkpoint {} to {} replicas".format(model_path, model_config_path, checkpoint_path, num_replicas))

  time.sleep(_WAIT.value)

  logging.info("Initiate language model for inferencing ... ")
  model = sax.Model(model_path)
  language_model = model.LM()
  option = sax.ModelOptions()
  option.SetTimeout(_WAIT.value)

  test_text = "1112,444,555"
  try:
    logging.info("Generating testing inferencing result ... ")
    inference_result = language_model.Generate(test_text, option)
  except status.StatusNotOk as e:
    logging.info(e)
  else:
    logging.info("Generated inference result: {}".format(inference_result))
    assert type(inference_result[0][0]) == str
    assert type(inference_result[0][1]) == float

  if _UNPUBLISH.value:
    try:
      sax.Unpublish(model_path)
    except status.StatusNotOk as e:
      logging.info(e)
    else:
      logging.info("Unpublished model {}".format(model_path))

if __name__ == '__main__':
  app.run(main)
