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
_MODEL_NAME = flags.DEFINE_string(
  'model_name',
  'lmcloudspmd2b4testtokenized',
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

  if model_name == 'gpt3175b8tokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP8Tokenized"
    assert (checkpoint_path != 'None')
  if model_name == 'gpt3175b16tokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP16Tokenized"
    assert (checkpoint_path != 'None')
  elif model_name == 'gpt3175b64tokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP64Tokenized"
    assert (checkpoint_path != 'None')
  elif model_name == 'gpt3175b64tokenizedgreedy':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP64TokenizedGreedy"
    assert (checkpoint_path != 'None')
  elif model_name == 'gpt3175b128tokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP128Tokenized"
    assert (checkpoint_path != 'None')
  elif model_name == 'gpt3175b256tokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP256Tokenized"
    assert (checkpoint_path != 'None')

  elif model_name == 'lmcloudspmd2b4testtokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B4TestTokenized"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b8testtokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B8TestTokenized"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b16testtokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B16TestTokenized"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b32testtokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B32TestTokenized"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b64testtokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B64TestTokenized"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b128testtokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B128TestTokenized"
    assert (checkpoint_path == 'None')
  elif model_name == 'lmcloudspmd175b256testtokenized':
    model_config_path = "saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B256TestTokenized"
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

  test_text = '1112,444,555'
  print('test_text: ', test_text)
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
