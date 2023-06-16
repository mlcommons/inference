from absl import app
from absl import flags
from absl import logging

import status
import sax
import time
import logging

_SAX_CELL = flags.DEFINE_string(
    'sax_cell',
    '/sax/test',
    'SAX cell of the admin server.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    'lm2b',
    'Model name.',
)
_MODEL_CONFIG_PATH = flags.DEFINE_string(
    'model_config_path',
    'saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B',
    'Model config path.',
)
_REPLICA = flags.DEFINE_integer(
    'replica',
    1,
    'Replica',
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    'None',
    'Checkpoint path, this is required if the model is not in test mode',
)
_WAIT = flags.DEFINE_integer(
    'wait',
    60,
    'Model loading wait time.',
)
_TOKENIZED = flags.DEFINE_boolean(
    'tokenized',
    True,
    'Requires input as tokenized or not',
)

def main(argv):

  cell = _SAX_CELL.value
  model_name = _MODEL_NAME.value
  model_config_path = _MODEL_CONFIG_PATH.value
  num_replicas = _REPLICA.value
  checkpoint_path = _CHECKPOINT_PATH.value
  tokenized = _TOKENIZED.value

  logging.info(f"------------------------------------ test {model_name} start ------------------------------------")

  model_path = f"{cell}/{model_name}"

  try:
    published_models_result = sax.ListAll(cell)
  except status.StatusNotOk as e:
    logging.info(e)
  else:
    logging.info("Published models in {}: {}".format(cell, published_models_result))

  try:
    sax.Publish(
      model_path,
      model_config_path,
      checkpoint_path,
      num_replicas
    )
  except status.StatusNotOk as e:
    logging.info(e)
  else:
    logging.info("Published model: {} with config {} from checkpoint {} to {} replicas".format(model_path, model_config_path, checkpoint_path, num_replicas))

  time.sleep(_WAIT.value)


  try:
    published_models_result = sax.ListAll(cell)
  except status.StatusNotOk as e:
    logging.info(e)
  else:
    logging.info("Published models in {}: {}".format(cell, published_models_result))


  try:
    published_model_result = sax.List(f"{cell}/{model_name}")
  except status.StatusNotOk as e:
    logging.info(e)
  else:
    logging.info("Published model {}: {}".format(f"{cell}/{model_name}", published_model_result))

  if tokenized:
    test_text = "1112,444,555"
  else:
    test_text = "Q: Who is Harry Porter's mother? A: "

  model = sax.Model(model_path)
  language_model = model.LM()
  option = sax.ModelOptions()
  option.SetTimeout(1200)

  try:
    inference_result = language_model.Generate(test_text, option)
  except status.StatusNotOk as e:
    logging.info(e)
  else:
    logging.info("Generate inference result: {}".format(inference_result))
    assert type(inference_result[0][0]) == str
    assert type(inference_result[0][1]) == float


  try:
    sax.Unpublish(model_path)
  except status.StatusNotOk as e:
    logging.info(e)
  else:
    logging.info("Unpublished model {}".format(model_path))

  logging.info(f"------------------------------------ test {model_name} end ------------------------------------")


if __name__ == '__main__':
  app.run(main)
