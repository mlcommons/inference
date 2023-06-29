
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

import status
import sax
import seqio
import numpy as np

spm_path = "gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model"
vocabulary = seqio.SentencePieceVocabulary(spm_path)

sax_cell = '/sax/test'
model_path = '/sax/test/gpt3175b64'


model = sax.Model(model_path)
language_model = model.LM()
option = sax.ModelOptions()
option.SetTimeout(1200)

text="""summarize: The owner of nine grocery stores in central Ohio is accused of leading a large-scale drug ring trafficking heroin, cocaine and marijuana from Mexico to Chicago and Ohio for years. Liborio Alcauter was one of 29 people indicted in a large-scale drugs sting last September. Investigators are also probing claims he hired a Mexican hitman to kill the judge and investigators after his arrest, the Columbus Dispatch reports. 'Ring leader': Liborio Alcauter, a grocery store owner, is accused of leading a gang that trafficked heroin and cocaine from Mexico into Chicago then onto Ohio for years . Alcauter was released from custody last year after a judge deemed him not to be a flight risk. His attorney dismissed the allegations made in a search warrant, telling the Dispatch 'They can say anything they want in a search warrant.' According to the documents read by the Dispatch, one of his co-defendants, Lorena Sevilla, was overheard in her cell describing Alcauter's plot to kill the investigators. 'Sevilla said out loud that Liborio Alcauter wanted to handle things like they did in Mexico and said "judge dead" and ran her finger across her throat in a cutting motion,' an affidavit allegedly says. The alleged hitman is believed to be in custody on immigration charges. Alcauter is not working in any of his groceries as the investigation continues. He is also accused of hiring a Mexican hitman to kill the judge and investigators after he was arrested ."""

max_length = 2048

token_ids = vocabulary.tokenizer.tokenize(text)
token_ids_str = ','.join([str(i) for i in list(token_ids)])
try:
    inference_result = language_model.Generate(token_ids_str, option)
except status.StatusNotOk as e:
    print(e)

pred_token_ids_str = inference_result[0][0]
pred_token_ids = pred_output = np.fromstring(pred_token_ids_str, dtype=int, sep=',')
pred_text = vocabulary.tokenizer.detokenize(pred_token_ids.tolist())


padded_token_ids = ['0']*(max_length-len(token_ids)) + token_ids
padded_token_ids_str = ','.join([str(i) for i in list(padded_token_ids)])
try:
    padded_inference_result = language_model.Generate(padded_token_ids_str, option)
except status.StatusNotOk as e:
    print(e)


padded_pred_token_ids_str = padded_inference_result[0][0]
padded_pred_token_ids = pred_output = np.fromstring(padded_pred_token_ids_str, dtype=int, sep=',')
padded_pred_text = vocabulary.tokenizer.detokenize(padded_pred_token_ids.tolist())

