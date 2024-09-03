"""Tool to generate the final results speadsheet from the submission checker csv output.

The resulting excel files can be imported into google sheets.
"""
import argparse
import os
import sys
import re
import numpy as np
import pandas as pd
import json

def get_args():
  """Parse commandline."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True, help='results csv from checker')
  parser.add_argument('--version', default='4.0', help='mlperf version')
  parser.add_argument('--repository', default='submissions_inference_4.0', help='mlperf repository')
  args = parser.parse_args()
  return args


def main():
  args = get_args()

  df = pd.read_csv(args.input).fillna('')

  if df.empty:
    return

  # rename some fields
  df.rename(
      columns={
          'Organization': 'Submitter',
          'Division': 'Category',
          'SystemType': 'Suite',
          'SystemName': 'System',
          'number_of_nodes': 'Nodes',
          'host_processor_model_name': 'Processor',
          'accelerator_model_name': 'Accelerator',
          'accelerators_per_node': 'a#',
          'notes': 'Notes',
          'framework': 'Software',
      },
      inplace=True)
  df.rename(columns={'Model': 'UsedModel'}, inplace=True)
  df.rename(columns={'MlperfModel': 'Model'}, inplace=True)

  # fix issues with raw data
  df['host_processor_core_count'] = df['host_processor_core_count'].apply(
      lambda x: 2 if x == '2 (big); 4 (LITTLE)' else x)
  df['Availability'] = df['Availability'].apply(lambda x: 'available'
                                                if x == 'on-premise' else x)

  # cleanup counts
  df['Accelerator'] = df['Accelerator'].apply(lambda x: x if x != '-' else '')
  df['a#'] = df['a#'].apply(lambda x: int(x) if str(x).isnumeric() else x)
  df['a#'] = df['a#'].apply(lambda x: x if x != 0 else '')
  df['p#'] = df.apply(lambda x: int(x['host_processors_per_node']), axis=1)

  # details url
  base_url = f'https://github.com/mlcommons/{args.repository}/tree/main'
  df['Details'] = df.apply(
      lambda x: '=HYPERLINK("{}","details")'.format('/'.join(
          [base_url, x['Category'], x['Submitter'], 'results', x['Platform']])),
      axis=1)

  # code url
  df['Code'] = df.apply(
      lambda x: '=HYPERLINK("{}","code")'.format('/'.join(
          [base_url, x['Category'], x['Submitter'], 'code'])),
      axis=1)

  output = args.input[:-4]
  writer = pd.ExcelWriter(output + '.xlsx', engine='xlsxwriter')
  outjsondata = []

  indices = {}
  indices['closed'] = [
      'ID',
      'Unique ID (e.g. for Audit)',
      'ColorKey',
      'Submitter',
      'Availability',
      'System',
      'Nodes',
      'Processor',
      'p#',
      'Accelerator',
      'a#',
      'Software',
      'Notes',
  ]
  indices['open'] = indices['closed'].copy()
  indices['closed'].append('Details')
  indices['closed'].append('Code')
  indices['network'] = indices['closed'].copy()
  indices['open'].append('UsedModel')
  indices['open'].append('Accuracy')
  indices['open'].append('Details')
  indices['open'].append('Code')
  columns = [
      'Model',
      'Scenario',
      'Units',
  ]
  columns_order = [['Result'],
                   [
                       'resnet', 'retinanet', '3d-unet-99', '3d-unet-99.9',
                       'rnnt', 'bert-99', 'bert-99.9', 'dlrm-v2-99', 'dlrm-v2-99.9',
                       'gptj-99', 'gptj-99.9', 'stable-diffusion-xl', 'llama2-70b-99', 'llama2-70b-99.9',
                       'mixtral-8x7b',
                   ], ['SingleStream', 'MultiStream', 'Server', 'Offline'],
                   [
                       'Latency (ms)',
                       'Samples/s',
                       'Queries/s',
                       'Tokens/s',
                       'millijoules',
                       'Watts',
                   ]]

  filter_scenarios = {
      'datacenter': {
          'resnet': ['Server', 'Offline'],
          'retinanet': ['Server', 'Offline'],
          'rnnt': ['Server', 'Offline'],
          'bert-99': ['Server', 'Offline'],
          'bert-99.9': ['Server', 'Offline'],
          'dlrm-v2-99': ['Server', 'Offline'],
          'dlrm-v2-99.9': ['Server', 'Offline'],
          '3d-unet-99': ['Offline'],
          '3d-unet-99.9': ['Offline'],
          'gptj-99': ['Server', 'Offline'],
          'gptj-99.9': ['Server', 'Offline'],
          'stable-diffusion-xl': ['Server', 'Offline'],
          'llama2-70b-99': ['Server', 'Offline'],
          'llama2-70b-99.9': ['Server', 'Offline'],
          'mixtral-8x7b': ['Server', 'Offline'],
      },
      'edge': {
          'resnet': ['SingleStream', 'MultiStream', 'Offline'],
          'retinanet': ['SingleStream', 'MultiStream', 'Offline'],
          'rnnt': ['SingleStream', 'Offline'],
          'bert-99': ['SingleStream', 'Offline'],
          'bert-99.9': [],
          'dlrm-v2-99': [],
          'dlrm-v2-99.9': [],
          '3d-unet-99': ['SingleStream', 'Offline'],
          '3d-unet-99.9': ['SingleStream', 'Offline'],
          'gptj-99': ['SingleStream', 'Offline'],
          'gptj-99.9': ['SingleStream', 'Offline'],
          'stable-diffusion-xl': ['SingleStream', 'Offline'],
      }
  }

  def MakeWorksheet(df, index, filter_dict, sheet_name, outjsondata=[]):
    for key, value in filter_dict.items():
      if type(key) == tuple:
        key = list(key)
      df = df[value(df[key])]
    if df.size == 0:
      return
    json_df = df.to_json(orient='records')
    outjsondata += json.loads(json_df)

    df = df.pivot_table(index=index, columns=columns, values=['Result'])
    df = df.fillna('')
    for i, order in enumerate(columns_order):
      df = df.reindex(columns=order, level=i)
    df.to_excel(writer, sheet_name=sheet_name)

  def Equal(x):
    return lambda y: y == x

  def NotEqual(x):
    return lambda y: y != x

  def Contain(x):
    return lambda y: y.str.find(x) != -1

  def And(x, y):
    return lambda z: x(z) & y(z)

  def Apply(f, *args):
    return lambda x: f(x, *args)

  def FilterScenario(x, suite):
    return x.apply(
        lambda y: y['Scenario'] in filter_scenarios[suite][y['Model']], axis=1)

  def MakeUniqueID(x):
    key_list = ['Suite', 'Category', 'Submitter', 'Platform']
    if x['Category'] == 'open':
      key_list.append('UsedModel')
    return '/'.join(x[key_list])

  df['Unique ID (e.g. for Audit)'] = df.apply(MakeUniqueID, axis=1)
  df['ColorKey'] = df.apply(
      lambda x: ''.join(x[['Availability', 'Submitter']]), axis=1)
  df.sort_values(
      by=[
          'Category', 'Availability', 'Submitter', 'Unique ID (e.g. for Audit)'
      ],
      inplace=True)
  id_dict = {
      key: 1 + value
      for (value,
           key) in enumerate(pd.unique(df['Unique ID (e.g. for Audit)']))
  }
  df['ID'] = df.apply(
      lambda x: '{}-{:04}'.format(args.version, id_dict[x['Unique ID (e.g. for Audit)']]),
      axis=1)

  for category in ['closed', 'open', 'network']:
    for suite in ['datacenter', 'edge']:
      MakeWorksheet(
          df, indices[category], {
              'Category':
                  Equal(category),
              'Suite':
                  Contain(suite),
              'Units':
                  And(
                      And(NotEqual('Watts'), NotEqual('millijoules')),
                      NotEqual('millijoules/Stream')),
              ('Scenario', 'Model'):
                  Apply(FilterScenario, suite)
          }, suite + ' - ' + category, outjsondata)

      MakeWorksheet(
          df, indices[category], {
              'Category': Equal(category),
              'Suite': Contain(suite),
              'has_power': Equal(True),
              ('Scenario', 'Model'): Apply(FilterScenario, suite)
          }, suite + ' - ' + category + ' - power', outjsondata)

  def reformatlink(data, key):
    details = data[key]
    details = details[details.find("(")+2:details.find(",")-1]
    return details

  for i,result in enumerate(outjsondata):
    result['Details'] = reformatlink(result, "Details")
    result['Code'] = reformatlink(result, "Code")
    result_id = result.pop('ID')
    outjsondata[i] = {'ID': result_id, **result}

  outjsondata.sort(key=lambda x:x["Units"])
  outjsondata.sort(key=lambda x:x["Scenario"])
  outjsondata.sort(key=lambda x:x["UsedModel"])
  outjsondata.sort(key=lambda x:x["ID"])

  #remove duplicate perf results
  keystomatch = ['ID', 'UsedModel', 'Scenario', 'Units']
  i = 0
  n = len(outjsondata)
  while i < n:
    result = outjsondata[i]
    while i < n - 1 and all(result[key] == outjsondata[i+1][key] for key in keystomatch):
      del(outjsondata[i+1])
      n -= 1
    i += 1

  #merge perf and power results
  keystomatch.pop()

  for i in range(len(outjsondata)):
    result = outjsondata[i]
    if not result:
      continue
    if i < len(outjsondata) - 1:
      if all(result[key] == outjsondata[i+1][key] for key in keystomatch):
        #print(result)
        #print(outjsondata[i+1])
        if "Watts" in result['Units'] or "joules" in result['Units']:
          result['Performance_Result'] = outjsondata[i+1]['Result']
          result['Performance_Units'] = outjsondata[i+1]['Units']
          result['Power_Result'] = result['Result']
          result['Power_Units'] = result['Units']
        else:
          result['Power_Result'] = outjsondata[i+1]['Result']
          result['Power_Units'] = outjsondata[i+1]['Units']
          result['Performance_Result'] = result['Result']
          result['Performance_Units'] = result['Units']
        outjsondata[i+1] = {}
        del(result['Result'])
        del(result['Units'])

  for i,result in enumerate(outjsondata):
    if result.get('Result'):
      result['Performance_Result'] = result['Result']
      result['Performance_Units'] = result['Units']
      del(result['Result'])
      del(result['Units'])

  outjsondata = [ i for i in outjsondata if i != {}]
  with open(f"{output}_results.json", "w") as f:
    f.write(json.dumps(outjsondata, indent=2))
  score_format = writer.book.add_format({'num_format': '#,##0.00'})
  bg_format = writer.book.add_format({'bg_color': '#efefef'})
  for ws in writer.book.worksheets():
    ws.set_column(1, 1, None, None, {'hidden': 1})
    ws.set_column(2, 2, None, None, {'hidden': 1})
    ws.set_column(len(indices['closed']), 100, None, score_format)
    ws.conditional_format(
        2 + len(columns), 0, 200, 100, {
            'type':
                'formula',
            'criteria':
                '=mod(countunique($c$' + str(len(columns) + 3) + ':$c' +
                str(len(columns) + 3) + '), 2) = 0',
            'format':
                bg_format,
        })

  writer.close()

  return 0


if __name__ == '__main__':
  sys.exit(main())
