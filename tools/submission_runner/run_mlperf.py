# !/usr/bin/env python
"""
 Copyright (c) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import argparse
from shutil import copyfile
import json
import shlex
import subprocess


def getInputParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str, help='Path to config file.')
    return parser


def run_process(args, dir, env, redirect_output):
    output = ""
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                         bufsize=1, universal_newlines=True, cwd=dir, env=env) 
    while True:
      data = process.stdout.readline()
      if data == '' and process.poll() is not None :
        break
      if data:
        output += data
        if redirect_output:
            print(data, end="")
    rc = process.poll()
    return rc, output


def expand_paths(cmd, dct):
    for val in dct:
        cmd = cmd.replace(val, dct[val])
    return cmd


def str2bool(v):
  return v.lower() in [ 'yes', 'true', 't', '1' ]


def create_env(config):
    run_env = os.environ.copy()
    if 'env' in config['config']:
        export = config['config']['env']
        for item in export:
            key = list(item.keys())[0] 
            run_env[key] = item[key]
    return run_env


def create_folder(path, name):
    if not os.path.isdir(path):
        print('Creating {} folder: {}'.format(name, path))
        os.makedirs(path)
    else:
        print('{} folder exists: {}'.format(name, path))


def main():
    args = getInputParameters().parse_args()

    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    run_env = create_env(config_dict)

    logs_path = config_dict['config']['logs_dir']

    division = os.path.join(logs_path, config_dict['division'])
    org = os.path.join(division, config_dict['submitter'])

    code = os.path.join(org, 'code')
    results = os.path.join(org, 'results')
    compliance = os.path.join(org, 'compliance')
    measurements = os.path.join(org, 'measurements')

    redirect_output = str2bool(config_dict['config']['redirect_output'])
    
    net = config_dict['config']['topology']
    net_name = net

    if 'accuracy' in config_dict['config']:
        net = '{}-{}'.format(net, config_dict['config']['accuracy'])

    framework = config_dict['framework']
    dtype = config_dict['config']['dtype'].upper()
    scenario = config_dict['config']['scenario']
    mlperf_path = config_dict['config']['mlperf_path']

    system = '{}-{}-{}'.format(config_dict['config']['system'], framework, dtype)

    mlperf_conf = os.path.join(mlperf_path, 'mlperf.conf')

    aliases = {}

    if 'aliases' in config_dict['config']:
        aliases = config_dict['config']['aliases']
    
    aliases['$LOGS_DIR'] = logs_path

    code_path = os.path.join(code, framework, net)

    create_folder(org, 'org')
    create_folder(code_path, 'code')

    compliance_tests = [ 'TEST01', 'TEST04-A', 'TEST04-B', 'TEST05' ]
    compliance_config_path = os.path.join(mlperf_path, 'compliance', 'nvidia')
    compliance_paths = {}

    for test in compliance_tests:
        compliance_path = os.path.join(compliance, system, net, scenario, test)
        compliance_tmp_path = os.path.join(compliance, system, net, scenario, 'tmp', test)
        compliance_paths[test] = compliance_path
        aliases['${}_DIR'.format(test)] = compliance_path
        aliases['${}_TMP_DIR'.format(test)] = compliance_tmp_path

        create_folder(compliance_path, test)
        create_folder(compliance_tmp_path, test)

        audit_dst = os.path.join(compliance_tmp_path, 'audit.config')

        if test == 'TEST01':
            audit_conf = os.path.join(compliance_config_path, test, net_name, 'audit.config')
        else:
            audit_conf = os.path.join(compliance_config_path, test, 'audit.config')

        if not os.path.isfile(audit_dst):
            print('Copying audit.config file: {}'.format(audit_dst))
            copyfile(audit_conf, audit_dst)
        else:
            print('audit.config file exists: {}'.format(audit_dst))

    aliases['$COMPLIANCE_DIR'] = os.path.join(compliance, system, net, scenario)

    measurements_path = os.path.join(measurements, system, net, scenario)

    create_folder(measurements_path, 'measurements')

    mlperf_dst_conf = os.path.join(measurements_path, 'mlperf.conf')
    if not os.path.isfile(mlperf_dst_conf):
        print('Copying mlperf.conf file: {}'.format(mlperf_dst_conf))
        copyfile(mlperf_conf, mlperf_dst_conf)
    else:
        print('mlperf.conf file exists: {}'.format(mlperf_dst_conf))

    results_path = os.path.join(results, system, net, scenario)

    create_folder(results_path, 'results')

    aliases['$RESULTS_DIR'] = results_path

    results_type = [ 'performance/run_1', 'accuracy' ]
    for typ in results_type:
        result_path = os.path.join(results_path, typ)
        create_folder(result_path, typ)
        aliases['${}_DIR'.format(typ.upper())] = result_path

    systems_path = os.path.join(org, 'systems')
    create_folder(systems_path, 'systems')

    if 'exec' in config_dict['config']:
        for exec in config_dict['config']['exec']:
            print('Running {}'.format(exec['name']))
            if 'dir' in exec:
                dir = expand_paths(exec['dir'], aliases)
            else:
                dir = None
            cmd = expand_paths(exec['cmd'], aliases)
            if 'skip' in exec:
                if str2bool(exec['skip']):
                    print('Skipping...')
                    continue
            args = shlex.split(cmd)
            code, output = run_process(args, dir, run_env, redirect_output)
            if 'save_output' in exec:
                path = os.path.join(dir, exec['save_output'])
                f = open(path, 'w+')
                f.write(output)
                f.close()
            
            if not code == 0:
                print('Error returned, exiting...')
                break


if __name__ == '__main__':
    main()