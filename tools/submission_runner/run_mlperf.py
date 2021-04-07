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
import sys
import argparse
from shutil import copyfile
import json
import shlex, subprocess

def getInputParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlperf', '-mlperf', required=True, type=str, help='Mlperf inference folder path.')
    parser.add_argument('--config', '-c', required=True, type=str, help='Path to config file.')
    return parser


def run_process(args, dir):
    output = ""
    with subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, cwd=dir) as p:
        for line in p.stdout:
            print(line, end='')
            output += line
    return output


def expand_paths(cmd, dct):
    for val in dct:
        cmd = cmd.replace(val, dct[val])
    return cmd


def main():
    args = getInputParameters().parse_args()

    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    if 'export' in config_dict['config']:
        val = config_dict['config']['export'].split('=')
        print(val[0])
        print(val[1])
        if len(val) == 2:
            os.environ[val[0]] = val[1]

    logs_path = config_dict['config']['logs_dir']

    division = os.path.join(logs_path, config_dict['division'])
    org = os.path.join(division, config_dict['submitter'])

    code = os.path.join(org, 'code')
    results = os.path.join(org, 'results')
    compliance = os.path.join(org, 'compliance')
    measurements = os.path.join(org, 'measurements')
    
    net = config_dict['config']['topology']
    net_name = net

    if 'accuracy' in config_dict['config']:
        net = '{}-{}'.format(net, config_dict['config']['accuracy'])

    framework = config_dict['framework']
    dtype = config_dict['config']['dtype'].upper()
    scenario = config_dict['config']['scenario']
    mlperf = args.mlperf

    system = '{}-{}-{}'.format(config_dict['config']['system'], framework, dtype)

    mlperf_conf = os.path.join(args.mlperf, 'mlperf.conf')

    aliases = {}

    if 'aliases' in config_dict['config']:
        aliases = config_dict['config']['aliases']

    if not os.path.isdir(org):
        os.makedirs(org)
        print('Creating org folder: {}'.format(org))

    code_path = os.path.join(code, framework, net)
    if not os.path.isdir(code_path):
        print('Creating code folder: {}'.format(code_path))
        os.makedirs(code_path)
    else:
        print('Code folder exists: {}'.format(code_path))

    compliance_tests = [ 'TEST01', 'TEST04-A', 'TEST04-B', 'TEST05' ]
    compliance_config_path = os.path.join(mlperf, 'compliance', 'nvidia')
    compliance_paths = {}

    for test in compliance_tests:
        compliance_path = os.path.join(compliance, system, net, test)
        compliance_paths[test] = compliance_path
        aliases['${}_DIR'.format(test)] = compliance_path
        if not os.path.isdir(compliance_path):
            print('Creating {} folder: {}'.format(test, compliance_path))
            os.makedirs(compliance_path)
        else:
            print('{} folder exists: {}'.format(test, compliance_path))
        
        audit_dst = os.path.join(compliance_path, 'audit.config')

        if test == 'TEST01':
            audit_conf = os.path.join(compliance_config_path, test, net_name, 'audit.config')
        else:
            audit_conf = os.path.join(compliance_config_path, test, 'audit.config')

        if not os.path.isfile(audit_dst):
            print('Copying audit.config file: {}'.format(audit_dst))
            copyfile(audit_conf, audit_dst)
        else:
            print('audit.config file exists: {}'.format(audit_dst))

    measurements_path = os.path.join(measurements, system, net, scenario)
    if not os.path.isdir(measurements_path):
        print('Creating measurements folder: {}'.format(measurements_path))
        os.makedirs(measurements_path)
    else:
        print('Measurements folder exists: {}'.format(measurements_path))

    mlperf_dst_conf = os.path.join(measurements_path, 'mlperf.conf')
    if not os.path.isfile(mlperf_dst_conf):
        print('Copying mlperf.conf file: {}'.format(mlperf_dst_conf))
        copyfile(mlperf_conf, mlperf_dst_conf)
    else:
        print('mlperf.conf file exists: {}'.format(mlperf_dst_conf))

    results_path = os.path.join(results, system, net, scenario)
    if not os.path.isdir(results_path):
        print('Creating results folder: {}'.format(results_path))
        os.makedirs(results_path)
    else:
        print('Results folder exists: {}'.format(results_path))

    results_type = [ 'performance/run_1', 'accuracy' ]
    for typ in results_type:
        result_path = os.path.join(results_path, typ)
        if not os.path.isdir(result_path):
            print('Creating results folder: {}'.format(result_path))
            os.makedirs(result_path)
        else:
            print('Results folder exists: {}'.format(result_path))
        aliases['${}_DIR'.format(typ.upper())] = result_path

    systems_path = os.path.join(org, 'systems')
    if not os.path.isdir(systems_path):
        print('Creating systems folder: {}'.format(systems_path))
        os.makedirs(systems_path)
    else:
        print('Systems folder exists: {}'.format(systems_path))

    if 'exec' in config_dict['config']:
        for exec in config_dict['config']['exec']:
            print('Running {}'.format(exec['name']))
            dir = expand_paths(exec['dir'], aliases)
            cmd = expand_paths(exec['cmd'], aliases)
            if 'skip' in exec:
                if bool(exec['skip']):
                    print('Skipping...')
                    continue
            args = shlex.split(cmd)
            output = run_process(args, dir)
            if 'save_output' in exec:
                path = os.path.join(dir, exec['save_output'])
                f = open(path, 'w+')
                f.write(output)
                f.close()


if __name__ == '__main__':
    main()