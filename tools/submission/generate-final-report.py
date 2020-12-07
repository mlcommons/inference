"""
Tool to generate the final results speadsheet from the checker csv output.
The resulting excel files can be imported into google sheets.
"""
import argparse
import os
import sys
import re
import numpy as np
import pandas as pd


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="results csv from checker")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    df = pd.read_csv(args.input).fillna("")

    # rename some fields
    df.rename(columns={
        "Organization": "Submitter",
        "Division": "Category",
        "SystemType": "Suite",
        "SystemName": "System",
        "host_processor_model_name": "Processor",
        "accelerator_model_name": "Accelerator",
        "accelerators_per_node": "a#",
        "notes": "Notes",
    }, inplace=True)
    df.rename(columns={"Model": "UsedModel"}, inplace=True)
    df.rename(columns={"MlperfModel": "Model"}, inplace=True)

    # fix issues with raw data
    df['host_processor_core_count'] = df['host_processor_core_count'].apply(lambda x: 2 if x == '2 (big); 4 (LITTLE)' else x)
    df['Availability'] = df['Availability'].apply(lambda x: "available" if x == 'on-premise' else x)

    # cleanup counts
    df['Accelerator'] = df['Accelerator'].apply(lambda x: x if x != "-" else "")
    df['a#'] = df['a#'].apply(lambda x: int(x) if x != "" else 0)
    df['a#'] = df['a#'].apply(lambda x: x if x > 0 else "")
    df['p#'] = df.apply(lambda x: int(x['host_processor_core_count']) * int(x['host_processors_per_node']), axis=1)

    # details url
    base_url = "https://github.com/mlperf/submissions_inference_0_7/tree/master"
    df['Details'] = df.apply(
        lambda x: '=HYPERLINK("{}","details")'.format("/".join([base_url, x['Category'], x['Submitter'], "results", x['Platform']])), axis=1)

    output = args.input[:-4]
    writer = pd.ExcelWriter(output + '.xlsx', engine='xlsxwriter')

    index = [
        'Unique ID (e.g. for Audit)', 'Submitter',
        'Availability', 'System', 'Processor', "p#", 'Accelerator', "a#",
        "Notes", "Details",
    ]
    columns = [
        'Model', 'Scenario',
    ]

    # closed
    df['Unique ID (e.g. for Audit)'] = df.apply(
        lambda x: "/".join([x['Suite'], x['Category'], x['Submitter'], x['Platform']]), axis=1)
    df1 = df[(df['Category'] == "closed") & (df['Suite'] == "datacenter")].pivot_table(index=index, columns=columns, values=['Result']).fillna("")
    df1.to_excel(writer, sheet_name="closed,datacenter")
    df1 = df[(df['Category'] == "closed") & (df['Suite'] == "edge")].pivot_table(index=index, columns=columns, values=['Result']).fillna("")
    df1.to_excel(writer, sheet_name="closed,edge")

    # open
    df['Unique ID (e.g. for Audit)'] = df.apply(
        lambda x: "/".join([x['Suite'], x['Category'], x['Submitter'], x['Platform'], x['UsedModel']]), axis=1)
    df1 = df[(df['Category'] == "open") & (df['Suite'] == "datacenter")].pivot_table(index=index, columns=columns, values=['Result']).fillna("")
    df1.to_excel(writer, sheet_name="open,datacenter")
    df1 = df[(df['Category'] == "open") & (df['Suite'] == "edge")].pivot_table(index=index, columns=columns, values=['Result']).fillna("")
    df1.to_excel(writer, sheet_name="open,edge")
    
    writer.save()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
