import argparse
import re
import json

# Parser for command line
parser = argparse.ArgumentParser(description='Input log, model, and mode.')
parser.add_argument('--log', action='store', type=str,
                    dest='log_filename', help='Path of log file')
parser.add_argument('--spec', action='store', type=str,
                    dest='spec_filename', help='Path of spec JSON file')
parser.add_argument('--model', action='store', type=str, dest='model_name',
                    help='Model name you are testing (e.g. Resnet50-v1.5)')

results = parser.parse_args()

bool_dict = {}

# Scrape lines from txt file
with open(results.log_filename) as logfile:
    log_lines = logfile.readlines()

log_lines = [line.strip() for line in log_lines]

# Import JSON file
with open(results.spec_filename) as json_file:
    json_dict = json.load(json_file)

def find_string(iter_list, substring):
    """Return index if a specific substring is inside each string in a list."""
    for i, s in enumerate(iter_list):
        if substring in s:
            return i


effective_string = 'Effective Settings:'
requested_string = 'Requested Settings:'

effective_index = find_string(log_lines, effective_string)
requested_index = find_string(log_lines, requested_string)

effective_settings = log_lines[effective_index+1:requested_index-1]
effective_dict = {}


# Find every ":" in each line
def find_char(character, string):
    """Returns index for every time a character is found in a string."""
    colon_counter = []
    for m in re.finditer(character, string):
        colon_counter.append(m.start())
    return colon_counter


def slice_colons(string, character, index):
    """Slices a string to remove everything before a specific character."""
    return string[find_char(character, string)[index] + 2:]


def get_key_value(string, character, index):
    """Creates a list with a key value pair from parsed log file line."""
    value = string[find_char(character, string)[index] + 2:]
    if value.isdigit():
        value = int(value)
    return [string[0: find_char(character, string)[index]].strip(), value]


for i, s in enumerate(effective_settings):
    """Parses the effective settings list and creates a dictionary out of it."""
    effective_settings[i] = slice_colons(effective_settings[i], ':', 3)
    effective_dict[get_key_value(effective_settings[i], ':', 0)[0]] = get_key_value(effective_settings[i], ':', 0)[1]

keys = list(effective_dict.keys())
exit_code = 0


def check_in_list(spec_dict, log_dict, result_dict, key):
    """Checks if attribute from log file equals one of the attributes in a list in the spec dictionary."""
    for b in range(len(spec_dict)):
        if spec_dict[b] == log_dict:
            result_dict[key] = True 
    if key not in result_dict:
        result_dict[key] = False


def check_in_dict(spec_dict, parsed_results, log_dict, result_dict, key):
    """Checks if attribute from log file equals one of the attributes in a dict in the spec dictionary."""
    dict_keys = list(spec_dict.keys())
    for b in range(len(spec_dict)):
        if dict_keys[b] == parsed_results.model_name:
            if spec_dict[dict_keys[b]] <= log_dict:
                # For MultiStream and Server: target_latency or min_query_count matches
                result_dict[key] = True
    if key not in result_dict:
        result_dict[key] = False


def check_samples_per_queries():
    """Checks the specific samples_per_queries attributes."""

    # if singlestream/server, must be equal
    if json_dict['Scenarios'][a][keys[0]] == 'SingleStream' or json_dict['Scenarios'][a][keys[0]] == 'Server':
        if json_dict['Scenarios'][a][keys[d]] == effective_dict[keys[d]]:
            bool_dict[keys[d]] = True
        else:
            bool_dict[keys[d]] = False
    elif json_dict['Scenarios'][a][keys[0]] == 'Offline':
        if json_dict['Scenarios'][a][keys[d]] <= effective_dict[keys[d]]:
            bool_dict[keys[d]] = True
        else:
            bool_dict[keys[d]] = False
    elif json_dict['Scenarios'][a][keys[0]] == 'MultiStream':
        bool_dict[keys[d]] = True


def check_greater_than():
    """Checks if attribute is greater than or equal to attribute in spec dict."""
    if (json_dict['Scenarios'][a][keys[0]] == 'Offline'
        or json_dict['Scenarios'][a][keys[0]] == 'SingleStream'
            or json_dict['Scenarios'][a][keys[0]] == 'Server'):

        if json_dict['Scenarios'][a][keys[d]] <= effective_dict[keys[d]]:
            bool_dict[keys[d]] = True
        else:
            bool_dict[keys[d]] = False


def default_singlestream_offline_true():
    """Checks for attributes that don't directly apply to SingleStream or Offline."""
    if json_dict['Scenarios'][a][keys[0]] == 'SingleStream' or json_dict['Scenarios'][a][keys[0]] == 'Offline':
        bool_dict[keys[d]] = True

def check_normal_attribute():
    "Checks if normal attributes equal attributes in the spec dict."
    if json_dict['Scenarios'][a][keys[d]] == effective_dict[keys[d]]:
        bool_dict[keys[d]] = True
    else:
        bool_dict[keys[d]] = False

# first iterate through the Scenarios list
for a in range(len(json_dict['Scenarios'])):
    """For Loop for final check"""
    # if the Scenario value matches, then go deeper into loop
    if json_dict['Scenarios'][a][keys[0]] == effective_dict[keys[0]]:
        exit_code = 0
        bool_dict[keys[0]] = True  # Scenario matches

        for d in range(1, len(json_dict['Scenarios'][a])):
            """Iterates through every attribute"""
            # check for list or dict
            if type(json_dict['Scenarios'][a][keys[d]]) == list:
                """See if the value matches any of the list values"""
                check_in_list(json_dict['Scenarios'][a][keys[d]],
                              effective_dict[keys[d]], bool_dict, keys[d])
            elif type(json_dict['Scenarios'][a][keys[d]]) == dict:
                """See if the value matches any of the dict values"""
                check_in_dict(json_dict['Scenarios'][a][keys[d]],
                              results, effective_dict[keys[d]], bool_dict, keys[d])
            else:  # so attribute is a number
                """Checks rest of the attributes"""
                if d == 2:
                    check_samples_per_queries()
                elif d == 7 or d == 9 or d == 11:
                    check_greater_than()
                elif d == 4:
                    default_singlestream_offline_true()
                else:
                    check_normal_attribute()
        if all(value == True for value in bool_dict.values()) == True:
            exit_code = 0
        else:
            exit_code = 1
        break
    else:
        exit_code = 2

print("\nExit Code = " + str(exit_code) + "\n")
print("Attribute Complies? \n" + str(bool_dict) + "\n")
