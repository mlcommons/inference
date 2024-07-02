import json
import random
# Open the JSON file
with open('./data/cnn_eval.json', 'r') as file:
    # Load the JSON data
    data = json.load(file)

# Now 'data' is a Python dictionary containing the JSON content
# You can access values using dictionary notation

random_subset = random.sample(data, 1000)


with open('./data/tendency_test_eval.json', 'w') as json_file:
    json.dump(random_subset, json_file, indent='\t', separators=(',', ': '))