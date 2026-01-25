import random 

random_sample = random.sample(range(1, 101), 50)
print(random_sample)

#normalize function 
def normalize(data):
    max_val = max(data)
    return(x / max_val for x in data)

normalized_data = list(normalize(random_sample))
print(normalized_data)

# store normalized data in dict 
my_dict = {i: val for i, val in enumerate(normalized_data, start=1)}
print(my_dict)

# write dict to json file and read it back
import json 

def write_json(data, filename='nrml_data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f)

def read_json(filename='nrml_data.json'):
    with open(filename, 'r') as f:
        return json.load(f)

write_json(my_dict)

readjson = read_json()
print(readjson)

# create a dataset class to load and access these numbers 
class NormalizedDataset:
    def __init__(self, filename='nrml_data.json'):
        self.data = read_json(filename)
    
    def get_data(self, index):
        return self.data.get(str(index))
    
# usage 
dataset = NormalizedDataset()
print(f"value at 9th position: {dataset.get_data(9)}")
