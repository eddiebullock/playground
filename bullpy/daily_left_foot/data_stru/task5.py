# Write a dictionary to JSON and read it back.
import json 

my_dict= {
    'name' : 'the bull',
    'alias' : 'el torro',
    'age' : '50',
}

def write_json(data, filename='bull_dict.json'):
    with open(filename, 'w') as f:
        json.dump(data, f)

def read_json(filename='bull_dict.json'):
    with open(filename, 'r') as f:
        return json.load(f)
    
#call funtions 
write_json(my_dict)

load = read_json()
print(load)