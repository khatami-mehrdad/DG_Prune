
import json
import os

def read_json_from_file(file_path : str):
    with open(file_path) as f:
        data = json.load(f)
    return data 

def dump_json(data: dict, file_name : str, output_dir : str = ''):
    with open(os.path.join(output_dir, file_name), 'w') as fp:
        fp.write(json.dumps(data, indent=1))