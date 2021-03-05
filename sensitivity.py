from .sensitivity_analyzer import sense_analyzers_factory
from .utils import read_json_from_file

def sense_analyzers_from_file(file_path : str):
    sense_analyzers_obj_dict = {}
    sense_analyzers_dict = read_json_from_file(file_path)
    for name, sense_analyzers_property in sense_analyzers_dict.items():
        sense_analyzers_obj_dict[name] = create_sense_analyzers(sense_analyzers_property)
    return sense_analyzers_obj_dict

def create_sense_analyzers(sense_analyzers_property):
    cls_ = sense_analyzers_factory(sense_analyzers_property["class"])
    return cls_(sense_analyzers_property)