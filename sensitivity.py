from .sensitivity_analyzer import sense_analyzers_factory
from .utils import read_json_from_file

def sense_analyzers_from_file(file_path : str):
    sense_analyzers_dict = read_json_from_file(file_path)
    return sense_analyzers_from_dict(sense_analyzers_dict)

def sense_analyzers_from_dict(sense_analyzers_dict : dict):
    sense_analyzers_obj_dict = {}
    for name, sense_analyzers_property in sense_analyzers_dict.items():
        sense_analyzers_obj_dict[name] = create_sense_analyzers(sense_analyzers_property)
    return sense_analyzers_obj_dict

def create_sense_analyzers(sense_analyzers_property):
    cls_ = sense_analyzers_factory(sense_analyzers_property["class"])
    return cls_(sense_analyzers_property)

## Applying Sparsity
def apply_sensitivity_step(sense_analyzers: dict, hooks: dict):
    for sense_analyzer in sense_analyzers.values():
        curr_sparsity = sense_analyzer.step_all()
        if ( (sense_analyzer.stage_cnt >= 0) and (sense_analyzer.stage_cnt <= sense_analyzer.num_stages) ):
            for name, sparsity in curr_sparsity.items():
                hooks[name].apply_sparsity( sparsity )