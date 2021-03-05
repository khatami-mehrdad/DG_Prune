
import math
from collections import OrderedDict

def sense_analyzers_factory(classname):
    cls = globals()[classname]
    return cls

class SenseAnalyzerBase():
    def __init__(self, opt : dict):
        self.opt = opt
        self.num_stages = opt['num_stages']
        self.curr_sparsity = {key:0.0 for key in opt['weights'].keys()}
        self.stage_cnt = 0
        self.layer_cnt = 0

    def step(self, final_sparsity: float):
        pass

    def step_all(self):
        self.stage_cnt_next()
        layer_name = self.get_curr_layername()
        final_sparsity = self.opt['weights'][layer_name]
        self.curr_sparsity[layer_name] = self.step(final_sparsity)
        return self.curr_sparsity

    def get_curr_layername(self):
        return list(self.curr_sparsity)[self.layer_cnt] if not self.done() else 'DONE'

    def stage_cnt_next(self):
        self.stage_cnt = self.stage_cnt + 1
        if (self.stage_cnt > self.num_stages):
            layer_name = self.get_curr_layername()
            self.curr_sparsity[layer_name] = 0
            self.layer_cnt = self.layer_cnt + 1
            self.stage_cnt = 1  # we only have stage_cnt = 0 once in the begining
    
    def done(self):
        return self.layer_cnt >=  len(self.curr_sparsity)

    def get_sensitivity_state(self) -> OrderedDict : 
        layer_name = self.get_curr_layername()
        return OrderedDict([('layer', self.layer_cnt), ('layerName', layer_name), ('stage', self.stage_cnt), ('sparsity', self.curr_sparsity[layer_name])])

class Linear(SenseAnalyzerBase):
    r"""
    sparsity_val = end * (n / num_stages)
    """
    def __init__(self, opt : dict):
        super().__init__(opt)

    def step(self, final_sparsity: float):
        val =  final_sparsity * (self.stage_cnt / self.num_stages)
        return val

class Exponential(SenseAnalyzerBase):
    r"""
    """
    def __init__(self, opt : dict):
        super().__init__(opt)

    def step(self, final_sparsity: float):
        mult = 0
        for i in range(1, self.stage_cnt + 1):
            mult += math.pow(2, self.num_stages - i)
        return final_sparsity * mult / (math.pow(2, self.num_stages) - 1)