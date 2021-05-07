
import math
from collections import OrderedDict

def sense_analyzers_factory(classname):
    cls = globals()[classname]
    return cls

class SenseAnalyzerBase():
    def __init__(self, opt : dict):
        self.opt = opt
        self.starting_sparsity = opt['starting_sparsity'] if 'starting_sparsity' in opt.keys() else 0
        if ('num_stages' in opt.keys() and 'weights' in opt.keys()):
            self.num_stages = opt['num_stages']
            self.curr_sparsity = {key:0.0 for key in opt['weights'].keys()}
        else:
            self.num_stages = (opt['final_sparsity'] - self.starting_sparsity) // opt['step_size'] + 1
            self.curr_sparsity = {key:0.0 for key in opt['layer_names']}
        self.stage_cnt = 0
        self.layer_cnt = 0

    def step(self, final_sparsity: float):
        pass

    def step_all(self):
        self.stage_cnt_next()
        layer_name = self.get_curr_layername()
        if (layer_name != 'DONE'):
            final_sparsity = self.opt['weights'][layer_name] if 'weights' in opt.keys() else opt['final_sparsity']
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
            self.stage_cnt = 0
    
    def done(self):
        return self.layer_cnt >=  len(self.curr_sparsity)

    def get_sensitivity_state(self) -> OrderedDict : 
        layer_name = self.get_curr_layername()
        return OrderedDict([('layer', self.layer_cnt), ('layerName', layer_name), ('stage', self.stage_cnt), ('sparsity', self.curr_sparsity[layer_name])])

class Linear(SenseAnalyzerBase):
    r"""
    sparsity_val = start + (end - start) * (n / num_stages)
    """
    def __init__(self, opt : dict):
        super().__init__(opt)

    def step(self, final_sparsity: float):
        val =  self.starting_sparsity + (final_sparsity - self.starting_sparsity) * (self.stage_cnt / self.num_stages)
        return val

class Exponential(SenseAnalyzerBase):
    r"""
    """
    def __init__(self, opt : dict):
        super().__init__(opt)
        self.T = opt['T'] if 'T' in opt.keys() else 2

    # def step(self, final_sparsity: float):
    #     mult = 0
    #     for i in range(1, self.stage_cnt + 1):
    #         mult += math.pow(self.T, self.num_stages - i)
    #     return final_sparsity * mult / (math.pow(self.T, self.num_stages) - 1)

    def step(self, final_sparsity: float):
        val =  final_sparsity - final_sparsity * ( (1.0 - (self.stage_cnt / self.num_stages)) ** self.T )
        return val