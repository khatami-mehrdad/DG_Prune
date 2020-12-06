
import math

def pruner_factory(classname):
    cls = globals()[classname]
    return cls

class PrunerBase():
    def __init__(self, opt : dict):
        self.opt = opt
        if ('ending_epoch' in opt) and ('starting_epoch' in opt) and ('frequency' in opt):
            self.num_stages = (opt['ending_epoch'] - opt['starting_epoch']) // opt['frequency']        
        elif 'num_stages' in opt:
            self.num_stages = opt['num_stages']

        self.curr_sparsity = {key:0.0 for key in opt['weights'].keys()}
        self.curr_grow = {key:0.0 for key in opt['weights'].keys()}
        self.stage_cnt = 0

    def compute_stage_cnt(self, epoch : float):
        self.stage_cnt = (epoch - self.opt['starting_epoch']) // self.opt['frequency']

    def prune_step(self, final_sparsity: float):
        return final_sparsity

    def grow_step(self, initial_grow: float):
        return 0

    def step_all(self, epoch: float):
        self.compute_stage_cnt(epoch)
        if (self.stage_cnt >= 0 and self.stage_cnt <= self.num_stages):
            for layer_name, final_sparsity in self.opt['weights'].items():
                self.curr_sparsity[layer_name] = self.prune_step(final_sparsity)
                self.curr_grow[layer_name] = self.grow_step()
        return self.curr_sparsity, self.curr_grow

class AGP(PrunerBase):
    r"""
    sparsity_val = end - (end - start) * ( 1 - (n / num_stages) )^3 )
    """
    def __init__(self, opt : dict):
        super().__init__(opt)

    def prune_step(self, final_sparsity: float):
        val =  final_sparsity - final_sparsity * ( (1.0 - (self.stage_cnt / self.num_stages)) ** self.opt['T'] )
        return val


class RigL(PrunerBase):
    r"""
    https://arxiv.org/pdf/1911.11134.pdf
    """
    def __init__(self, opt : dict):
        super().__init__(opt)

    def prune_step(self, final_sparsity: float):
        return final_sparsity
    
    def grow_step(self):
        val =  self.opt['alpha'] * ( (1 + math.cos(self.stage_cnt * math.pi / self.num_stages)) / 2 )
        return val

class LTH(PrunerBase):
    r"""
    https://arxiv.org/pdf/2003.02389.pdf
    """
    def __init__(self, opt : dict):
        super().__init__(opt)

    def prune_step(self, final_sparsity: float):
        return final_sparsity * ( (1 - math.pow(2, -self.stage_cnt)) if self.stage_cnt < self.num_stages else 1 ) 

    def compute_stage_cnt(self, epoch : float):
        self.stage_cnt = self.stage_cnt + 1