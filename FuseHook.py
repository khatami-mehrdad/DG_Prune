
from torch.quantization.fuse_modules import fuse_modules, fuse_conv_bn
from torch import nn
import torch.nn.intrinsic.modules.fused as torch_fused

class Fuse_Hook():
    def __init__(self, model, fuse_type_list : list):
        self.modules_to_fuse = []
        self.fuse_list = fuse_type_list
        self.module_names = [[] * len(fuse_type_list)]
        self.fuse_stage = [0] * len(fuse_type_list)
        for name, m in model.named_modules():
            m.register_forward_hook(self.hook_fn)
            m.name = name
    
    def CheckFuse(self, module):
        for f in range( len(self.fuse_list) ):
            fuse_type = self.fuse_list[f]
            if (type(module) == fuse_type[ self.fuse_stage[f] ]):
                self.fuse_stage[f] = self.fuse_stage[f] + 1
                self.module_names[f].append(module.name)
                if ( self.fuse_stage[f] == len(fuse_type) ):
                    self.modules_to_fuse.append(self.module_names[f])
                    self.reset(f)
            else:
                self.reset(f)

    def reset(self, f : int):
        self.fuse_stage[f] = 0
        self.module_names[f] = []        

    def hook_fn(self, module, input, output):
        self.CheckFuse(module)
