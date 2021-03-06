
from torch.quantization.fuse_modules import fuse_modules, fuse_conv_bn
from torch import nn
import torch.nn.intrinsic.modules.fused as torch_fused

class Fuse_Hook():
    def __init__(self, model, fuse_type_list : list):
        self.modules_to_fuse = []
        self.fuse_list = fuse_type_list
        self.module_names = [[] * len(fuse_type_list)]
        self.fuse_stage = [0] * len(fuse_type_list)
        self.hook_handles = []
        for name, m in model.named_modules():
            self.hook_handles.append( m.register_forward_hook(self.hook_fn) )
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

    def remove(self):
        for h in self.hook_handles:
            h.remove()

def get_modules_to_fuse(model, fuse_type_list : list, sample_input):
    model.to('cpu')
    model.eval()
    # fuse_type_list = [ [PrunableConv2d, nn.BatchNorm2d] ]
    h = Fuse_Hook(model, fuse_type_list)
    output_test = model( sample_input )
    h.remove()
    return h.modules_to_fuse
