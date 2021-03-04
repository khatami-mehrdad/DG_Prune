
from torch.quantization.fuse_modules import fuse_modules, fuse_conv_bn
from torch import nn
import torch.nn.intrinsic.modules.fused as torch_fused

class Fuse_ConvBN_Hook():
    def __init__(self, model):
        self.modules_to_fuse = []
        self.prev_module_name = ''
        self.is_prev_conv = False
        for name, m in model.named_modules():
            m.register_forward_hook(self.hook_fn)
            m.name = name
    
    def CheckFuse(self, module):
        if (type(module) == nn.Conv2d):
            self.is_prev_conv = True
        else:
            self.Stage2_Conv2dBN(module)
            self.is_prev_conv = False

    def Stage2_Conv2dBN(self, module):
        if ( self.is_prev_conv & (type(module) == nn.BatchNorm2d) ):
            self.modules_to_fuse.append([self.prev_module_name, module.name])

    def hook_fn(self, module, input, output):
        self.CheckFuse(module)
        self.prev_module_name = module.name


def modified_fuse_known_modules(mod_list):
    OP_LIST_TO_FUSER_METHOD = {
        (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    }

    types = tuple(type(m) for m in mod_list)
    fuser_method = OP_LIST_TO_FUSER_METHOD.get(types, None)
    if fuser_method is None:
        raise NotImplementedError("Cannot fuse modules: {}".format(types))
    new_mod = [None] * len(mod_list)
    new_mod[0] = fuser_method(*mod_list)

    for i in range(1, len(mod_list)):
        new_mod[i] = nn.Identity()
        new_mod[i].training = mod_list[0].training

    return new_mod

def fuse_modules(model_not_fused : nn.modules)
    model_not_fused.eval()
    h = FuseHook(model_not_fused)
    for image, target in data_loader_test:
        output_test = model_not_fused(image[0].unsqueeze(0))
        break
    model = fuse_modules(model_not_fused, h.modules_to_fuse, inplace=False, fuser_func=modified_fuse_known_modules )
    return model