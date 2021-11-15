
from . import prune as prn
from . import sensitivity as sense
import torch.nn as nn
import copy
from collections import OrderedDict
import csv

class DG_Pruner():
    def __init__(self):
        self.hooks = {}
        self.pruners = {}
        self.sense_analyzers = {}

    @staticmethod
    def swap_prunable_modules(model : nn.Module):
        return prn.swap_prunable_modules(model)

    @staticmethod
    def strip_prunable_modules(old_model : nn.Module):
        model = copy.deepcopy(old_model)
        return prn.strip_prunable_modules(model)

    @staticmethod
    def attach_bn_to_prunables(model : nn.Module, fuse_list : list):
        prn.attach_bn_to_prunables(model, fuse_list)

    def add_custom_pruning(self, model : nn.Module, custom_class, parent_name : str = ''):
        self.hooks = prn.add_custom_pruning(model, custom_class, parent_name)
        return self.hooks

    def pruners_from_file(self, file_path : str):
        self.pruners = prn.pruners_from_file(file_path)
        return self.pruners

    def sense_analyzers_from_file(self, file_path : str):
        self.sense_analyzers = sense.sense_analyzers_from_file(file_path)
        return self.sense_analyzers

    def sense_analyzers_from_dict(self, sense_analyzers_dict : dict):
        self.sense_analyzers = sense.sense_analyzers_from_dict(sense_analyzers_dict)
        return self.sense_analyzers

    def reset_importance(self):
        for h in self.hooks.values():
            h.reset_importance()

    def reset_growth(self):
        for h in self.hooks.values():
            h.reset_growth()

    def apply_pruning_step(self, epoch: float):
        prn.apply_pruning_step(epoch, self.pruners, self.hooks)

    def apply_sensitivity_step(self):
        sense.apply_sensitivity_step(self.sense_analyzers, self.hooks)

    def apply_mask_to_weight(self):
        prn.apply_mask_to_weight(self.hooks)

    @staticmethod
    def remove_mask_lt_thr(model : nn.Module, thr : float = 0.5):
        return prn.remove_mask_lt_thr(model, thr)

    @staticmethod
    def apply_zero_weight_to_mask(model : nn.Module):
        return prn.apply_zero_weight_to_mask(model)

    def dump_importance_stat(self, output_dir : str = '', epoch : int = 0):
        prn.dump_importance_stat(self.hooks, output_dir, epoch)

    def dump_growth_stat(self, output_dir : str = '', epoch : int = 0):
        prn.dump_growth_stat(self.hooks, output_dir, epoch)

    @staticmethod
    def dump_sparsity_stat_mask_base(model : nn.Module, output_dir : str = '', epoch : int = 0):
        prn.dump_sparsity_stat_mask_base(model, output_dir, epoch)

    @staticmethod
    def dump_sparsity_stat_weight_base(model : nn.Module, output_dir : str = '', epoch : int = 0):
        prn.dump_sparsity_stat_weight_base(model, output_dir, epoch)

    @staticmethod
    def get_prunable_module_names(model : nn.Module, output_dir : str = ''):
        return prn.get_prunable_module_names(model, output_dir)

    @staticmethod
    def dump_json(data: dict, file_name : str, output_dir : str = ''):
        prn.dump_json(data, file_name, output_dir)

    def compute_sparsity_table_from_layer(self, layer_name : str, sparsity : float):
        return prn.compute_sparsity_table_from_layer(self.hooks, layer_name, sparsity)

    def prune_n_reset(self, epoch : float):
        self.apply_pruning_step(epoch)
        self.reset_importance()        
        self.reset_growth()

    def num_iter_per_update(self, num_batch_per_epoch : int) -> int:
        return round( self.pruners[list(self.pruners)[0]].opt['frequency'] * num_batch_per_epoch )

    def rewind_epoch(self, total_epochs : int) -> int:
        if (self.pruners[list(self.pruners)[0]].opt['rewind_epoch'] < 1):
            return int( self.pruners[list(self.pruners)[0]].opt['rewind_epoch'] * total_epochs )
        else:
            return int( self.pruners[list(self.pruners)[0]].opt['rewind_epoch'] )

    def num_stages(self) -> int:
        return self.pruners[list(self.pruners)[0]].num_stages

    def save_rewind_checkpoint(self, checkpoint:dict):
        self.rewind_checkpoint = copy.deepcopy(checkpoint)
    
    def save_final_checkpoint(self, checkpoint:dict):
        self.final_checkpoint = copy.deepcopy(checkpoint)

    def rewind_masked_checkpoint(self, state_dict: str, state_dict_wMask: str = 'model') -> dict:
        for k in self.rewind_checkpoint[state_dict].keys():
            if k.endswith('mask'):
                self.rewind_checkpoint[state_dict][k] = self.final_checkpoint[state_dict_wMask][k]
        return self.rewind_checkpoint

    def mask_unmasked_checkpoint(self, unmaskedCheckpoint : dict, maskedCheckpoint : dict) -> dict:
        for k in unmaskedCheckpoint.keys():
            if k.endswith('mask'):
                unmaskedCheckpoint[k] = maskedCheckpoint[k]
        return unmaskedCheckpoint

    def get_final_checkpoint(self) -> dict:
        return self.final_checkpoint
    
    def get_rewind_checkpoint(self) -> dict:
        return self.rewind_checkpoint
    
    def sense_done(self) -> int:
        return self.sense_analyzers[list(self.sense_analyzers)[0]].done()

    def get_sensitivity_state(self) -> OrderedDict : 
        return self.sense_analyzers[list(self.sense_analyzers)[0]].get_sensitivity_state()

    def update_summary(self, eval_metrics, filename, write_header=False):
        rowd = OrderedDict()
        rowd.update( self.get_sensitivity_state().items() )
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        with open(filename, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=rowd.keys())
            if write_header:  # first iteration (epoch == 1 can't be used)
                dw.writeheader()
            dw.writerow(rowd)



    