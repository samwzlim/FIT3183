"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import torch

class CheckpointIO(object):
    def __init__(self, fname_template, **module_dict):
        self.fname_template = fname_template
        self.module_dict = module_dict

    def save(self, step):
        """Save all modules."""
        fname = self.fname_template.format(step)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        module_dict = {k: v.module if hasattr(v, 'module') else v
                       for k, v in self.module_dict.items()}
        state_dict = {k: v.state_dict() for k, v in module_dict.items()}
        torch.save(state_dict, fname)

    def load(self, step):
        """Load all modules."""
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        module_dict = torch.load(fname, weights_only=True)
        for name, module in self.module_dict.items():
            if name in module_dict:
                print(f"Loading {name} with strict=False")
                # Allowing mismatched layers to be skipped during loading
                module.module.load_state_dict(module_dict[name], strict=False)
            else:
                print(f"Warning: {name} not found in checkpoint, skipping.")
