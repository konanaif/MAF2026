################################################################################
# nethook (CausalPathTracing_anonymous/CPT/vision/lib/nethook.py)
################################################################################
"""
Utilities for instrumenting a torch model.

Trace will hook one layer at a time.
TraceDict will hook multiple layers at once.
subsequence slices intervals from Sequential modules.
get_module, replace_module, get_parameter resolve dotted names.
set_requires_grad recursively sets requires_grad in module parameters.
"""

import contextlib
import copy
import inspect
from collections import OrderedDict

import torch


class Trace(contextlib.AbstractContextManager):
    """
    To retain the output of the named layer during the computation of
    the given network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and
    its output will be retained.  By default, a direct reference to
    the output object is returned, but options can control this:

        clone=True  - retains a copy of the output, which can be
            useful if you want to see the output before it might
            be modified by the network in-place later.
        detach=True - retains a detached reference or copy.  (By
            default the value would be left attached to the graph.)
        retain_grad=True - request gradient to be retained on the
            output.  After backward(), ret.output.grad is populated.

        retain_input=True - also retains the input.
        retain_output=False - can disable retaining the output.
        edit_output=fn - calls the function to modify the output
            of the layer before passing it the rest of the model.
            fn can optionally accept (output, layer) arguments
            for the original output and the layer name.
        stop=True - throws a StopForward exception after the layer
            is run, which allows running just a portion of a model.
    """

    def __init__(
        self,
        module,
        layer=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        retainer = self
        self.layer = layer
        if layer is not None:
            module = get_module(module, layer)

        def retain_hook(m, inputs, output):
            if retain_input:
                retainer.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )  # retain_grad applies to output only.
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations
                # to follow without error.
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    To retain the output of multiple named layers during the computation
    of the given network:

        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    two arguments: output, and the layer name; and then it returns the
    modified output.

    Other arguments are the same as Trace.  If stop is True, then the
    execution of the network will be stopped after the last layer
    listed (even if it would not have been the last to be executed).
    """

    def __init__(
        self,
        module,
        layers=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        self.stop = stop

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()



class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """

    pass


def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


def subsequence(
    sequential,
    first_layer=None,
    last_layer=None,
    after_layer=None,
    upto_layer=None,
    single_layer=None,
    share_weights=False,
):
    """
    Creates a subsequence of a pytorch Sequential model, copying over
    modules together with parameters for the subsequence.  Only
    modules from first_layer to last_layer (inclusive) are included,
    or modules between after_layer and upto_layer (exclusive).
    Handles descent into dotted layer names as long as all references
    are within nested Sequential models.

    If share_weights is True, then references the original modules
    and their parameters without copying them.  Otherwise, by default,
    makes a separate brand-new copy.
    """
    assert (single_layer is None) or (
        first_layer is last_layer is after_layer is upto_layer is None
    )
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    first, last, after, upto = [
        None if d is None else d.split(".")
        for d in [first_layer, last_layer, after_layer, upto_layer]
    ]
    return hierarchical_subsequence(
        sequential,
        first=first,
        last=last,
        after=after,
        upto=upto,
        share_weights=share_weights,
    )


def hierarchical_subsequence(
    sequential, first, last, after, upto, share_weights=False, depth=0
):
    """
    Recursive helper for subsequence() to support descent into dotted
    layer names.  In this helper, first, last, after, and upto are
    arrays of names resulting from splitting on dots.  Can only
    descend into nested Sequentials.
    """
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    assert isinstance(sequential, torch.nn.Sequential), (
        ".".join((first or last or after or upto)[:depth] or "arg") + " not Sequential"
    )
    including_children = (first is None) and (after is None)
    included_children = OrderedDict()
    # A = current level short name of A.
    # AN = full name for recursive descent if not innermost.
    (F, FN), (L, LN), (A, AN), (U, UN) = [
        (d[depth], (None if len(d) == depth + 1 else d))
        if d is not None
        else (None, None)
        for d in [first, last, after, upto]
    ]
    for name, layer in sequential._modules.items():
        if name == F:
            first = None
            including_children = True
        if name == A and AN is not None:  # just like F if not a leaf.
            after = None
            including_children = True
        if name == U and UN is None:
            upto = None
            including_children = False
        if including_children:
            # AR = full name for recursive descent if name matches.
            FR, LR, AR, UR = [
                n if n is None or n[depth] == name else None for n in [FN, LN, AN, UN]
            ]
            chosen = hierarchical_subsequence(
                layer,
                first=FR,
                last=LR,
                after=AR,
                upto=UR,
                share_weights=share_weights,
                depth=depth + 1,
            )
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:  # just like L if not a leaf.
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError("Layer %s not found" % ".".join(name))
    # Omit empty subsequences except at the outermost level,
    # where we should not return None.
    if not len(included_children) and depth > 0:
        return None
    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result


def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def replace_module(model, name, new_module):
    """
    Replaces the named module within the given model.
    """
    if "." in name:
        parent_name, attr_name = name.rsplit(".", 1)
        model = get_module(model, parent_name)
    # original_module = getattr(model, attr_name)
    setattr(model, attr_name, new_module)


def invoke_with_optional_args(fn, *args, **kwargs):
    """
    Invokes a function with only the arguments that it
    is written to accept, giving priority to arguments
    that match by-name, using the following rules.
    (1) arguments with matching names are passed by name.
    (2) remaining non-name-matched args are passed by order.
    (3) extra caller arguments that the function cannot
        accept are not passed.
    (4) extra required function arguments that the caller
        cannot provide cause a TypeError to be raised.
    Ordinary python calling conventions are helpful for
    supporting a function that might be revised to accept
    extra arguments in a newer version, without requiring the
    caller to pass those new arguments.  This function helps
    support function callers that might be revised to supply
    extra arguments, without requiring the callee to accept
    those new arguments.
    """
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    # Pass positional args that match name first, then by position.
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]
            )
    # Fill unmatched positional args with unmatched keyword args in order.
    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(
                    argspec.args[u] for u in unmatched_pos if u < defaulted_pos
                )
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")
    # Pass remaining kw args if they can be accepted.
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
    # Pass remaining positional args if they can be accepted.
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)


################################################################################
# models (CausalPathTracing_anonymous/CPT/vision/lib/models.py)
################################################################################
import re
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import timm

# Create a wrapper class to match the expected interface
class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.config = model.default_cfg
        
        # Create a nested model structure to match the expected interface
        class NestedModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

                # Handle ViT and PiT block structure
                if hasattr(model, 'transformers'):  # PiT
                    self.transformers = model.transformers
                    # Flatten all blocks from all transformer modules
                    self.blocks = []
                    for t in model.transformers:
                        self.blocks.extend(t.blocks)
                elif hasattr(model, 'blocks'):  # ViT
                    self.blocks = model.blocks
                else:
                    raise ValueError("Model must have either 'blocks' or 'transformers' attribute.")

                self.patch_embed = getattr(model, 'patch_embed', None)
            

            def forward(self, **kwargs):
                # Forward pass through the ViT model
                if 'pixel_values' in kwargs:
                    x = kwargs['pixel_values']
                    output = self.model(x)
                    # Convert output tuple to dictionary format
                    if isinstance(output, tuple):
                        return {"logits": output[0]}  # Assuming first element contains logits
                    return {"logits": output}
                else:
                    raise ValueError("ViT models expect pixel_values")
        
        self.model = NestedModel(model)

        # Add pre-forward hook to handle tuple inputs for normalization layers
        for module in self.model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.register_forward_pre_hook(self._handle_tuple_input)

    def _handle_tuple_input(self, module, inputs):
        # inputs could be a single tuple or a tuple of tuples
        if not inputs:  # empty tuple
            return inputs
        if isinstance(inputs[0], tuple):
            return (inputs[0][0],)  # Return first element of first tuple
        if isinstance(inputs, tuple):
            return (inputs[0],)  # Return first element as single-element tuple
        return inputs

    def __call__(self, **kwargs):
        # Handle both pixel_values and input_ids for compatibility
        if 'pixel_values' in kwargs:
            return self.model(**kwargs)
        else:
            raise ValueError("No input provided")

    def forward(self, x):
        # Ensure input is a tensor before passing to model
        if isinstance(x, tuple):
            x = x[0]
        return self.model(x)

class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """
    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )
            set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


################################################################################
# args (CausalPathTracing_anonymous/CPT/vision/lib/args.py)
################################################################################
import argparse

def arg_func(arg_list=None):

        
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="6", type=str, help='gpu')
    parser.add_argument('--end_symbol', default=[".", "?"])
    parser.add_argument('--save_root', default=os.path.join(str(MAF_ROOT), "result", "casual_path_tracing", "run"), type=str)
    parser.add_argument('--load_path_saver', default=None, type=str) # folder

    parser.add_argument('--num_noise_sample', default=10, type=int)
    parser.add_argument('--model', default="vit_tiny_patch16_224", choices=['vit_tiny_patch16_224'], type=str)
    parser.add_argument('--dataset', default="imagenet", choices=['imagenet'], type=str)
    parser.add_argument('--data_root', default=None, type=str, help='ImageNet root (torchvision layout). None: sample.IMAGENET_ROOT')
    parser.add_argument('--noise_type', default="other", required=True, choices=['other', 'emb_added', 'zero','mean'], help="noise type")

    parser.add_argument('--debug_index', default=None, type=int)
    parser.add_argument('--subset_search', default="minimality", type=str)
    parser.add_argument('--efficient_mode', help='not check counterfactual', action='store_true')
    parser.add_argument('--detailed_save', help='detailed_save', action='store_true')
    parser.add_argument('--slightly_quiet_mode', help='slightly_quiet_mode', action='store_true')
    parser.add_argument('--debug_min_search', help='debug_min_search', action='store_true')
    parser.add_argument('--debug_min_search_start_block', default=8, type=int)
    
    parser.add_argument('--target_sample_idx', default=None, type=int)
    parser.add_argument('--evaluation_mode', help='evaluation_mode', action='store_true')
    args = parser.parse_args(arg_list)

    args.out_num = 1

    return args


def build_arg_list(**kwargs):
    # {'model': 'gpt2-xs', 'efficient_mode': True} -> ['--model', 'gpt2-xs', '--efficient_mode']
    arg_list = []
    for k, v in kwargs.items():
        if v is None or v is False:
            continue
        if v is True:
            arg_list.append('--{}'.format(k))
        else:
            arg_list.extend(['--{}'.format(k), str(v)])
    return arg_list


################################################################################
# utils (CausalPathTracing_anonymous/CPT/vision/lib/utils.py)
################################################################################
import os
import sys
import logging
import tarfile
import time
import copy
import torch
import json
import tqdm
import dill 
import types
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from datetime import datetime

from functools import wraps
from pathlib import Path
from MAF.datamodule.casual_path_tracing.sample import ImageNetDataset, SAMPLE_DIR, IMAGENET_ROOT, MAF_ROOT
from MAF.datamodule.casual_path_tracing.download_data import MODEL_DIR

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = {"imagenet": IMAGENET_ROOT}  # overwritten by Vision_Causal_Path_Tracing(data_root=...)



def set_utils(args):

    # Save Folder
    if "debug" not in args.save_root:
        if os.path.isdir(args.save_root) is True:
            print("Check your path!")
            import pdb; pdb.set_trace()

    curr_time = datetime.now().strftime('%Y%m%d%H%M')
    save_parent, save_name = os.path.split(os.path.normpath(args.save_root))

    args.save_root = os.path.join(save_parent, args.model if "/" not in args.model else args.model.split("/")[-1], "{}_{}".format(save_name, curr_time))
    os.makedirs(args.save_root, exist_ok=True)

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(args.save_root, "log.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Timer
    # if "debug" in args.save_root:
    func_time_saver = function_time_saver()
    current_dir = PKG_DIR
    imported_modules = {}
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if getattr(module, '__file__', '') is None:
            continue
        if module and getattr(module, '__file__', '').startswith(current_dir):
            imported_modules.update({name: module})
    for module_name, module in imported_modules.items():
        for name, func in vars(module).items():
            if callable(func) and func.__module__ == module_name and not getattr(func, '__is_decorated__', False):
                if func.__name__ == "measure_time_and_memory":
                    continue
                decorated_func = measure_time_and_memory(logger=logger, func_time_saver=func_time_saver)(measure_time_and_memory(logger=logger, func_time_saver=func_time_saver)(func))
                setattr(module, name, decorated_func)

    # Code Backup
    tar = tarfile.open( os.path.join(args.save_root, 'sources.tar'), 'w' )
    tar.add(PKG_DIR, arcname=os.path.basename(PKG_DIR), filter=lambda t: None if "__pycache__" in t.name else t)
    tar.close()

    # Arg Backup
    with open(os.path.join(args.save_root,'args.txt'), 'w') as f:
        json.dump(dict(vars(args)), f, indent=2)


    args.save_result_root = os.path.join(args.save_root, "results")
    args.save_inpinfo_root = os.path.join(args.save_root, "inp_info")
    os.makedirs(args.save_result_root, exist_ok=True)
    os.makedirs(args.save_inpinfo_root, exist_ok=True)
    return logger, func_time_saver


def convert2graph(paths, save_path=None, debug=False):
    if save_path is None:
        save_path = "debug{}.png"
    if debug:
        save_path = "z_" + save_path.split("/")[-1]
    node_name_temp = '${}_{{{}}}^{{{}}}$'

    node_type = ["IA", "Q", "K", "V", "KH", "VH", "A", "OA", "IM", "OM", "OB"]
    node_type_position = {
        "IA": (0, 3),
        "Q" : (1, 0),
        "K" : (1, 1),
        "V" : (1, 2),
        "KH": (2, 1),
        "VH": (2, 2),
        "A" : (3, 1),
        "OA": (4, 2),
        "IM": (5, 3),
        "OM": (6, 2),
        "OB": (7, 3)
    }
    xys = np.asarray([[v[0], v[1]] for k, v in node_type_position.items()])
    x_size = xys[:, 0].max()-xys[:, 0].min() +1
    y_size = xys[:, 1].max()-xys[:, 1].min() +1
    

    min_max_bidx = []
    min_max_tidx = []
    for path_idx, path in paths.items():
        [i[1][0].split("_") for i in path]

        buffer_bidx = []
        buffer_tidx = []
        for edge in path:
            if len(edge[1])!=1:
                print("There cannot exist multiple dst nodes!")
                import pdb; pdb.set_trace()
            
            srcs = edge[0]
            if srcs[0]!="ROOT":
                for src in srcs:
                    s_br, s_ti, s_bi = src.split("_")
                    buffer_bidx.append(s_bi)
                    buffer_tidx.append(s_ti)
            d_br, d_ti, d_bi = edge[1][0].split("_")
            buffer_bidx.append(d_bi)
            buffer_tidx.append(d_ti)
        plot_target_bidx = np.unique(buffer_bidx).astype(int)
        plot_target_tidx = np.unique(buffer_tidx).astype(int)
        
        min_max_bidx.append([plot_target_bidx.min(), plot_target_bidx.max()])
        min_max_tidx.append([plot_target_tidx.min(), plot_target_tidx.max()])
    min_max_bidx = np.asarray(min_max_bidx)
    min_max_tidx = np.asarray(min_max_tidx)

    min_bidx = min_max_bidx[:,0].min()
    max_bidx = min_max_bidx[:,1].max()

    min_tidx = min_max_tidx[:,0].min()
    max_tidx = min_max_tidx[:,1].max()

    total_node_pos = {}
    for path_idx, path in paths.items():
        node_pos = []
        total_node_positions = {}
        G = nx.DiGraph()
        for block_idx in range(min_bidx, max_bidx+1):
            for token_idx in range(min_tidx, max_tidx+1):
                for curr_n_type in node_type:
                    locals()["{}_{}_{}".format(curr_n_type, block_idx, token_idx)] = node_name_temp.format(curr_n_type, block_idx, token_idx)
                    G.add_node(locals()["{}_{}_{}".format(curr_n_type, block_idx, token_idx)])
                    base_position = node_type_position[curr_n_type]

                    rel_x_idx = block_idx - min_bidx
                    rel_y_idx = token_idx - min_tidx
                    curr_node_position = (
                        base_position[0] + rel_x_idx * x_size, 
                        base_position[1] + rel_y_idx * y_size)
                    total_node_positions[locals()["{}_{}_{}".format(curr_n_type, block_idx, token_idx)]] = curr_node_position
                    node_pos.append(curr_node_position)
        total_node_pos.update({path_idx: node_pos})
        for edge in path:
            srcs, dst = edge
            if len(dst)!=1:
                print("There cannot exist multiple dst nodes!")
                import pdb; pdb.set_trace()
            dst = dst[0]
            d_br, d_ti, d_bi = dst.split("_")
            dst_node = node_name_temp.format(d_br, d_bi, d_ti)
            
            if srcs[0] == "ROOT":
                continue
            for src in srcs:
                s_br, s_ti, s_bi = src.split("_")
                src_node = node_name_temp.format(s_br, s_bi, s_ti)
                src_dst = (src_node, dst_node)
                G.add_edge(*src_dst)
            
        plt.figure()
        nx.draw(G, 
                pos=total_node_positions, 
                with_labels=True, 
                labels={node: node for node in G.nodes()}, 
                node_size=800, node_color='lightblue', font_size=20, arrowsize=10)

        for token_idx in range(min_tidx, max_tidx+2): 
            rel_y_idx = token_idx - min_tidx
            plt.axhline(
                y=-0.5+y_size*rel_y_idx,
                xmin=-0.5, 
                xmax=x_size*(max_bidx+1-min_bidx)+0.5, 
                color='gray', 
                linestyle='-', 
                linewidth=1)
        for block_idx in range(min_bidx, max_bidx+2): 
            rel_x_idx = block_idx - min_bidx
            plt.axvline(
                x=-0.5+x_size*rel_x_idx, 
                ymin=-0.5, 
                ymax=y_size*(max_tidx+1-min_tidx)+0.5, 
                color='gray', 
                linestyle='-', 
                linewidth=1)
            
        for block_idx in range(min_bidx, max_bidx+1):
            for token_idx in range(min_tidx, max_tidx+1):
                rel_x_idx = block_idx - min_bidx
                rel_y_idx = token_idx - min_tidx
                plt.text(
                    rel_x_idx * x_size + 5, 
                    rel_y_idx * y_size, 
                    '{:2d}th Block {:2d}th Token'.format(block_idx, token_idx), fontweight='bold', fontsize=20, ha="center")

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(-1, x_size*(max_bidx+1-min_bidx)+1)
        plt.ylim(-1, y_size*(max_tidx+1-min_tidx)+0.5)

        fig = plt.gcf() 
        figsize = fig.get_size_inches()
        scale_factor = max((max_tidx+1-min_tidx), (max_bidx+1-min_bidx))
        new_figsize = figsize * scale_factor
        new_figsize[0] = new_figsize[0] * 1.25
        fig.set_size_inches(new_figsize)

        if debug:
            if os.path.isfile(save_path.format(path_idx)):
                os.remove(save_path.format(path_idx))
        plt.savefig(save_path.format(path_idx))
        plt.close()

def exclude_subsets(candidate, exclude_target, logger=None):
    if len(exclude_target)==0:
        return candidate
    new_candidate = []
    for cand in candidate:
        exclude_flag = False
        for ex in exclude_target:
            # ex is always smaller than cand, if the search proceeds with small number of steps.
            if len(set(ex).union(cand))==len(cand):
                # if logger is not None:
                    # logger.info(f"[exclude_subsets] Excluding candidate {cand} due to overlap with {ex}")
                exclude_flag = True
                break
        if exclude_flag is False:
            new_candidate.append(cand)
    return new_candidate

def get_data(dataset="imagenet", root=None):
    """
    Load ImageNet validation dataset
    """
    # ImageNet only. Also called inside CausalFlowTracer.trace_corrupted, so the root is taken from DATA_ROOT.
    if root is None:
        root = DATA_ROOT[dataset]
    dataset = ImageNetDataset(root=root, split='val')
    return dataset, dataset.num_classes

def get_model(model_name="vit_base_patch16_224", num_classes=1000, dataset_name="imagenet"):
    import timm
    vit_model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        cache_dir=str(MODEL_DIR),
    )
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = vit_model.to(device)

    # Add _split_heads method
    if model_name == "pit_ti_224":
        for i in range(len(vit_model.transformers)):
            for block in vit_model.transformers[i].blocks:
                block.attn._split_heads = types.MethodType(_split_heads, block.attn)
                block.attn._merge_heads = types.MethodType(_merge_heads, block.attn)
                block.attn._attn = types.MethodType(_attn, block.attn)
    else:
        for block in vit_model.blocks:
            block.attn._split_heads = types.MethodType(_split_heads, block.attn)
            block.attn._merge_heads = types.MethodType(_merge_heads, block.attn)
            block.attn._attn = types.MethodType(_attn, block.attn)
    
    # Wrap the model to handle tuple inputs
    return ModelWrapper(vit_model)


def get_noise_level(mt, fpath_nlv=os.path.join(SAMPLE_DIR, "vit_noise_level.pt")):
    """Calculate noise level based on patch embedding standard deviation"""
    if os.path.isfile(fpath_nlv) is False:
        # Create a small batch of random images
        batch_size = 100
        random_images = torch.randn(batch_size, 3, 224, 224).cuda()
        
        # Get patch embeddings
        with Trace(mt.model, "model.patch_embed") as t:
            mt.model(pixel_values=random_images)
            patch_embeddings = t.output
        
        # Calculate standard deviation across all dimensions
        noise_level = patch_embeddings.std().item()
        
        # Save the computed noise level
        torch.save(noise_level, fpath_nlv)
    else:
        noise_level = torch.load(fpath_nlv)
    
    return noise_level

def collect_embedding_std(mt, subjects):
    alldata = []
    for s in tqdm.tqdm(subjects, desc="collecting_std.."):
        inp = make_inputs(mt.tokenizer, [s])
        with Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level

def layername(model, num, kind=None, check_subl_inp=False, verf_type="all"):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        
        name = f'transformer.h.{num}'
        if verf_type=="all":
            if check_subl_inp:
                if kind == "mlp":
                    ln_name = "ln_2"
                elif kind == "attn":
                    ln_name = "ln_1"
                else:
                    import pdb; pdb.set_trace()
                name += "" if kind is None else "." + ln_name
            else:
                name += "" if kind is None else "." + kind
        else:
            if check_subl_inp:
                print("Check!! -> check_subl_inp option cannot operate!!")
            if verf_type=="attn":
                if "attn_" in kind:
                    name+=".ln_1"
                else:
                    name += "" if kind is None else "." + kind
            else:
                import pdb; pdb.set_trace()
        return name
    if hasattr(model, "gpt_neox"):
        if verf_type != "all":
            import pdb; pdb.set_trace()
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_path_set_idx(path_set, B):
    connected_idx = []
    for set_idx, set_edges in path_set.items():
        flag = False
        for set_edge in set_edges:
            if set_edge[0] == B.tolist():
                flag = True 
                break
        if flag:
            connected_idx.append(set_idx)

    return connected_idx

def _split_heads(self, tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

def _merge_heads(self, tensor, num_heads, attn_head_size):
    """
    Merges attn_head_size dim and num_attn_heads dim into hidden_size
    """
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)

def _attn(self, query, key, value, attention_mask, head_mask):
    """
    Perform attention operation with query, key, and value tensors
    """
    # Get the number of heads and head dimension

    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    attn_weights = attn_weights / torch.full([], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device )

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights

def predict_from_input(model, inp, multipred=False, end_symbol=[], mt=None, force_idx=None, use_mean=False):
    # multipred option makes outputs until resulting one sentence output
    # Don't reduce the batch size...: https://discuss.pytorch.org/t/why-is-the-output-of-a-linear-layer-different-when-the-batch-size-is-1/93515
    if multipred is False:
        # import pdb; pdb.set_trace()
        out = model(**inp)["logits"]
        if use_mean:
            probs = torch.softmax(out, dim=1).mean(dim=0).unsqueeze(0)
        else:
            probs = torch.softmax(out, dim=1)
        if force_idx is None:
            p, preds = torch.max(probs, dim=1)

        else:
            p = probs[:, force_idx]
            preds = force_idx.repeat(p.shape[0])
        return preds.unsqueeze(0), p.unsqueeze(0), out
    else:
        iter_inp = copy.deepcopy(inp)
        # import pdb; pdb.set_trace()
        iter_p = []
        iter_preds = []

        while 1:
            out = model(**iter_inp)["logits"]
            probs = torch.softmax(out[:, -1], dim=1)
            if use_mean:
                import pdb; pdb.set_trace()

            if force_idx is None:
                p, preds = torch.max(probs, dim=1)
            else:
                import pdb; pdb.set_trace()

            iter_p.append(p)
            iter_preds.append(preds)
            next_token = preds.unsqueeze(-1)
            buffer_ids = torch.cat((iter_inp["input_ids"], next_token), dim=1)

            if (torch.unique(iter_inp["attention_mask"]).shape[0]==1) is False:
                import pdb; pdb.set_trace()
            buffer_mask = torch.cat((iter_inp["attention_mask"], iter_inp["attention_mask"][:, -1].unsqueeze(-1)), dim=-1)
            iter_inp = {"input_ids":buffer_ids, "attention_mask":buffer_mask}

            if torch.unique(next_token).shape[0] != 1:
                import pdb; pdb.set_trace()
            currsym = mt.tokenizer.decode(next_token[0], skip_special_tokens=True, clean_up_tokenization=True)
            if currsym in end_symbol:
                
                break
        iter_preds = torch.vstack(iter_preds)
        iter_p = torch.vstack(iter_p)
        print("ToDo: check!")
        import pdb; pdb.set_trace()
        return iter_preds, iter_p
    

def predict_from_normal_and_noise_input(
        images, flow_tracer, labels, logger,
        mt, n_lev, model_name, dataset_name, num_noise_sample=3, num_normal_sample=3, noise_type="other",
        end_symbol=None, out_num=1):
    
    # Create input dictionary for the model
    expanded_images = images.repeat(num_normal_sample, 1, 1, 1)
    normal_inp = {"pixel_values": expanded_images}
    
    if out_num==1:
        # Run the model and capture intermediate representations
        answer_t = flow_tracer.trace_normal(mt.model, normal_inp, model_name)
        # Fix: answer_t is a 1D tensor with shape [1], so we need to handle it differently
        if answer_t.dim() == 1:
            answer = [answer_t.item()]
        else:
            answer = answer_t.argmax(dim=1).tolist()

        if labels[0].item() not in answer:
            return -1, -1, -1, -1, -1
            
        noise_rand_seed = 0
        # Find a noise that changes the output
        while 1:
            corrupted_answer_t = flow_tracer.trace_corrupted(
                mt.model, None, images, noise_level=n_lev, rand_seed=noise_rand_seed, model_name=model_name,
                dataset_name=dataset_name, noise_type=noise_type, num_noise_sample=num_noise_sample)
            # Fix: Handle corrupted_answer_t similarly - it's now a 1D tensor with shape [1]
            # print(answer_t.item(),corrupted_answer_t)

            if answer_t.item() not in corrupted_answer_t:
                break
            noise_rand_seed += 1
            logging.info("[Noise Finder] Changing the seed to find noise that changes the output... Current Seed:{}, Ans:{}".format(
                noise_rand_seed, answer[0]))

    # For ViT, token_num is the number of patches
    curr_total_token_num = (images.shape[2] // mt.model.model.patch_embed.patch_size[0]) * (images.shape[3] // mt.model.model.patch_embed.patch_size[1])
    if model_name == "pit_ti_224":
        curr_total_block_num = len(mt.model.model.transformers[0].blocks) + len(mt.model.model.transformers[1].blocks) + len(mt.model.model.transformers[2].blocks)
    else:
        curr_total_block_num = len(mt.model.model.blocks)

    return flow_tracer, answer, curr_total_token_num, curr_total_block_num, normal_inp


def load_saver(saver, args, curr_meta_arg, except_list=["save_result_root"]):
    if args.load_path_saver is None:
        print("[Not Loaded] Please specify the path!")
    print("[SAVER Load] {}".format(args.load_path_saver))

    load_list = saver.parse_load_saver_file(args)
    
    f = open(os.path.join(args.load_path_saver, load_list[-1]), "rb")
    loaded_data = dill.load(f)

    loaded_args_dict = vars(loaded_data["args"])
    args_dict = vars(args)
    print("\t args is loaded")
    saver.dict_equal_checker(args_dict, loaded_args_dict, warning_template="\t\t {} is different!")

    print("\t curr_meta_arg is loaded")
    saver.dict_equal_checker(curr_meta_arg, loaded_data["curr_meta_arg"])
    curr_meta_arg = loaded_data["curr_meta_arg"]

    print("\t saver is loaded")
    saver.dict_equal_checker(saver.__dict__, loaded_data["saver"])
    saver.set_values(loaded_data["saver"])
    return curr_meta_arg

# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda", pass_encoding=False):
    if pass_encoding is False:
        token_lists = [tokenizer.encode(p) for p in prompts]
    else:
        token_lists = prompts
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


class function_time_saver(object):
    def __init__(self):
        self.function_timer = {}
    def save(self, func_name, val_dict):
        if func_name not in self.function_timer:
            self.function_timer[func_name] = {
                "total_time":0,
                "cpu_time":0,
                "gpu_time":0,
                "gpu_mem_u":0,
                "gpu_mem_r":0,
                "func_file_path":val_dict["func_file_path"]
            }
        self.function_timer[func_name]["total_time"] += val_dict["total_time"]
        self.function_timer[func_name]["cpu_time"] += val_dict["cpu_time"]
        self.function_timer[func_name]["gpu_time"] += val_dict["gpu_time"]
        self.function_timer[func_name]["gpu_mem_u"] += val_dict["gpu_mem_u"]
        self.function_timer[func_name]["gpu_mem_r"] += val_dict["gpu_mem_r"]

    def logging(self, logger, header=None):
        keys = list(self.function_timer.keys())
        time_list = [self.function_timer[k]["total_time"] for k in keys]
        sorted_index = np.argsort(time_list)
        sorted_key = np.asarray(keys)[sorted_index]

        if header is not None:
            logger.info(header)
        for k in sorted_key:
            log_message = "\t[Function Log] '{}' => Total time: {:.4f}s (CPU time: {:.4f}s, GPU time: {:.4f}s) | GPU mem: (U) {:.2f}MB (R) {:.2f}MB\n\t\tFunction Path: '{}'".format(
                k,
                self.function_timer[k]["total_time"],
                self.function_timer[k]["cpu_time"],
                self.function_timer[k]["gpu_time"],
                self.function_timer[k]["gpu_mem_u"],
                self.function_timer[k]["gpu_mem_r"],
                self.function_timer[k]["func_file_path"],
            )
            logger.info(log_message)
        logger.info("==============================\n\n")


def measure_time_and_memory(logger=None, func_time_saver=None):
    def decorator(func):
        if getattr(func, '__is_decorated__', False):
            return func
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Measure CPU start time
            cpu_start_time = time.time()

            # Measure GPU start time (if CUDA is available)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all CUDA kernels to finish
                gpu_start_time = torch.cuda.Event(enable_timing=True)
                gpu_end_time = torch.cuda.Event(enable_timing=True)
                gpu_start_time.record()
                gpu_start_abs_time = time.time()

                initial_memory = torch.cuda.memory_allocated()
                initial_reserved = torch.cuda.memory_reserved()

            result = func(*args, **kwargs)

            # Measure CPU end time
            cpu_end_time = time.time()
            cpu_elapsed_time = cpu_end_time - cpu_start_time

            # Measure GPU end time (if CUDA is available)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all CUDA kernels to finish
                gpu_end_time.record()
                torch.cuda.synchronize()  # Wait for the events to be recorded
                gpu_end_abs_time = time.time()
                gpu_elapsed_time = gpu_start_time.elapsed_time(gpu_end_time) / 1000.0  # Convert milliseconds to seconds
                final_memory = torch.cuda.memory_allocated()
                final_reserved = torch.cuda.memory_reserved()
                memory_usage = (final_memory - initial_memory) / (1024 * 1024) 
                memory_reserved = (final_reserved - initial_reserved) / (1024 * 1024) 
            else:
                gpu_elapsed_time = None

         
            if gpu_start_abs_time is not None:
                total_start = min(cpu_start_time, gpu_start_abs_time) 
                total_end = max(cpu_end_time, gpu_end_abs_time)
            else:
                total_start = cpu_start_time
                total_end = cpu_end_time
                gpu_elapsed_time = 0.0
                memory_usage = 0.0
                memory_reserved = 0.0
            total_elapsed_time = total_end - total_start

            # Get the absolute path of the function's module
            func_module = sys.modules[func.__module__]
            func_file_path = os.path.abspath(func_module.__file__)

            val_dict = {
                "total_time":total_elapsed_time,
                "cpu_time":cpu_elapsed_time,
                "gpu_time":gpu_elapsed_time,
                "gpu_mem_u":memory_usage,
                "gpu_mem_r":memory_reserved,
                "func_file_path":func_file_path
            }
            func_time_saver.save(func.__name__, val_dict)
            # if logger:
            #     logger.info(log_message)
            # else:
            #     print(log_message)

            return result

        wrapper.__is_decorated__ = True
        return wrapper
    return decorator

# class CausalFlowTracer(object):

#     def __init__(self):
#         self.traced_paths = {}
    
#     def init_node(self):
#         self.in_term1 = None
#         self.in_term2 = None
#         self.in_term3_list = None
#         self.in_term4_list = None

#     def trace_normal(
#             self, model, inp):
#         """
#         Trace normal inputs through the model.
#         For ViT, we capture the outputs of each transformer block.
#         """
#         self.feats_normal = None
#         self.feats_k_normal = None
#         self.feats_v_normal = None
#         self.scores_normal = None

#         outputs = {"output": [], "output_k": [], "output_v": []}

#         def hook_fn(module, input, output):
#             # For ViT, output is a tuple (hidden_states, attention_outputs)
#             # attention_outputs contains (attn_output, attn_weights)
#             outputs["output"].append(output[0].detach())
#             if isinstance(output[1], tuple):
#                 outputs["output_k"].append(output[1][1].detach())  # attn_weights
#                 outputs["output_v"].append(output[1][0].detach())  # attn_output
#             else:
#                 # If no attention outputs, use None
#                 outputs["output_k"].append(None)
#                 outputs["output_v"].append(None)

#         # Register the hook for each transformer block
#         for i, block in enumerate(model.model.blocks):
#             block.register_forward_hook(hook_fn)

#         # Forward pass through the model
#         out = model(**inp)

#         # Remove the hooks
#         for block in model.model.blocks:
#             block.remove_hook(hook_fn)

#         # Store the captured outputs
#         self.feats_normal = outputs["output"]
#         self.feats_k_normal = outputs["output_k"]
#         self.feats_v_normal = outputs["output_v"]

#         return out


################################################################################
# causal_flow_tracer (CausalPathTracing_anonymous/CPT/vision/lib/causal_flow_tracer.py)
################################################################################
import copy
import torch
import random
from torch import nn
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple, Union
from torch.utils.data import DataLoader

import torch.nn.functional as F

import types

def _split_heads(self, tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


class CausalFlowTracer(object):

    def __init__(self):
        self.traced_paths = {}
    
    def init_node(self):
        self.in_term1 = None
        self.in_term2 = None
        self.in_term3_list = None
        self.in_term4_list = None

    def trace_normal(
            self, model, inp, model_name):
        """
        Trace normal inputs through the model.
        For ViT, we capture the outputs of each transformer block.
        """
        self.feats_normal = None
        self.feats_k_normal = None
        self.feats_v_normal = None
        self.scores_normal = None

        outputs = {"output": [], "output_k": [], "output_v": []}

        def hook_fn(module, input, output):
            outputs["output"].append(output.detach())
            outputs["output_k"].append([1])
            outputs["output_v"].append([1])

        forward_hooks = []
        # Use model.model.blocks for ViT
        if model_name == "pit_ti_224":
            for i in range(len(model.model.transformers)):
                for block in model.model.transformers[i].blocks:
                    forward_hook = block.register_forward_hook(hook_fn)
                    forward_hooks.append(forward_hook)
        else:
            for i, block in enumerate(model.model.blocks):
                forward_hook = block.register_forward_hook(hook_fn)
                forward_hooks.append(forward_hook)
        with torch.no_grad():
            outputs_exp = model(**inp)

            # Handle potential tuple output
            if isinstance(outputs_exp, tuple):
                outputs_exp = outputs_exp[0]
            
            # For ViT, outputs_exp is a tensor, not a dictionary with 'logits'
            if isinstance(outputs_exp, dict) and 'logits' in outputs_exp:
                logits = outputs_exp['logits']
            else:
                # If outputs_exp is a tensor, use it directly
                logits = outputs_exp

        for forward_hook in forward_hooks:
            forward_hook.remove()

        # For ViT, logits are directly the output
        scores_normal = torch.softmax(logits, dim=1)[0] ##softmax(logit)
        answer_t = torch.max(scores_normal, dim=0).indices.unsqueeze(0) #argmax(softmax(logit))

        all_traced_normal = outputs["output"]
        all_traced_k_normal = outputs["output_k"]
        all_traced_v_normal = outputs["output_v"]

        # torch.clone makes different outputs -> please use 
        self.feats_normal = copy.deepcopy(all_traced_normal)
        self.feats_k_normal = copy.deepcopy(all_traced_k_normal)
        self.feats_v_normal = copy.deepcopy(all_traced_v_normal)
        self.scores_normal = copy.deepcopy(scores_normal)

        return answer_t


    def trace_corrupted(
            self, model, tokenizer, images, rand_seed, model_name, dataset_name, noise_type="other", num_noise_sample=3, noise_level=0.5):
        """
        Trace corrupted inputs through the model.
        For ViT, we replace random patches with patches from other images.
        Args:
            noise_level: Float between 0 and 1, fraction of patches to corrupt
        """
        self.feats_corrupted = None
        self.feats_k_corrupted = None
        self.feats_v_corrupted = None
        self.scores_corrupted = None

        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)
        random.seed(rand_seed)

        dataset, _ = get_data(dataset_name)
        dataloader_for_corrupt = DataLoader(dataset, batch_size=num_noise_sample, shuffle=True)

        # Get a batch of random images from the loader
        corrupted_images = None
        for batch_images, _ in dataloader_for_corrupt:
            # Create corrupted images by replacing random patches
            patch_size = 16
            B, C, H, W = batch_images.shape
            num_patches_h = H // patch_size
            num_patches_w = W // patch_size
            total_patches = num_patches_h * num_patches_w
            
            # Extract all patches from all images
            patches = batch_images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)  # [B, C, num_patches, 16, 16]
            
            # Create random indices for shuffling
            shuffle_indices = torch.randperm(B * total_patches) 
            
            # Create new images with shuffled patches
            corrupted_images = torch.zeros_like(batch_images)
            for i in range(B):
                for p in range(total_patches):
                    # Calculate source batch and patch indices
                    shuffle_idx = shuffle_indices[i * total_patches + p]
                    src_batch_idx = shuffle_idx // total_patches
                    src_patch_idx = shuffle_idx % total_patches
                    
                    # Calculate patch positions
                    h_idx = (p // num_patches_w) * patch_size
                    w_idx = (p % num_patches_w) * patch_size
                    
                    # Place shuffled patch in new position
                    corrupted_images[i, :, h_idx:h_idx+patch_size, w_idx:w_idx+patch_size] = \
                        patches[src_batch_idx, :, src_patch_idx]
            
            corrupted_images = corrupted_images.cuda()
            break  # Just take the first batch
        if corrupted_images is None:
            raise ValueError("Could not load images from the loader")

        inputs_block = {"input": []}
        outputs = {"output": [], "output_k": [], "output_v": []}

        def make_hook_fn(block_idx):
            def hook_fn(module, input, output):
                if block_idx==0:
                    inputs_block["input"].append(input[0].detach())
                outputs["output"].append(output.detach())
                outputs["output_k"].append([1])
                outputs["output_v"].append([1])
            return hook_fn

        forward_hooks = []
        # Use model.model.blocks for ViT
        if model_name == "pit_ti_224":
            for i in range(len(model.model.transformers)):
                for j, block in enumerate(model.model.transformers[i].blocks):
                    forward_hook = block.register_forward_hook(make_hook_fn(j))
                    forward_hooks.append(forward_hook)
        else:
            for i, block in enumerate(model.model.blocks):
                forward_hook = block.register_forward_hook(make_hook_fn(i))
                forward_hooks.append(forward_hook)

        with torch.no_grad():
            outputs_exp = model(**{"pixel_values": corrupted_images})
            
             # Handle potential tuple output
            if isinstance(outputs_exp, tuple):
                outputs_exp = outputs_exp[0]

            # For ViT, outputs_exp is a tensor, not a dictionary with 'logits'
            if isinstance(outputs_exp, dict) and 'logits' in outputs_exp:
                logits = outputs_exp['logits']
            else:
                # If outputs_exp is a tensor, use it directly
                logits = outputs_exp

        for forward_hook in forward_hooks:
            forward_hook.remove()

        # For ViT, logits are directly the output
        probs_corrupted = torch.softmax(logits, dim=1).mean(dim=0).unsqueeze(0)
        answer_t = torch.max(probs_corrupted, dim=0).indices #.unsqueeze(0)

        all_traced_corrupted = outputs["output"]
        all_traced_k_corrupted = outputs["output_k"]
        all_traced_v_corrupted = outputs["output_v"]

        self.feats_corrupted_init = copy.deepcopy(inputs_block["input"][0])
        self.feats_corrupted = copy.deepcopy(all_traced_corrupted)
        self.feats_k_corrupted = copy.deepcopy(all_traced_k_corrupted)
        self.feats_v_corrupted = copy.deepcopy(all_traced_v_corrupted)
        self.scores_corrupted = copy.deepcopy(probs_corrupted)

        return answer_t

def custom_sort_indices(arr, order):
    priority = {string: i for i, string in enumerate(order)}
    sorted_indices = sorted(range(len(arr)), key=lambda x: priority[arr[x]])
    
    return sorted_indices
def converter(path, quiet=False, node_seq=["OB", "OM", "IM", "OA", "A", "VH", "KH", "Q", "K", "V", "IA"]):
    path = np.unique(path, axis=0)
    block_idx = []
    for edge in path:
        block_idx.append(int(edge[0][2]))
    sorted_idx = np.argsort(block_idx)[::-1]
    sorted_path = np.asarray(path)[sorted_idx]

    buffer = []
    for b_idx in np.unique(block_idx)[::-1]:
        w_idx = np.where(sorted_path[:, 0, 2]==str(b_idx))[0]
        data = sorted_path[w_idx]

        curr_token_list = data[:, 0, 1].astype(int)
        t_sorted_idx = np.argsort(curr_token_list)
        t_sorted_data = data[t_sorted_idx]
        t_sorted_list = curr_token_list[t_sorted_idx]

        for c_t_i in np.unique(t_sorted_list):
            t_idx = np.where(t_sorted_list==c_t_i)[0]
            curr_buffer = t_sorted_data[t_idx]
            s_idx = custom_sort_indices(curr_buffer[:, 0, 0], node_seq)
            curr_buffer = curr_buffer[s_idx].tolist()
            
            buffer.extend(curr_buffer)

    lines = []
    for edge in buffer:
        line = "{:2s}.{:2s}.{:2s} -> {:2s}.{:2s}.{:2s}".format(edge[0][0], edge[0][1], edge[0][2], edge[1][0], edge[1][1], edge[1][2])
        lines.append(line)
        if quiet is False:
            print(line)
  
    return lines 


def get_lm_inp_mu_std(x, eps):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    std = torch.sqrt(var + eps)
    return mu, std

def make_layernorm2mm(x, layernorm_layer, prefix_mu=None, prefix_std=None):
    # x.shape == [batch, token, hidden]
    if (prefix_mu is None) | (prefix_std is None):
        mu, std = get_lm_inp_mu_std(x, eps=layernorm_layer.eps)
    else:
        mu = prefix_mu
        std = prefix_std


    w_ = layernorm_layer.weight / std
    b = (-layernorm_layer.weight * mu / std + layernorm_layer.bias)
    w = torch.diag_embed(w_)

    return w, b

def ln_mm(x, W, b=None):
    out = x.unsqueeze(-2) @ W.transpose(-1, -2)
    out = out.squeeze(-2)
    if b is not None:
        out = out + b 
    return out 

def make_homoneneous_coord(x):
    ones = torch.ones_like(x[..., :1])
    x_ext = torch.cat([x, ones], dim=-1)
    return x_ext

def homogen_mm(x, W):
    return (make_homoneneous_coord(x).unsqueeze(-2) @ W).squeeze(-2)

def layernorm2mm(x, layernorm_layer, prefix_mu=None, prefix_std=None, bias_divide=None):
    # x.shape == [batch, token, hidden]
    w, b = make_layernorm2mm(x, layernorm_layer, prefix_mu, prefix_std)
    out = x * w + b  # Broadcasting: [batch, token, hidden] * [hidden] + [hidden]
    return out

def linear2mm(x, linear_layer, bias_divide=None, ignore_bias=False, weight_slice=None):
    # x.shape == [batch, token, hidden]
    if weight_slice is not None:
        W = linear_layer.weight
    else:
        W = linear_layer.weight[:, weight_slice[0]:weight_slice[1]]
    b = linear_layer.bias

    if (b is None) or (ignore_bias):
        return torch.matmul(x, W)

    x_ext = make_homoneneous_coord(x)
    if bias_divide is None:
        A = torch.cat([W, b.unsqueeze(1)], dim=1)
    else:
        A = torch.cat([W, b.unsqueeze(1)/bias_divide], dim=1)

    out = torch.matmul(x_ext, A.T)

    return out

def subsetidx2pathnodeidx(subsetidx, num_heads):
    subsetidx_list = list(subsetidx)
    target_idx_term1 = []
    target_idx_term2 = []
    target_idx_term3 = []
    target_idx_term4 = []

    for idx in subsetidx_list:
        if (idx==0):
            target_idx_term1.append(idx)
        elif (idx==1):
            target_idx_term2.append(idx-1)
        elif (idx>=2) & (idx<(2+num_heads)):
            target_idx_term3.append(idx-2)
        elif (idx>=(2+num_heads)):
            target_idx_term4.append(idx-2-num_heads)

    other_idx_term1 = np.setdiff1d([0], target_idx_term1).tolist()
    other_idx_term2 = np.setdiff1d([0], target_idx_term2).tolist()
    other_idx_term3 = np.setdiff1d(np.arange(num_heads), target_idx_term3).tolist()
    other_idx_term4 = np.setdiff1d(np.arange(num_heads), target_idx_term4).tolist()

    return (target_idx_term1, target_idx_term2, target_idx_term3, target_idx_term4), (other_idx_term1, other_idx_term2, other_idx_term3, other_idx_term4)

def path_level_intervention_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:


    if hasattr(self, "anonymoususer_save_mode"):
        if self.anonymoususer_save_mode:
            if hasattr(self, "anonymoususer_feats_corrupted_init"):
                # import pdb; pdb.set_trace()
                hidden_states = copy.deepcopy(self.anonymoususer_feats_corrupted_init)
                # import pdb; pdb.set_trace()
    
    residual = hidden_states
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    #-------for Path Division-------#
    residual_in = copy.deepcopy(residual)
    #-------for Path Division-------#
          
    hidden_states = self.norm1(hidden_states)

    inp_ln_1 = copy.deepcopy(hidden_states)
    
    # print(attention_mask)
    attn_outputs, head_wise_attn_output = self.attn(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )

    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]
    # residual connection
    residual = residual[0]
    hidden_states = attn_output + residual

    if encoder_hidden_states is not None:
        # add one self-attention block for cross-attention
        if not hasattr(self, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                "cross-attention layers by setting `config.add_cross_attention=True`"
            )
        residual = hidden_states
        hidden_states = self.ln_cross_attn(hidden_states)
        cross_attn_outputs = self.crossattention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = cross_attn_outputs[0]
        # residual connection
        hidden_states = residual + attn_output
        outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights
    
    residual = copy.deepcopy(hidden_states.detach())
    #-------for Path Division-------#
    ln2_mu, ln2_std = get_lm_inp_mu_std(copy.deepcopy(hidden_states.detach()), self.norm2.eps)
    W_ln_mlp, b_ln_mlp = make_layernorm2mm([], self.norm2, prefix_mu=ln2_mu, prefix_std=ln2_std)
    #-------for Path Division-------#

    # torch.allclose(layernorm2mm(hidden_states, self.ln_2, prefix_mu=ln2_mu, prefix_std=ln2_std), self.ln_2(hidden_states), rtol=1e-5, atol=1e-6)

    hidden_states_ln2 = self.norm2(hidden_states)
    feed_forward_hidden_states, gelu_input_states = self.mlp(hidden_states_ln2)

    #-------for Path Division-------#
    D_a = torch.where(gelu_input_states.abs() > 1e-6, nn.functional.gelu(gelu_input_states) / gelu_input_states, torch.full_like(gelu_input_states, 0.5))
    #-------for Path Division-------#

    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    #-------for Path Division-------#
    org_output = copy.deepcopy(hidden_states)
    #-------for Path Division-------#

    #-------Path Division-------#
    # in_term1 = layernorm2mm(inp_ln_1, self.norm2, prefix_mu=ln2_mu, prefix_std=ln2_std)
    # in_term2 = self.mlp(inp_ln_1, custom_gelu=D_a)[0]

    in_term1 = residual_in[0]
    in_term2 = self.mlp(ln_mm(residual_in[0], W_ln_mlp, b=None), custom_gelu=D_a, bias_divide=None, ignore_bias=True)[0]

    in_term3_list, in_term4_list = [], []
    debug_attn_out_proj = []
    for h_i in range(self.attn.num_heads):
        # Note: nn.Linear -> x*W^T, but, Conv1d -> x*W
        h_attn_out_proj = nn.functional.linear(
            head_wise_attn_output[:, h_i], 
            weight=self.attn.proj.weight[:, self.attn.head_dim * (h_i):self.attn.head_dim  * (h_i+1)],
            bias=self.attn.proj.bias/self.attn.num_heads
            )
        
        # in_term3_i = layernorm2mm(h_attn_out_proj, self.norm2, prefix_mu=ln2_mu, prefix_std=ln2_std, bias_divide=self.attn.num_heads)
        in_term3_i = h_attn_out_proj
        in_term3_list.append(in_term3_i.unsqueeze(0))

        # in_term4_i = self.mlp(in_term3_i, custom_gelu=D_a, bias_divide=self.attn.num_heads)[0]
        in_term4_i = self.mlp(ln_mm(copy.deepcopy(h_attn_out_proj.detach()), W_ln_mlp, b=None), custom_gelu=D_a, bias_divide=None, ignore_bias=True)[0]
        in_term4_list.append(in_term4_i.unsqueeze(0))

        # Debug dimensions and adjust in_term3_i
        # print(f"Original in_term3_i shape: {in_term3_i.shape}")
        # print(f"fc1 weight shape: {self.mlp.fc1.weight.shape}")
        # print(f"D_a shape: {D_a.shape}")
        # print(f"D_a element: {torch.diag_embed(D_a).shape}")
        # print(f"fc2 weight shape: {self.mlp.fc2.weight.shape}")

        # in_term3_i.unsqueeze(2) @ self.mlp.fc1.weight.T @ torch.diag_embed(D_a) @ self.mlp.fc2.weight.T

    in_term3_list = torch.cat(in_term3_list, dim=0)
    in_term4_list = torch.cat(in_term4_list, dim=0)
    #-------Path Division-------#

    #-------Floating-point Error Correction-------#
    mlp_bias_term1 = ((b_ln_mlp @ self.mlp.fc1.weight.T) * D_a) @ self.mlp.fc2.weight.T
    mlp_bias_term2 = (self.mlp.fc1.bias * D_a) @ self.mlp.fc2.weight.T
    mlp_bias_term3 = self.mlp.fc2.bias
    mlp_bias_term_all = mlp_bias_term1 + mlp_bias_term2 + mlp_bias_term3 

    in_term2 += (mlp_bias_term_all/(self.attn.num_heads+1))
    in_term4_list += (mlp_bias_term_all/(self.attn.num_heads+1)).unsqueeze(0)

    divided_output = in_term1 + in_term2 + torch.sum(in_term3_list, dim=0) + torch.sum(in_term4_list, dim=0)
                    
    comp_err = divided_output-org_output
    per_comp_err = comp_err/(2*self.attn.num_heads + 2)
    #-------Floating-point Error Correction-------#

    if hasattr(self, "anonymoususer_save_mode"):
        if self.anonymoususer_save_mode:
            self.anonymoususer_flow_tracer.in_term1 = copy.deepcopy(in_term1)
            self.anonymoususer_flow_tracer.in_term2 = copy.deepcopy(in_term2)
            self.anonymoususer_flow_tracer.in_term3_list = copy.deepcopy(in_term3_list)
            self.anonymoususer_flow_tracer.in_term4_list = copy.deepcopy(in_term4_list)
            self.anonymoususer_flow_tracer.org_output = copy.deepcopy(org_output)

    if hasattr(self, "anonymoususer_trace_mode"):
        if self.anonymoususer_trace_mode:
            
            target_idx_term, other_idx_term = subsetidx2pathnodeidx(self.anonymoususer_curr_subset, self.attn.num_heads)
            if self.anonymoususer_cond in ["path", "contingency"]:
                
                if len(target_idx_term[0])==0: 
                    in_term1 = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term1)
                if len(target_idx_term[1])==0:
                    in_term2 = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term2)
                if len(other_idx_term[2])!=0:
                    in_term3_list[other_idx_term[2]] = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term3_list[other_idx_term[2]])
                if len(other_idx_term[3])!=0:
                    in_term4_list[other_idx_term[3]] = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term4_list[other_idx_term[3]])
            elif self.anonymoususer_cond in ["counterfactual"]:
                in_term1 = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term1)
                in_term2 = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term2)
                in_term3_list = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term3_list)
                in_term4_list = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term4_list)
            else:
                import pdb; pdb.set_trace()

    # Unfolded output
    hidden_states = in_term1 + in_term2 + torch.sum(in_term3_list, dim=0) + torch.sum(in_term4_list, dim=0)

    if hasattr(self, "anonymoususer_trace_mode"):
        if self.anonymoususer_trace_mode:

            if self.anonymoususer_cond in ["path", "contingency"]:
                other_idx_term_len = [len(i) for i in other_idx_term]
                # import pdb; pdb.set_trace()
                if sum(other_idx_term_len)==0:  # If the target subset is equal to the entire set, compensates for the error.
                    hidden_states = org_output
                    # import pdb; pdb.set_trace()
            elif self.anonymoususer_cond in ["counterfactual"]:
                hidden_states = copy.deepcopy(self.anonymoususer_corrupted_feats.org_output)
            else:
                import pdb; pdb.set_trace()


    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)



def custom_linear_forward(x, layer, bias_divide=1, ignore_bias=False):
    size_out = x.size()[:-1] + (layer.out_features,)
    # x = torch.addmm(layer.bias/bias_divide, x.view(-1, x.size(-1)), layer.weight)
    if ignore_bias:
        x_reshaped = x.view(-1, x.size(-1))
        x = x_reshaped @ layer.weight.T + torch.zeros_like(layer.bias)
    else:
        x_reshaped = x.view(-1, x.size(-1))
        x = x_reshaped @ layer.weight.T + layer.bias/bias_divide

    x = x.view(size_out)

    return x


def custom_mlp_forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], custom_gelu=None, bias_divide=None, ignore_bias=False) -> torch.FloatTensor:
    if (bias_divide==None) & (ignore_bias==False):
        hidden_states = self.fc1(hidden_states)
    else:
        hidden_states = custom_linear_forward(hidden_states, self.fc1, bias_divide=bias_divide, ignore_bias=ignore_bias)

    gelu_input_states = copy.deepcopy(hidden_states.detach())
    if custom_gelu is None:
        hidden_states = self.act(hidden_states)
    else:
        hidden_states = hidden_states * custom_gelu

    if (bias_divide==None) & (ignore_bias==False):
        hidden_states = self.fc2(hidden_states)
    else:
        hidden_states = custom_linear_forward(hidden_states, self.fc2, bias_divide=bias_divide, ignore_bias=ignore_bias)
    hidden_states = self.drop2(hidden_states)
    return hidden_states, gelu_input_states

def _split_heads(self, tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


# def custom_attn_forward(
#     self,
#     hidden_states: Optional[Tuple[torch.FloatTensor]],
#     layer_past: Optional[Tuple[torch.Tensor]] = None,
#     attention_mask: Optional[torch.FloatTensor] = None,
#     head_mask: Optional[torch.FloatTensor] = None,
#     encoder_hidden_states: Optional[torch.Tensor] = None,
#     encoder_attention_mask: Optional[torch.FloatTensor] = None,
#     use_cache: Optional[bool] = False,
#     output_attentions: Optional[bool] = False,
# ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
#     if encoder_hidden_states is not None:
#         if not hasattr(self, "q_attn"):
#             raise ValueError(
#                 "If class is used as cross attention, the weights `q_attn` have to be defined. "
#                 "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
#             )

#         query = self.q_attn(hidden_states)
#         key, value = self.qkv(encoder_hidden_states).split(self.split_size, dim=2)
#         attention_mask = encoder_attention_mask
#     else:
#         self.split_size = self.qkv.in_features
#         query, key, value = self.qkv(hidden_states).split(self.split_size, dim=2)
#     query = self._split_heads(query, self.num_heads, self.head_dim)
#     key = self._split_heads(key, self.num_heads, self.head_dim)
#     value = self._split_heads(value, self.num_heads, self.head_dim)

#     if layer_past is not None:
#         past_key, past_value = layer_past
#         key = torch.cat((past_key, key), dim=-2)
#         value = torch.cat((past_value, value), dim=-2)

#     if use_cache is True:
#         present = (key, value)
#     else:
#         present = None

#     # if self.reorder_and_upcast_attn:
#         # print("Not Supported!")
#         # import pdb; pdb.set_trace()
#         # attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
#     # else:
#     #### 아마도 _attn을 만들어야함. 어디서 가져오던지. 이건 sm(q*v)*v 연산임. 
#     attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    

#     head_wise_attn_output = copy.deepcopy(attn_output.detach())

#     attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
#     attn_output = self.proj(attn_output)
#     attn_output = self.proj_drop(attn_output)

#     outputs = (attn_output, present)
#     if output_attentions:
#         outputs += (attn_weights,)

#     return outputs, head_wise_attn_output # a, present, (attentions)

def custom_attn_forward(self, x, **kwargs) -> torch.Tensor:

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    head_wise_attn_output = copy.deepcopy(x)

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return (x,x,x), head_wise_attn_output

def org_attn_forward(self, x, **kwargs) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return (x,x,x), x

# def org_attn_forward(
#     self,
#     hidden_states: Optional[Tuple[torch.FloatTensor]],
#     layer_past: Optional[Tuple[torch.Tensor]] = None,
#     attention_mask: Optional[torch.FloatTensor] = None,
#     head_mask: Optional[torch.FloatTensor] = None,
#     encoder_hidden_states: Optional[torch.Tensor] = None,
#     encoder_attention_mask: Optional[torch.FloatTensor] = None,
#     use_cache: Optional[bool] = False,
#     output_attentions: Optional[bool] = False,
# ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
#     if encoder_hidden_states is not None:
#         if not hasattr(self, "q_attn"):
#             raise ValueError(
#                 "If class is used as cross attention, the weights `q_attn` have to be defined. "
#                 "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
#             )

#         query = self.q_attn(hidden_states)
#         key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
#         attention_mask = encoder_attention_mask
#     else:
#         query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
#     query = self._split_heads(query, self.num_heads, self.head_dim)
#     key = self._split_heads(key, self.num_heads, self.head_dim)
#     value = self._split_heads(value, self.num_heads, self.head_dim)

#     if layer_past is not None:
#         past_key, past_value = layer_past
#         key = torch.cat((past_key, key), dim=-2)
#         value = torch.cat((past_value, value), dim=-2)

#     if use_cache is True:
#         present = (key, value)
#     else:
#         present = None

#     if self.reorder_and_upcast_attn:
#         print("Not Supported!")
#         import pdb; pdb.set_trace()
#         attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
#     else:
#         attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

#     attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
#     attn_output = self.c_proj(attn_output)
#     attn_output = self.resid_dropout(attn_output)

#     outputs = (attn_output, present)
#     if output_attentions:
#         outputs += (attn_weights,)

#     return outputs  # a, present, (attentions)



def org_attn(self, query, key, value, *args, **kwargs) -> torch.Tensor:
    import pdb; pdb.set_trace()

    q = query * self.scale
    attn = query @ key.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn @ value

    return x, attn


    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # if self.scale_attn_weights:
    attn_weights = attn_weights / torch.full(
        [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
    )

    # if attention_mask is not None:
    #     # Apply the attention mask
    #     attn_weights = attn_weights + attention_mask

    # attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    # attn_weights = attn_weights.type(value.dtype)
    # attn_weights = self.attn_drop(attn_weights)

    # # Mask heads if we want to
    # if head_mask is not None:
    #     attn_weights = attn_weights * head_mask

    # attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights

# def org_mlp_forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
#     hidden_states = self.c_fc(hidden_states)
#     hidden_states = self.act(hidden_states)
#     hidden_states = self.c_proj(hidden_states)
#     hidden_states = self.dropout(hidden_states)
#     return hidden_states

def org_mlp_forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x

def org_block_forward(self, x: torch.Tensor) -> torch.Tensor:

    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))[1]))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


def org_pit_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    print("org_pit_block_forward input type:", type(x))
    if isinstance(x, tuple):
        print("org_pit_block_forward input tuple shapes:", [xx.shape if hasattr(xx, 'shape') else type(xx) for xx in x])
    else:
        print("org_pit_block_forward input shape:", x.shape)

    x, cls_tokens = x
    token_length = cls_tokens.shape[1]
    if self.pool is not None:
        x, cls_tokens = self.pool(x, cls_tokens)

    B, C, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = torch.cat((cls_tokens, x), dim=1)

    x = self.norm(x)
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    
    print(x.shape)
    cls_tokens = x[:, :token_length]
    x = x[:, token_length:]
    x = x.transpose(1, 2).reshape(B, C, H, W)

    return x, cls_tokens

# def org_block_forward(
#     self,
#     hidden_states: Optional[Tuple[torch.FloatTensor]],
#     layer_past: Optional[Tuple[torch.Tensor]] = None,
#     attention_mask: Optional[torch.FloatTensor] = None,
#     head_mask: Optional[torch.FloatTensor] = None,
#     encoder_hidden_states: Optional[torch.Tensor] = None,
#     encoder_attention_mask: Optional[torch.FloatTensor] = None,
#     use_cache: Optional[bool] = False,
#     output_attentions: Optional[bool] = False,
# ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
    
    
#     residual = hidden_states
#     hidden_states = self.norm1(hidden_states)
#     attn_outputs = self.attn(
#         hidden_states,
#         layer_past=layer_past,
#         attention_mask=attention_mask,
#         head_mask=head_mask,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#     )
#     attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
#     outputs = attn_outputs[1:]
#     # residual connection
#     hidden_states = attn_output + residual

#     if encoder_hidden_states is not None:
#         # add one self-attention block for cross-attention
#         if not hasattr(self, "crossattention"):
#             raise ValueError(
#                 f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
#                 "cross-attention layers by setting `config.add_cross_attention=True`"
#             )
#         residual = hidden_states
#         hidden_states = self.ln_cross_attn(hidden_states)
#         cross_attn_outputs = self.crossattention(
#             hidden_states,
#             attention_mask=attention_mask,
#             head_mask=head_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             output_attentions=output_attentions,
#         )
#         attn_output = cross_attn_outputs[0]
#         # residual connection
#         hidden_states = residual + attn_output
#         outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

#     residual = hidden_states
#     hidden_states = self.ln_2(hidden_states)

#     feed_forward_hidden_states = self.mlp(hidden_states)

#     # residual connection
#     hidden_states = residual + feed_forward_hidden_states

#     if use_cache:
#         outputs = (hidden_states,) + outputs
#     else:
#         outputs = (hidden_states,) + outputs[1:]

#     return outputs  # hidden_states, present, (attentions, cross_attentions)




def pass_block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
    
    if isinstance(hidden_states, tuple):
        # print(len(hidden_states))
        # print(hidden_states[0] == hidden_states[1], 'check!')
        B, _, _ = hidden_states[0].shape
    else:
        B, _, _ = hidden_states.shape
    attn_outputs = (copy.deepcopy(self.anonymoususer_feats_k_normal), copy.deepcopy(self.anonymoususer_feats_v_normal))
    outputs = (copy.deepcopy(self.anonymoususer_feats_normal), attn_outputs)

    return outputs  # hidden_states, present, (attentions, cross_attentions)

def pass_corrupted_block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
    if isinstance(hidden_states, tuple):
        B, _, _ = hidden_states[0].shape
    else:
        B, _, _ = hidden_states.shape
    attn_outputs = (copy.deepcopy(self.anonymoususer_feats_k_corrupted), copy.deepcopy(self.anonymoususer_feats_v_corrupted))
    outputs = (copy.deepcopy(self.anonymoususer_feats_corrupted), attn_outputs)

    return outputs  # hidden_states, present, (attentions, cross_attentions)


FUNCTION_MAP = {
    "deit": {
        "org_block_forward": org_block_forward,
        "org_attn_forward": org_attn_forward,
        "org_mlp_forward": org_mlp_forward,
        "intervention_forward": path_level_intervention_forward,
        "custom_attn_forward": custom_attn_forward,
        "custom_mlp_forward": custom_mlp_forward,
        "pass_block_forward": pass_block_forward,
        "pass_corrupted_block_forward": pass_corrupted_block_forward
        },
    "vit": {
        "org_block_forward": org_block_forward,
        "org_attn_forward": org_attn_forward,
        "org_mlp_forward": org_mlp_forward,
        "intervention_forward": path_level_intervention_forward,
        "custom_attn_forward": custom_attn_forward,
        "custom_mlp_forward": custom_mlp_forward,
        "pass_block_forward": pass_block_forward,
        "pass_corrupted_block_forward": pass_corrupted_block_forward
    },
    "pit": {
        "org_block_forward": org_pit_block_forward,
        "org_attn_forward": org_attn_forward,
        "org_mlp_forward": org_mlp_forward,
        "intervention_forward": path_level_intervention_forward,
        "custom_attn_forward": custom_attn_forward,
        "custom_mlp_forward": custom_mlp_forward,
        "pass_block_forward": pass_block_forward,
        "pass_corrupted_block_forward": pass_corrupted_block_forward
    },
}

def return_forward_method_dict(args):
    if "vit" in args.model:
        return FUNCTION_MAP["vit"]
    elif "deit" in args.model:
        return FUNCTION_MAP["deit"]
    elif "pit" in args.model:
        return FUNCTION_MAP["pit"]
    import pdb; pdb.set_trace()


################################################################################
# search_func (CausalPathTracing_anonymous/CPT/vision/lib/search_func.py)
################################################################################
import types
import time
import pickle
import itertools



def block_initializer(args, curr_block, symbolic_name="anonymoususer"):
    forward_method_dict = return_forward_method_dict(args)

    ### INITIALIZE
    # restore the original function
    curr_block.forward = types.MethodType(forward_method_dict["org_block_forward"], curr_block)
    curr_block.mlp.forward = types.MethodType(forward_method_dict["org_mlp_forward"], curr_block.mlp)
    curr_block.attn.forward = types.MethodType(forward_method_dict["org_attn_forward"], curr_block.attn)

    # remove previous used values
    dellist = [k for k in curr_block.__dict__.keys() if "anonymoususer" in k]
    _ = [ delattr(curr_block, k) for k in dellist]
    dellist = [k for k in curr_block.mlp.__dict__.keys() if "anonymoususer" in k]
    _ = [ delattr(curr_block.mlp, k) for k in dellist]
    dellist = [k for k in curr_block.attn.__dict__.keys() if "anonymoususer" in k]
    _ = [ delattr(curr_block.attn, k) for k in dellist]
    ### INITIALIZE
    return curr_block

def get_detailed_corrupted_flow_tracer_one_block(
        args, curr_meta_arg, flow_tracer, detailed_flow_tracer_idx):
    fwd_method_dict = return_forward_method_dict(args)
    # just for saving corrupted features
    detailed_flow_tracer = CausalFlowTracer()
    detailed_flow_tracer.init_node()

    # Noise RUN
    for bidx in range(curr_meta_arg["total_block_num"]):
        curr_block = curr_meta_arg["mt"].model.blocks[bidx]
        curr_block = block_initializer(args, curr_block)

        # change target function
        if bidx < detailed_flow_tracer_idx-1:
            curr_block.forward = types.MethodType(fwd_method_dict["pass_block_forward"], curr_block)
            curr_block.anonymoususer_feats_normal = flow_tracer.feats_normal[bidx]
            curr_block.anonymoususer_feats_k_normal = flow_tracer.feats_k_normal[bidx]
            curr_block.anonymoususer_feats_v_normal = flow_tracer.feats_v_normal[bidx]
        elif bidx == detailed_flow_tracer_idx-1:
            # prepare corrupted features
            curr_block.forward = types.MethodType(fwd_method_dict["pass_corrupted_block_forward"], curr_block)

            curr_block.anonymoususer_feats_corrupted = flow_tracer.feats_corrupted[bidx]
            curr_block.anonymoususer_feats_k_corrupted = flow_tracer.feats_k_corrupted[bidx]
            curr_block.anonymoususer_feats_v_corrupted = flow_tracer.feats_v_corrupted[bidx]
        elif bidx == detailed_flow_tracer_idx:
            if bidx == 0: 
                # Since the process of preparing the corrupted feature is missing when bidx==0, replace it with this line instead.
                curr_block.anonymoususer_feats_corrupted_init = flow_tracer.feats_corrupted_init

            # save corrupted features
            curr_block.forward = types.MethodType(fwd_method_dict["intervention_forward"], curr_block)
            curr_block.mlp.forward = types.MethodType(fwd_method_dict["custom_mlp_forward"], curr_block.mlp)
            curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)

            curr_block.anonymoususer_flow_tracer = detailed_flow_tracer
            curr_block.anonymoususer_save_mode = True
        else:
            curr_block.forward = types.MethodType(fwd_method_dict["pass_block_forward"], curr_block)

            curr_block.anonymoususer_feats_normal = flow_tracer.feats_normal[bidx]
            curr_block.anonymoususer_feats_k_normal = flow_tracer.feats_k_normal[bidx]
            curr_block.anonymoususer_feats_v_normal = flow_tracer.feats_v_normal[bidx]

    with torch.no_grad():
        _, _, _ = predict_from_input(curr_meta_arg["mt"].model, curr_meta_arg["inp"], multipred=(args.out_num!=1), end_symbol=args.end_symbol)

    # Restore orignal module
    for bidx in range(curr_meta_arg["total_block_num"]):
        curr_block = curr_meta_arg["mt"].model.blocks[bidx]
        curr_block = block_initializer(args, curr_block)

    return detailed_flow_tracer


def cause_edge_classifier(
    args, curr_meta_arg, flow_tracer
):  

    cond_bool = []
    raw_outs = []
    preds = []
    ps = []
    for cond in ["counterfactual", "contingency"]:
        if args.efficient_mode:
            if cond == "counterfactual":
                cond_bool.append(True)
                preds.append(copy.deepcopy(curr_meta_arg["normal_pred"]))
                ps.append(copy.deepcopy(curr_meta_arg["normal_p"]))
                continue

       
        for bidx in range(curr_meta_arg["total_block_num"]):
            curr_block = curr_meta_arg["mt"].model.blocks[bidx]
            curr_block = block_initializer(args, curr_block)

            fwd_method_dict = return_forward_method_dict(args)

            # change target function
            if bidx < curr_meta_arg["curr_block_idx"]: ###using it!
                curr_block.forward = types.MethodType(fwd_method_dict["pass_block_forward"], curr_block)
                curr_block.anonymoususer_feats_normal = flow_tracer.feats_normal[bidx]
                curr_block.anonymoususer_feats_k_normal = flow_tracer.feats_k_normal[bidx]
                curr_block.anonymoususer_feats_v_normal = flow_tracer.feats_v_normal[bidx]
            elif bidx == curr_meta_arg["curr_block_idx"]:
                curr_block.forward = types.MethodType(fwd_method_dict["intervention_forward"], curr_block)
                curr_block.mlp.forward = types.MethodType(fwd_method_dict["custom_mlp_forward"], curr_block.mlp)
                curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)

                curr_block.anonymoususer_curr_block_idx = curr_meta_arg["curr_block_idx"] 
                curr_block.anonymoususer_normal_feats = flow_tracer.feats_normal
                curr_block.anonymoususer_trace_mode = True
                curr_block.anonymoususer_cond = cond
                curr_block.anonymoususer_corrupted_feats = curr_meta_arg["detailed_flow_tracer"][bidx]["corrupted"]
                curr_block.anonymoususer_curr_subset = curr_meta_arg["curr_subset"]

            elif bidx > curr_meta_arg["curr_block_idx"]: #### for subset unique-path search, forard ct and make condition in here. 
                curr_block.forward = types.MethodType(fwd_method_dict["intervention_forward"], curr_block)
                curr_block.mlp.forward = types.MethodType(fwd_method_dict["custom_mlp_forward"], curr_block.mlp)
                curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)

                curr_block.anonymoususer_curr_block_idx = curr_meta_arg["curr_block_idx"] 
                curr_block.anonymoususer_trace_mode = True
                curr_block.anonymoususer_cond = "path"
                curr_block.anonymoususer_corrupted_feats = curr_meta_arg["detailed_flow_tracer"][bidx]["corrupted"]
                curr_block.anonymoususer_curr_subset = tuple(set().union(*flow_tracer.traced_paths[bidx]))
              
        with torch.no_grad():
            curr_pred, curr_p, raw_out = predict_from_input(
                curr_meta_arg["mt"].model, curr_meta_arg["inp"], multipred=(args.out_num!=1), end_symbol=args.end_symbol, use_mean=True)

        preds.append(copy.deepcopy(curr_pred))
        ps.append(copy.deepcopy(curr_p))
        # raw_outs.append(raw_out)
        curr_pred_unq = torch.unique(curr_pred)
        change_bool = curr_pred_unq==curr_meta_arg["normal_pred"]

        if (change_bool.sum()!=change_bool.shape[0]).item():
            if cond == "counterfactual":
                cond_bool.append(True)
            elif cond == "contingency":
                cond_bool.append(False)
        else:
            if cond == "counterfactual":
                cond_bool.append(False)
            elif cond == "contingency":
                cond_bool.append(True)
    cond_bool = torch.tensor(cond_bool)

    # print(f"cond_bool: {cond_bool}")
    actual_cause_bool = (cond_bool==True).all().item()
    if args.slightly_quiet_mode:
        if actual_cause_bool:
            log_line = "\t\t ({})[Subset: {}] | org:{}({}) path:CF-{}({}), CT-{}({})".format(
                "Causal" if actual_cause_bool else "Non-causal",
                curr_meta_arg["curr_subset"],
                curr_meta_arg["normal_pred"].item(), 
                round(curr_meta_arg["normal_p"].item(), 5), 
                preds[0].item() if args.efficient_mode is False else "",
                round(ps[0].item(), 5)if args.efficient_mode is False else "PASS",
                preds[1].item(), 
                round(ps[1].item(), 5),
            )
            curr_meta_arg["logger"].info(log_line)
    else:
        log_line = "\t\t ({})[Subset: {}] | org:{}({}) path:CF-{}({}), CT-{}({})".format(
            "Causal" if actual_cause_bool else "Non-causal",
            curr_meta_arg["curr_subset"],
            curr_meta_arg["normal_pred"].item(), 
            round(curr_meta_arg["normal_p"].item(), 5), 
            preds[0].item() if args.efficient_mode is False else "",
            round(ps[0].item(), 5)if args.efficient_mode is False else "PASS",
            preds[1].item(), 
            round(ps[1].item(), 5),
        )
        curr_meta_arg["logger"].info(log_line)


    decision_save = {
        "CF":  [str(preds[0].item()), str(ps[0].item())] if args.efficient_mode is False else ["None", "None"],
        "CT":  [str(preds[1].item()), str(ps[1].item())]
    }
    
    return actual_cause_bool, cond_bool, decision_save

def cause_edge_classifier_for_evaluation(
    args, curr_meta_arg, flow_tracer
):  

    cond_bool = []
    raw_outs = []
    preds = []
    ps = []
    for cond in ["counterfactual", "contingency"]:
        if args.efficient_mode:
            if cond == "counterfactual":
                cond_bool.append(True)
                preds.append(copy.deepcopy(curr_meta_arg["normal_pred"]))
                ps.append(copy.deepcopy(curr_meta_arg["normal_p"]))
                continue

       
        for bidx in range(curr_meta_arg["total_block_num"]):
            curr_block = curr_meta_arg["mt"].model.blocks[bidx]
            curr_block = block_initializer(args, curr_block)

            fwd_method_dict = return_forward_method_dict(args)

            curr_block.forward = types.MethodType(fwd_method_dict["intervention_forward"], curr_block)
            curr_block.mlp.forward = types.MethodType(fwd_method_dict["custom_mlp_forward"], curr_block.mlp)
            curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)

            curr_block.anonymoususer_curr_block_idx = curr_meta_arg["curr_block_idx"] 
            curr_block.anonymoususer_trace_mode = True
            curr_block.anonymoususer_cond = "path"
            curr_block.anonymoususer_corrupted_feats = curr_meta_arg["detailed_flow_tracer"][bidx]["corrupted"]
            curr_block.anonymoususer_curr_subset = tuple(set().union(*flow_tracer.traced_paths[bidx]))
              
        with torch.no_grad():
            curr_pred, curr_p, raw_out = predict_from_input(
                curr_meta_arg["mt"].model, curr_meta_arg["inp"], multipred=(args.out_num!=1), end_symbol=args.end_symbol, use_mean=True)

        preds.append(copy.deepcopy(curr_pred))
        ps.append(copy.deepcopy(curr_p))
        # raw_outs.append(raw_out)
        curr_pred_unq = torch.unique(curr_pred)
        change_bool = curr_pred_unq==curr_meta_arg["normal_pred"]

        if (change_bool.sum()!=change_bool.shape[0]).item():
            if cond == "counterfactual":
                cond_bool.append(True)
            elif cond == "contingency":
                cond_bool.append(False)
        else:
            if cond == "counterfactual":
                cond_bool.append(False)
            elif cond == "contingency":
                cond_bool.append(True)
    cond_bool = torch.tensor(cond_bool)

    # print(f"cond_bool: {cond_bool}")
    actual_cause_bool = (cond_bool==True).all().item()
    if args.slightly_quiet_mode:
        if actual_cause_bool:
            log_line = "\t\t ({})[Subset: {}] | org:{}({}) path:CF-{}({}), CT-{}({})".format(
                "Causal" if actual_cause_bool else "Non-causal",
                curr_meta_arg["curr_subset"],
                curr_meta_arg["normal_pred"].item(), 
                round(curr_meta_arg["normal_p"].item(), 5), 
                preds[0].item() if args.efficient_mode is False else "",
                round(ps[0].item(), 5)if args.efficient_mode is False else "PASS",
                preds[1].item(), 
                round(ps[1].item(), 5),
            )
            curr_meta_arg["logger"].info(log_line)
    else:
        log_line = "\t\t ({})[Subset: {}] | org:{}({}) path:CF-{}({}), CT-{}({})".format(
            "Causal" if actual_cause_bool else "Non-causal",
            curr_meta_arg["curr_subset"],
            curr_meta_arg["normal_pred"].item(), 
            round(curr_meta_arg["normal_p"].item(), 5), 
            preds[0].item() if args.efficient_mode is False else "",
            round(ps[0].item(), 5)if args.efficient_mode is False else "PASS",
            preds[1].item(), 
            round(ps[1].item(), 5),
        )
        curr_meta_arg["logger"].info(log_line)


    decision_save = {
        "CF":  [str(preds[0].item()), str(ps[0].item())] if args.efficient_mode is False else ["None", "None"],
        "CT":  [str(preds[1].item()), str(ps[1].item())]
    }
    
    return actual_cause_bool, cond_bool, decision_save

def minimality_search(
    args, curr_meta_arg, flow_tracer):  
    curr_meta_arg["logger"].info("[MinSearch Start]")


    detailed_flow_tracer = {}
    for detailed_flow_tracer_idx in range(
            curr_meta_arg["curr_block_idx"], curr_meta_arg["total_block_num"], 1):
        curr_detailed_corrupted_flow_tracer = get_detailed_corrupted_flow_tracer_one_block(
            args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer,
            detailed_flow_tracer_idx=detailed_flow_tracer_idx)
        
        detailed_flow_tracer.update({
            detailed_flow_tracer_idx: {"corrupted": curr_detailed_corrupted_flow_tracer}
            })

    curr_meta_arg["detailed_flow_tracer"] = detailed_flow_tracer


    normal_p, normal_pred = torch.max(flow_tracer.scores_normal, dim=0)
    curr_meta_arg["normal_p"] = normal_p
    curr_meta_arg["normal_pred"] = normal_pred

    # Option 1: Get n_head directly from the attention module
    n_head = curr_meta_arg['mt'].model.blocks[0].attn.num_heads
    curr_meta_arg["num_path_node"] = n_head * 2 + 2

    # Option 2: If num_heads isn't accessible, you could pass it as an argument
    # curr_meta_arg["num_path_node"] = args.num_heads * 2 + 2


    Xs_index = np.arange(curr_meta_arg["num_path_node"])
    min_search_memory = {}
    selected_subset = []
    step_subset = {}
    step_subset_cond_bool = {}
    stop_step = curr_meta_arg["num_path_node"]

    if args.debug_min_search:
        curr_meta_arg["logger"].info("\t [NOTE] This is the debug mode of Min. Search. It will pass 1~{} steps".format((curr_meta_arg["num_path_node"]-3)-1))

    minsear_iter = sorted(np.arange(curr_meta_arg["num_path_node"])+1)
    for curr_step in minsear_iter:
        
        if args.debug_min_search:
            if curr_step < (curr_meta_arg["num_path_node"]-3):
                continue

        curr_step_subset = list(itertools.combinations(Xs_index, curr_step))

        if args.subset_search == "minimality":
            subset_candidate = exclude_subsets(curr_step_subset, selected_subset, curr_meta_arg["logger"])
        else:
            subset_candidate = copy.deepcopy(curr_step_subset)
        
        if len(subset_candidate) == 0:
            stop_step = curr_step-1

        buffer_cond_bool = []
        if len(subset_candidate) != 0:
            curr_meta_arg["logger"].info("\t MinSearch Step: ({}/{}) -> Target Num.: {}".format(curr_step, max(minsear_iter), len(subset_candidate) ))
        if args.detailed_save:
            save_curr_step_causal_decision = {}
            save_curr_step_causal_subset = {}
            save_curr_step_causal_index = 0

        curr_meta_arg["curr_step"] = curr_step
        for subidx, curr_subset in enumerate(subset_candidate):
            
            curr_meta_arg["curr_subset"] = curr_subset

            if args.evaluation_mode:
                actual_cause_bool, cond_bool, decision_save = cause_edge_classifier_for_evaluation(
                    args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer
                )
            else:
                actual_cause_bool, cond_bool, decision_save = cause_edge_classifier(
                    args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer
                )

            buffer_cond_bool.append(cond_bool.tolist())

            if actual_cause_bool:
                selected_subset.append(curr_subset)
                if args.detailed_save:
                    save_curr_step_causal_decision.update({save_curr_step_causal_index: decision_save})
                    save_curr_step_causal_subset.update({save_curr_step_causal_index: list(map(str, curr_subset))})
                    save_curr_step_causal_index+=1
        step_subset[curr_step] = subset_candidate
        step_subset_cond_bool[curr_step] = buffer_cond_bool

        if args.detailed_save:
            save_curr_step_data = {
                "org_decision": [curr_meta_arg["normal_pred"].item(), curr_meta_arg["normal_p"].item()],
                "subset_decision": save_curr_step_causal_decision,
                "subset": save_curr_step_causal_subset
            }
            
            detailed_curr_save_dir = os.path.join(args.curr_save_dir, "detailed")
            os.makedirs(detailed_curr_save_dir, exist_ok=True)
            detailed_block_dir = os.path.join(detailed_curr_save_dir, "block_{:02d}".format(curr_meta_arg["curr_block_idx"]))
            os.makedirs(detailed_block_dir, exist_ok=True)

            with open(os.path.join(detailed_block_dir, "step_{:03d}.json".format(curr_step)), "w") as f:
                json.dump(save_curr_step_data, f, indent=4)

    stop_flag = False
    if len(selected_subset)==0:
        curr_meta_arg["logger"].info("[Error] Selected subsets are empty!")
        
        # Log the error to a file in the results directory
        error_log_dir = os.path.dirname(args.save_root)
        os.makedirs(error_log_dir, exist_ok=True)  # Create directory if it doesn't exist
        error_log_path = os.path.join(error_log_dir, "log_error_samples.txt")
        with open(error_log_path, "a") as f:
            f.write(f"sample {args.target_sample_idx} has empty selected subset error.\n")
            
        stop_flag = True
        

    min_search_memory["selected_subset"] = selected_subset
    min_search_memory["stop_step"] = stop_step
    min_search_memory["step_subset"] = step_subset
    min_search_memory["step_subset_cond_bool"] = step_subset_cond_bool

    # Restore orignal module
    for bidx in range(curr_meta_arg["total_block_num"]):
        curr_block = curr_meta_arg["mt"].model.blocks[bidx]
        curr_block = block_initializer(args, curr_block)

    return min_search_memory, stop_flag

def top_down_causal_flow_trace(
    args, curr_meta_arg, flow_tracer):

    expanded_images = curr_meta_arg["images"].repeat(args.num_noise_sample, 1, 1, 1)
    inp = {"pixel_values": expanded_images}
    
    curr_meta_arg["inp"] = inp
    stop_flag = False
    for curr_block_idx in tqdm.tqdm(range(curr_meta_arg["total_block_num"]-1, -1, -1)):
        start_block_time = time.time()
        curr_meta_arg["curr_block_idx"] = curr_block_idx
        curr_meta_arg["logger"].info("-----------Block Change----------- Block:{}".format( curr_meta_arg["curr_block_idx"]))
        
        min_search_memory, stop_flag = minimality_search(args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer)
        if stop_flag:
            curr_meta_arg["logger"].info(f"Stopping early at block {curr_block_idx} due to empty selected subset")
            break
        else:
            flow_tracer.traced_paths.update({curr_block_idx: copy.deepcopy(min_search_memory["selected_subset"])})
       

    return stop_flag


################################################################################
# path_forward (CausalPathTracing_anonymous/CPT/vision/path_forward.py)
################################################################################
# Per-sample body of the original path_forward.py main(), as functions.
from collections import Counter


def aggregate_paths(path_data, path_agg_type="union"):
    # path_data: {block_idx(str): [path, ...]} (C*.json) -> {block_idx(int): [node, ...]}
    curr_result_zip = {}
    for block_idx, paths in path_data.items():
        if "union" in path_agg_type:
            sets = [set(lst) for lst in paths]
            target_paths = list(set.union(*sets))
        elif "min" in path_agg_type:
            min_len = min(len(lst) for lst in paths)
            target_paths = [lst for lst in paths if len(lst) == min_len][0]
        elif path_agg_type == "intersection":
            sets = [set(lst) for lst in paths]
            target_paths = set.intersection(*sets)
        elif path_agg_type == "over1":
            flattened = [item for sublist in paths for item in sublist]
            counter = Counter(flattened)
            target_paths = sorted([item for item, count in counter.items() if count >= 2])
        elif path_agg_type == "top_min_path_size":
            min_len = min(len(lst) for lst in paths)
            flattened = [item for sublist in paths for item in sublist]
            counter = Counter(flattened)
            target_paths = [item for item, count in counter.most_common(min_len)]
        curr_result_zip.update({int(block_idx): target_paths})
    return curr_result_zip


def get_num_path_node(mt):
    # number of path nodes in one block
    n_head = mt.model.blocks[0].attn.num_heads
    return n_head * 2 + 2


def path_forward(args, mt, images, labels, causal_path, interv_target="off_path"):
    """
    off_path: keep the causal path nodes and intervene (corrupt) all the other nodes.
    on_path : intervene the causal path nodes.
    causal_path: {block_idx(int): [node, ...]} (aggregate_paths output)
    Returns org/path decision and outputs (sorted by the original prediction order).
    """
    flow_tracer = CausalFlowTracer()
    flow_tracer, answer, curr_total_token_num, curr_total_block_num, normal_inp = predict_from_normal_and_noise_input(
        images, flow_tracer, labels, None,
        mt, n_lev=None, end_symbol=None, out_num=1,
        model_name=args.model,
        dataset_name=args.dataset,
        num_normal_sample=args.num_noise_sample,
        num_noise_sample=args.num_noise_sample,
        noise_type=args.noise_type)

    curr_meta_arg = {}
    curr_meta_arg["total_block_num"] = curr_total_block_num
    curr_meta_arg["total_token_num"] = curr_total_token_num
    curr_meta_arg["images"] = images
    curr_meta_arg["mt"] = mt
    expanded_images = curr_meta_arg["images"].repeat(args.num_noise_sample, 1, 1, 1)
    inp = {"pixel_values": expanded_images}

    curr_meta_arg["inp"] = inp

    detailed_flow_tracer = {}
    for detailed_flow_tracer_idx in range(0, curr_meta_arg["total_block_num"], 1):

        curr_detailed_corrupted_flow_tracer = get_detailed_corrupted_flow_tracer_one_block(
            args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer,
            detailed_flow_tracer_idx=detailed_flow_tracer_idx)

        detailed_flow_tracer.update({
            detailed_flow_tracer_idx: {"corrupted": curr_detailed_corrupted_flow_tracer}
            })

    curr_meta_arg["detailed_flow_tracer"] = detailed_flow_tracer

    normal_p, normal_pred = torch.max(flow_tracer.scores_normal, dim=0)
    desc_idx = torch.argsort(flow_tracer.scores_normal, dim=0, descending=True)
    normal_output = torch.index_select(flow_tracer.scores_normal, 0, desc_idx)
    pred_ord = desc_idx

    curr_meta_arg["normal_p"] = normal_p
    curr_meta_arg["normal_pred"] = normal_pred
    curr_meta_arg["normal_output"] = normal_output
    curr_meta_arg["normal_pred_order"] = pred_ord
    curr_meta_arg["num_path_node"] = get_num_path_node(curr_meta_arg['mt'])

    for bidx in range(curr_meta_arg["total_block_num"]):
        curr_block = curr_meta_arg["mt"].model.blocks[bidx]
        curr_block = block_initializer(args, curr_block)

        fwd_method_dict = return_forward_method_dict(args)

        curr_block.forward = types.MethodType(fwd_method_dict['intervention_forward'], curr_block)
        curr_block.mlp.forward = types.MethodType(fwd_method_dict['custom_mlp_forward'], curr_block.mlp)
        curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)

        curr_block.anonymoususer_trace_mode = True
        curr_block.anonymoususer_cond = "path"
        curr_block.anonymoususer_corrupted_feats = curr_meta_arg["detailed_flow_tracer"][bidx]["corrupted"]

        if interv_target=="off_path":
            curr_block.anonymoususer_curr_subset = tuple(causal_path[bidx])
        elif interv_target=="on_path":
            curr_block.anonymoususer_curr_subset = tuple(np.setdiff1d(np.arange(curr_meta_arg["num_path_node"]), causal_path[bidx]).tolist())

    with torch.no_grad():
        curr_pred, curr_p, raw_out = predict_from_input(
            curr_meta_arg["mt"].model, curr_meta_arg["inp"], multipred=(args.out_num!=1), end_symbol=[], use_mean=True)

        raw_out_sftmx = torch.softmax(raw_out, dim=1).mean(dim=0).unsqueeze(0)
        desc_idx = torch.argsort(raw_out_sftmx, dim=1, descending=True)

        output = torch.index_select(raw_out_sftmx, 1, desc_idx.squeeze(0)).squeeze(0)
        pred_order = desc_idx.squeeze(0)

    for bidx in range(curr_meta_arg["total_block_num"]):
        curr_block = curr_meta_arg["mt"].model.blocks[bidx]
        curr_block = block_initializer(args, curr_block)

    token_id_to_index = {token.item(): i for i, token in enumerate(pred_order)}
    sorted_pred_order = torch.tensor([token_id_to_index[i.item()] for i in curr_meta_arg["normal_pred_order"]])

    result = {
        "org_decision": curr_meta_arg["normal_pred"].item(),
        "path_decision": curr_pred.item(),
        "org_output": curr_meta_arg["normal_output"].cpu(),
        "path_output": output[sorted_pred_order].cpu(),
        "total_block_num": curr_meta_arg["total_block_num"],
        "num_path_node": curr_meta_arg["num_path_node"],
    }
    return result
