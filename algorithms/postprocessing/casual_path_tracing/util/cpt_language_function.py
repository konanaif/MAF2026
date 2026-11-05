################################################################################
# nethook (CausalPathTracing_anonymous/CPT/language/lib/nethook.py)
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
# models (CausalPathTracing_anonymous/CPT/language/lib/models.py)
################################################################################
import re

from transformers import AutoModelForCausalLM, AutoTokenizer


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
# args (CausalPathTracing_anonymous/CPT/language/lib/args.py)
################################################################################
import argparse

def arg_func(arg_list=None):

        
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="6", type=str, help='gpu')
    parser.add_argument('--end_symbol', default=[".", "?"])
    parser.add_argument('--save_root', default=os.path.join(str(MAF_ROOT), "result", "casual_path_tracing", "run"), type=str)
    parser.add_argument('--load_path_saver', default=None, type=str) # folder

    parser.add_argument('--num_noise_sample', default=10, type=int)
    parser.add_argument('--model', default="pythia-14m", type=str)
    parser.add_argument('--noise_type', default="other", required=True, choices=['other', 'emb_added', 'zero','mean'], help="noise type")
    parser.add_argument('--dataset_type', default="rome_factual_1000", choices=['rome_factual_1000'], help="dataset_type")

    parser.add_argument('--debug_index', default=None, type=int)
    parser.add_argument('--subset_search', default="minimality", type=str)
    parser.add_argument('--efficient_mode', help='not check counterfactual', action='store_true')
    parser.add_argument('--slightly_quiet_mode', help='slightly_quiet_mode', action='store_true')
    parser.add_argument('--debug_min_search', help='debug_min_search', action='store_true')
    parser.add_argument('--reverse', help='debug_min_search', action='store_true')
    parser.add_argument('--model_save', default=None, type=str)
    
    parser.add_argument('--detailed_save', help='detailed_save', action='store_true')
    parser.add_argument('--except_stopword', help='except_stopword', action='store_true')
    parser.add_argument('--correct_check_first', help='correct_check_first', action='store_true')
    
    parser.add_argument('--target_sample_index', default=None, type=str)
    parser.add_argument('--target_sample_index_list', default=None, type=str)
    
    
    args = parser.parse_args(arg_list)

    args.out_num = 1

    if args.model == "pythia-14m":
        args.model = "EleutherAI/pythia-14m"
        

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
# utils (CausalPathTracing_anonymous/CPT/language/lib/utils.py)
################################################################################
import os
import sys
import logging
import tarfile
import pickle
import time
import copy
import torch
import json
import tqdm
import dill 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from datetime import datetime

from functools import wraps
from pathlib import Path
from MAF.datamodule.casual_path_tracing.sample import KnownsDataset, SAMPLE_DIR, MAF_ROOT

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = Path(SAMPLE_DIR)



def set_utils(args):

    # Save Folder
    if "debug" not in args.save_root:
        if os.path.isdir(args.save_root) is True:
            print("Check your path!")
            import pdb; pdb.set_trace()

    curr_time = datetime.now().strftime('%Y%m%d_%H%M')
    save_parent, save_name = os.path.split(os.path.normpath(args.save_root))

    args.save_root = os.path.join(save_parent, "{}_{}_{}".format(curr_time, save_name, args.model if "/" not in args.model else args.model.split("/")[-1]))
    if args.reverse:
        args.save_root += '_rev'
    if args.dataset_type != "rome_factual_1000":
        args.save_root += ("_" + args.dataset_type)
    if args.except_stopword:
        args.save_root += '_exStWd'
    if args.target_sample_index is not None:
        args.save_root += '_R{}'.format(args.target_sample_index)
    if args.target_sample_index_list is not None:
        spt = args.target_sample_index_list.split(", ")
        args.save_root += '_R{}-{}'.format(spt[0], spt[-1])
        
    os.makedirs(args.save_root, exist_ok=True)

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
    func_time_saver = function_time_saver()
    if "debug" in args.save_root:
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

def exclude_subsets(candidate, exclude_target):
    if len(exclude_target)==0:
        return candidate
    new_candidate = []
    for cand in candidate:
        exclude_flag = False
        for ex in exclude_target:
            # ex is always smaller than cand, if the search proceeds with small number of steps.
            if len(set(ex).union(cand))==len(cand):
                exclude_flag = True
                break
        if exclude_flag is False:
            new_candidate.append(cand)
    return new_candidate

def get_data(args, reverse=False):
    # known_1000 (ROME factual) only
    line_data = KnownsDataset(DATA_DIR, reverse=reverse)
    return line_data

def get_model(args):
    
    mt = ModelAndTokenizer(
        args.model,
        low_cpu_mem_usage=False,
        torch_dtype=(torch.float16 if "20b" in args.model else None),
    )
    if args.model_save is not None:
        os.makedirs(args.model_save, exist_ok=True)
        curr_model_path = os.path.join(args.model_save, args.model.replace("/", "_"))
        os.makedirs(curr_model_path, exist_ok=True)
        torch.save(mt.model.state_dict(), os.path.join(curr_model_path, "model.pth"))
        mt.tokenizer.save_pretrained(os.path.join(curr_model_path, "tokenizer"))
    
    if args.except_stopword:
        get_stopwords(args, mt)
        
    return mt


def get_noise_level(args, mt, fpath_nlv=os.path.join(SAMPLE_DIR, "knowns_n_lv.pt")):
    if os.path.isfile(fpath_nlv) is False:
        knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts
        noise_level = collect_embedding_std(mt, [k["subject"] for k in knowns])
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


def predict_from_input(model, inp, multipred=False, end_symbol=[], mt=None, force_idx=None, use_mean=False, stwd_mask=None):
    # multipred option makes outputs until resulting one sentence output
    # Don't reduce the batch size...: https://discuss.pytorch.org/t/why-is-the-output-of-a-linear-layer-different-when-the-batch-size-is-1/93515
    if multipred is False:
        out = model(**inp)["logits"]
        if use_mean:
            probs = torch.softmax(out[:, -1], dim=1).mean(dim=0).unsqueeze(0)
        else:
            probs = torch.softmax(out[:, -1], dim=1)
        
        if (stwd_mask is not None):
            if force_idx is None:
                desc_idx = torch.argsort(probs, dim=1, descending=True)
                sorted_stwd_mask = stwd_mask[desc_idx]
                preds = desc_idx[sorted_stwd_mask][0]
                p = probs[:, preds][0]
            else: 
                import pdb; pdb.set_trace()
        else:   
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
        prompt, flow_tracer, y, logger,
        mt, n_lev, num_noise_sample=3, num_normal_sample=3, noise_type="other",
        end_symbol=[".", "?"], out_num=1, correct_check_only=False):
    
    # num_noise_sample and num_normal_sample should be same.
    # their outputs are slightly different (it maybe the batch-wise operation in hardware-level.)
    
    normal_inp = make_inputs(mt.tokenizer, [prompt] * (num_normal_sample))
    # noise_inp = make_inputs(mt.tokenizer, [prompt] * (num_noise_sample))

    if out_num==1:
        answer_t = flow_tracer.trace_normal(mt.model, normal_inp)
        answer = decode_tokens(mt.tokenizer, answer_t)
        if correct_check_only:
            return y in answer[0]
        if y not in answer[0]:
            return -1, -1, -1, -1, -1
        noise_rand_seed = 0
        while 1:
            corrupted_answer_t, used_token = flow_tracer.trace_corrupted(
                mt.model, mt.tokenizer, prompt, noise_level=n_lev, rand_seed=noise_rand_seed, noise_type=noise_type, num_noise_sample=num_noise_sample)
            
            if answer_t.item() != corrupted_answer_t.item():
                break
            noise_rand_seed += 1
            ### For future debugging
            # corrupted_answer = decode_tokens(mt.tokenizer, corrupted_answer_t)
            # corrupted_token = decode_tokens(mt.tokenizer, used_token)
            logging.info("[Noise Finder] Changing the seed to find noise that changes the output... Current Seed:{}, Ans:{}".format(
                noise_rand_seed, answer[0]))


    
    curr_total_token_num = normal_inp["input_ids"].shape[-1]
    
    if hasattr(mt.model, "transformer"):
        curr_total_block_num = len(mt.model.transformer.h)
    elif hasattr(mt.model, "gpt_neox"):
        curr_total_block_num = len(mt.model.gpt_neox.layers)
    
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

def get_stopwords(args, mt):
    stopword_list = stopwords.words('english')
    
    stopword_ids = []
    for stopword in stopword_list:
        token_ids = mt.tokenizer.encode(' '+stopword, add_special_tokens=False)
        if len(token_ids) == 1:
            stopword_ids.append(token_ids[0])
        token_ids = mt.tokenizer.encode(stopword, add_special_tokens=False)
        if len(token_ids) == 1:
            stopword_ids.append(token_ids[0])
            
    args.stwd_ids = sorted(list(set(stopword_ids)))


################################################################################
# causal_flow_tracer (CausalPathTracing_anonymous/CPT/language/lib/causal_flow_tracer.py)
################################################################################
import copy
import torch
from torch import nn
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple, Union



from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:
    class FlashAttentionKwargs:
        pass
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from transformers.models.gpt_neox.modeling_gpt_neox import eager_attention_forward as eager_attention_forward_gpt_neox


class CausalFlowTracer(object):

    def __init__(self, args):
        self.traced_paths = {}
        self.stwd_ids = None
        if args.except_stopword:
            self.stwd_ids = args.stwd_ids
            self.stwd_mask = None
            
    
    def init_node(self):
        self.in_term1 = None
        self.in_term2 = None
        self.in_term3_list = None
        self.in_term4_list = None

    def trace_normal(
            self, model, inp):

        self.feats_normal = None
        self.feats_k_normal = None
        self.feats_v_normal = None
        self.scores_normal = None

        inputs_block = {"input": []}
        outputs = {"output": [], "output_k": [], "output_v": []}

        def make_hook_fn(block_idx):
            def hook_fn(module, input, output):
                if block_idx==0:
                    inputs_block["input"].append(input[0].detach())
                
                if len(output)==1:
                    outputs["output"].append(output[0].detach())
                elif len(output)==2:
                    outputs["output"].append(output[0].detach())
                    outputs["output_k"].append(output[1][0].detach())
                    outputs["output_v"].append(output[1][1].detach())
                else:
                    import pdb; pdb.set_trace()
            return hook_fn

        
        forward_hooks = []
        if hasattr(model, "transformer"):
            for block_idx, block in enumerate(model.transformer.h):
                forward_hook = block.register_forward_hook(make_hook_fn(block_idx))
                forward_hooks.append(forward_hook)
        elif hasattr(model, "gpt_neox"):
            for block_idx, block in enumerate(model.gpt_neox.layers):
                forward_hook = block.register_forward_hook(make_hook_fn(block_idx))
                forward_hooks.append(forward_hook)

        with torch.no_grad():
            outputs_exp = model(**inp)

        for forward_hook in forward_hooks:
            forward_hook.remove()

        scores_normal = torch.softmax(outputs_exp.logits[:, -1, :], dim=1)[0]
        if self.stwd_ids is None:
            answer_t = torch.max(scores_normal, dim=0).indices.unsqueeze(0)
        else:
            if self.stwd_mask is None:
                self.stwd_mask = torch.ones(scores_normal.shape[0], dtype=torch.bool, device=scores_normal.device)
                self.stwd_mask[self.stwd_ids] = False 
                
            desc_idx = torch.argsort(scores_normal, dim=0, descending=True)
            sorted_stwd_mask = self.stwd_mask[desc_idx]
            answer_t = desc_idx[sorted_stwd_mask][0].unsqueeze(0)
        
        all_traced_normal = outputs["output"]
        if len(outputs["output_k"])!=0:
            all_traced_k_normal = outputs["output_k"]
            self.feats_k_normal = copy.deepcopy(all_traced_k_normal)
        if len(outputs["output_v"])!=0:
            all_traced_v_normal = outputs["output_v"]
            self.feats_v_normal = copy.deepcopy(all_traced_v_normal)

        # torch.clone makes different outputs -> please use 
        self.feats_normal_init = copy.deepcopy(inputs_block["input"][0])
        self.feats_normal = copy.deepcopy(all_traced_normal)
        self.scores_normal = copy.deepcopy(scores_normal)

        return answer_t


    def trace_corrupted(
            self, model, tokenizer, prompt, noise_level=0.1, rand_seed=0, noise_type="other", num_noise_sample=3):
        
        self.feats_corrupted = None
        self.feats_k_corrupted = None
        self.feats_v_corrupted = None
        self.scores_corrupted = None

        prng = np.random.RandomState(rand_seed)  # For reproducibility, use pseudorandom noise

        def add_noise_to_embedding(module, input, output):
            noise = noise_level * torch.from_numpy(prng.randn(output.shape[0], output.shape[1], output.shape[2])).to(output.device).float()
            return output + noise
            # return output

        if noise_type=="emb_added":
            inp = make_inputs(tokenizer, [prompt] * (num_noise_sample))
            embedding_layer = model.transformer.wte 
            noise_hook_handle = embedding_layer.register_forward_hook(add_noise_to_embedding)
        elif noise_type=="other":
            enc_prompt = tokenizer.encode(prompt)
            available_tokens = list(set(range(tokenizer.vocab_size)) - set(enc_prompt))
            selected_tokens = prng.choice(available_tokens, size=num_noise_sample*len(enc_prompt))
            inp = make_inputs(tokenizer, selected_tokens.reshape(num_noise_sample, len(enc_prompt)).tolist(), pass_encoding=True)
        else:
            import pdb; pdb.set_trace()

        inputs_block = {"input": []}
        outputs = {"output": [], "output_k": [], "output_v": []}

        def make_hook_fn(block_idx):
            def hook_fn(module, input, output):
                if block_idx==0:
                    inputs_block["input"].append(input[0].detach())
                    
                if len(output)==1:
                    outputs["output"].append(output[0].detach())
                elif len(output)==2:
                    outputs["output"].append(output[0].detach())
                    outputs["output_k"].append(output[1][0].detach())
                    outputs["output_v"].append(output[1][1].detach())
                else:
                    import pdb; pdb.set_trace()
            return hook_fn

        forward_hooks = []
        if hasattr(model, "transformer"):
            for block_idx, block in enumerate(model.transformer.h):
                forward_hook = block.register_forward_hook(make_hook_fn(block_idx))
                forward_hooks.append(forward_hook)
        elif hasattr(model, "gpt_neox"):
            for block_idx, block in enumerate(model.gpt_neox.layers):
                forward_hook = block.register_forward_hook(make_hook_fn(block_idx))
                forward_hooks.append(forward_hook)

        with torch.no_grad():
            outputs_exp = model(**inp)

        for forward_hook in forward_hooks:
            forward_hook.remove()
            
        if noise_type=="emb_added":
            noise_hook_handle.remove()

        probs_corrupted = torch.softmax(outputs_exp.logits[:, -1, :], dim=1).mean(dim=0).unsqueeze(0)
   
        if self.stwd_ids is None:
            corrupted_answer_t = torch.max(probs_corrupted, dim=1).indices
        else:
            if self.stwd_mask is None:
                self.stwd_mask = torch.ones(probs_corrupted.shape[0], dtype=torch.bool, device=probs_corrupted.device)
                self.stwd_mask[self.stwd_ids] = False 
                
            desc_idx = torch.argsort(probs_corrupted, dim=1, descending=True)
            sorted_stwd_mask = self.stwd_mask[desc_idx]
            corrupted_answer_t = desc_idx[sorted_stwd_mask][0].unsqueeze(0)
                
                
                
        all_traced_corrupted = outputs["output"]
        if len(outputs["output_k"])!=0:
            all_traced_k_corrupted = outputs["output_k"]
            self.feats_k_corrupted = copy.deepcopy(all_traced_k_corrupted)
        if len(outputs["output_v"])!=0:
            all_traced_v_corrupted = outputs["output_v"]
            self.feats_v_corrupted = copy.deepcopy(all_traced_v_corrupted)

        self.feats_corrupted_init = copy.deepcopy(inputs_block["input"][0])
        self.feats_corrupted = copy.deepcopy(all_traced_corrupted)
        self.scores_corrupted = copy.deepcopy(probs_corrupted)
        return corrupted_answer_t, inp['input_ids']



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

def layernorm2mm(x, layernorm_layer, prefix_mu=None, prefix_std=None, bias_divide=None):
    # x.shape == [batch, token, hidden]
    a, c = make_layernorm2mm(x, layernorm_layer, prefix_mu, prefix_std, bias_divide)
    out = x * a + c  # Broadcasting: [batch, token, hidden] * [hidden] + [hidden]
    return out
def make_homoneneous_coord(x):
    ones = torch.ones_like(x[..., :1])
    x_ext = torch.cat([x, ones], dim=-1)
    return x_ext

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
                hidden_states = copy.deepcopy(self.anonymoususer_feats_corrupted_init)
    
    residual = hidden_states

    #-------for Path Division-------#
    residual_in = copy.deepcopy(residual)
    #-------for Path Division-------#

    hidden_states = self.ln_1(hidden_states)
    
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
    ln2_mu, ln2_std = get_lm_inp_mu_std(copy.deepcopy(hidden_states.detach()), self.ln_2.eps)
    W_ln_mlp, b_ln_mlp = make_layernorm2mm([], self.ln_2, prefix_mu=ln2_mu, prefix_std=ln2_std)
    #-------for Path Division-------#

    hidden_states_ln2 = self.ln_2(hidden_states)

    feed_forward_hidden_states, gelu_input_states = self.mlp(hidden_states_ln2)

    #-------for Path Division-------#
    D_a = torch.where(gelu_input_states.abs() > 1e-4, nn.functional.gelu(gelu_input_states) / gelu_input_states, torch.full_like(gelu_input_states, 0.5))
    #-------for Path Division-------#

    hidden_states = residual + feed_forward_hidden_states

    #-------for Path Division-------#
    org_output = copy.deepcopy(hidden_states)
    #-------for Path Division-------#


    #-------Path Division-------#
    in_term1 = residual_in
    in_term2 = self.mlp(ln_mm(residual_in, W_ln_mlp, b=None), custom_gelu=D_a, bias_divide=None, ignore_bias=True)[0]

    in_term3_list, in_term4_list = [], []
    # debug_attn_out_proj = []
    for h_i in range(self.attn.num_heads):
        # Note: nn.Linear -> x*W^T, but, Conv1d -> x*W
        h_attn_out_proj = nn.functional.linear(
            head_wise_attn_output[:, h_i], 
            weight=self.attn.c_proj.weight.T[:, self.attn.head_dim * (h_i):self.attn.head_dim  * (h_i+1)],
            bias=self.attn.c_proj.bias/self.attn.num_heads
        )
        in_term3_i = h_attn_out_proj
        in_term3_list.append(in_term3_i.unsqueeze(0))
        
        in_term4_i = self.mlp(ln_mm(copy.deepcopy(h_attn_out_proj.detach()), W_ln_mlp, b=None), custom_gelu=D_a, bias_divide=None, ignore_bias=True)[0]
        in_term4_list.append(in_term4_i.unsqueeze(0))

    in_term3_list = torch.cat(in_term3_list, dim=0)
    in_term4_list = torch.cat(in_term4_list, dim=0)
    #-------Path Division-------#


    #-------Floating-point Error Correction-------#
    mlp_bias_term1 = ((b_ln_mlp @ self.mlp.c_fc.weight) * D_a) @ self.mlp.c_proj.weight
    mlp_bias_term2 = (self.mlp.c_fc.bias * D_a) @ self.mlp.c_proj.weight
    mlp_bias_term3 = self.mlp.c_proj.bias
    mlp_bias_term_all = mlp_bias_term1 + mlp_bias_term2 + mlp_bias_term3 

    in_term2 += (mlp_bias_term_all/(self.attn.num_heads+1))
    in_term4_list += (mlp_bias_term_all/(self.attn.num_heads+1)).unsqueeze(0)

    divided_output = in_term1 + in_term2 + torch.sum(in_term3_list, dim=0) + torch.sum(in_term4_list, dim=0)

    comp_err = divided_output-org_output
    per_comp_err = comp_err/(2*self.attn.num_heads + 2)

    # in_term1 += per_comp_err
    # in_term2 += per_comp_err
    # in_term3_list += per_comp_err.unsqueeze(0)
    # in_term4_list += per_comp_err.unsqueeze(0)
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
                # interv
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

    # # Unfolded output
    hidden_states = in_term1 + in_term2 + torch.sum(in_term3_list, dim=0) + torch.sum(in_term4_list, dim=0)
    
    if hasattr(self, "anonymoususer_trace_mode"):
        if self.anonymoususer_trace_mode:
            if self.anonymoususer_cond in ["path", "contingency"]:
                other_idx_term_len = [len(i) for i in other_idx_term]
                if sum(other_idx_term_len)==0:  # If the target subset is equal to the entire set, compensates for the error.
                    hidden_states = org_output
            elif self.anonymoususer_cond in ["counterfactual"]:
                hidden_states = copy.deepcopy(self.anonymoususer_corrupted_feats.org_output)
            else:
                import pdb; pdb.set_trace()

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


def custom_linear_forward(x, layer, bias_divide=1, ignore_bias=False):
    size_out = x.size()[:-1] + (layer.nf,)
    if ignore_bias:
        x = torch.addmm(torch.zeros_like(layer.bias), x.view(-1, x.size(-1)), layer.weight)
    else:
        x = torch.addmm(layer.bias/bias_divide, x.view(-1, x.size(-1)), layer.weight)
    x = x.view(size_out)
    return x

def custom_mlp_forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], custom_gelu=None, bias_divide=None, ignore_bias=False) -> torch.FloatTensor:
    if (bias_divide==None) & (ignore_bias==False):
        hidden_states = self.c_fc(hidden_states)
    else:
        hidden_states = custom_linear_forward(hidden_states, self.c_fc, bias_divide=bias_divide, ignore_bias=ignore_bias)

    gelu_input_states = copy.deepcopy(hidden_states.detach())
    if custom_gelu is None:
        hidden_states = self.act(hidden_states)
    else:
        hidden_states = hidden_states * custom_gelu

    if (bias_divide==None) & (ignore_bias==False):
        hidden_states = self.c_proj(hidden_states)
    else:
        hidden_states = custom_linear_forward(hidden_states, self.c_proj, bias_divide=bias_divide, ignore_bias=ignore_bias)
    hidden_states = self.dropout(hidden_states)
    return hidden_states, gelu_input_states


def custom_attn_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    
    
    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        query_states = self.q_attn(hidden_states)
        key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

    shape_q = (*query_states.shape[:-1], -1, self.head_dim)
    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

    query_states = query_states.view(shape_q).transpose(1, 2)
    key_states = key_states.view(shape_kv).transpose(1, 2)
    value_states = value_states.view(shape_kv).transpose(1, 2)

    if layer_past is not None:
        past_key, past_value = layer_past
        key_states = torch.cat((past_key, key_states), dim=-2)
        value_states = torch.cat((past_value, value_states), dim=-2)

    if use_cache is True:
        present = (key_states, value_states)
    else:
        present = None

    is_cross_attention = encoder_hidden_states is not None
    is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

    using_eager = self.config._attn_implementation == "eager"
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
            using_eager = True
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
            # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
            # not necessarily to eager (if mentionned options are provided).
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if using_eager and self.reorder_and_upcast_attn:
        attn_output, attn_weights = self._upcast_and_reordered_attn(
            query_states, key_states, value_states, attention_mask, head_mask
        )
    else:
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            head_mask=head_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
            **kwargs,
        )
    
    head_wise_attn_output = copy.deepcopy(attn_output.permute(0, 2, 1, 3).contiguous().detach())
    # batch, num_head, token, head_dim
    
    attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs, head_wise_attn_output  # a, present, (attentions)

def org_mlp_forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
    hidden_states = self.c_fc(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.c_proj(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


def org_attn_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        query_states = self.q_attn(hidden_states)
        key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

    shape_q = (*query_states.shape[:-1], -1, self.head_dim)
    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

    query_states = query_states.view(shape_q).transpose(1, 2)
    key_states = key_states.view(shape_kv).transpose(1, 2)
    value_states = value_states.view(shape_kv).transpose(1, 2)

    if layer_past is not None:
        past_key, past_value = layer_past
        key_states = torch.cat((past_key, key_states), dim=-2)
        value_states = torch.cat((past_value, value_states), dim=-2)

    if use_cache is True:
        present = (key_states, value_states)
    else:
        present = None

    is_cross_attention = encoder_hidden_states is not None
    is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

    using_eager = self.config._attn_implementation == "eager"
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
            using_eager = True
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
            # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
            # not necessarily to eager (if mentionned options are provided).
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if using_eager and self.reorder_and_upcast_attn:
        attn_output, attn_weights = self._upcast_and_reordered_attn(
            query_states, key_states, value_states, attention_mask, head_mask
        )
    else:
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            head_mask=head_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
            **kwargs,
        )
    attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)
def org_block_forward(
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
    
    
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
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

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)

    feed_forward_hidden_states = self.mlp(hidden_states)

    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)





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
    
    B, _, _ = hidden_states.shape
    attn_outputs = (copy.deepcopy(self.anonymoususer_feats_k_corrupted), copy.deepcopy(self.anonymoususer_feats_v_corrupted))
    outputs = (copy.deepcopy(self.anonymoususer_feats_corrupted), attn_outputs)

    return outputs  # hidden_states, present, (attentions, cross_attentions)



def pass_block_forward_gpt_neox(
    self,
    hidden_states: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):  
    outputs = (copy.deepcopy(self.anonymoususer_feats_normal), )
    return outputs
def pass_corrupted_block_forward_gpt_neox(
    self,
    hidden_states: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):  
    outputs = (copy.deepcopy(self.anonymoususer_feats_corrupted), )
    return outputs


def path_level_intervention_forward_gpt_neox(
    self,
    hidden_states: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):  
    if hasattr(self, "anonymoususer_save_mode"):
        if self.anonymoususer_save_mode:
            if hasattr(self, "anonymoususer_feats_corrupted_init"):
                hidden_states = copy.deepcopy(self.anonymoususer_feats_corrupted_init)
                
    #-------for Path Division-------#
    residual_in = copy.deepcopy(hidden_states)
    #-------for Path Division-------#
                
    attn_output, attn_weights, head_wise_attn_output = self.attention(
        self.input_layernorm(hidden_states),
        attention_mask=attention_mask,
        position_ids=position_ids,
        layer_past=layer_past,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    attn_output = self.post_attention_dropout(attn_output)

    if self.use_parallel_residual:
        # pseudocode:
        # x = x + attn(ln1(x)) + mlp(ln2(x))
        mlp_output, gelu_input_states = self.mlp(self.post_attention_layernorm(hidden_states))
        mlp_output = self.post_mlp_dropout(mlp_output)
        mlp_output_copy = copy.deepcopy(mlp_output)
        hidden_states = mlp_output + attn_output + hidden_states
    else:
        # pseudocode:
        # x = x + attn(ln1(x))
        # x = x + mlp(ln2(x))
        attn_output = attn_output + hidden_states
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output
        
    org_output = copy.deepcopy(hidden_states)
        
    in_term3_list = []
    for h_i in range(self.attention.config.num_attention_heads):
        # Note: nn.Linear -> x*W^T, but, Conv1d -> x*W
        h_attn_out_proj = nn.functional.linear(
            head_wise_attn_output[:, h_i], 
            weight=self.attention.dense.weight[:, self.attention.head_size * (h_i):self.attention.head_size  * (h_i+1)],
            bias=self.attention.dense.bias/self.attention.config.num_attention_heads
        )
        in_term3_list.append(h_attn_out_proj.unsqueeze(0))
        
    in_term1 = residual_in
    in_term2 = mlp_output_copy
    in_term3_list = torch.cat(in_term3_list, dim=0)
    
    if hasattr(self, "anonymoususer_save_mode"):
        if self.anonymoususer_save_mode:
            self.anonymoususer_flow_tracer.in_term1 = copy.deepcopy(in_term1)
            self.anonymoususer_flow_tracer.in_term2 = copy.deepcopy(in_term2)
            self.anonymoususer_flow_tracer.in_term3_list = copy.deepcopy(in_term3_list)
            self.anonymoususer_flow_tracer.org_output = copy.deepcopy(org_output)
            
    if hasattr(self, "anonymoususer_trace_mode"):
        if self.anonymoususer_trace_mode:
            target_idx_term, other_idx_term = subsetidx2pathnodeidx(self.anonymoususer_curr_subset, self.attention.config.num_attention_heads)
            if self.anonymoususer_cond in ["path", "contingency"]:
                # interv
                if len(target_idx_term[0])==0: 
                    in_term1 = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term1)
                if len(target_idx_term[1])==0:
                    in_term2 = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term2)
                if len(other_idx_term[2])!=0:
                    in_term3_list[other_idx_term[2]] = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term3_list[other_idx_term[2]])
            elif self.anonymoususer_cond in ["counterfactual"]:
                in_term1 = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term1)
                in_term2 = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term2)
                in_term3_list = copy.deepcopy(self.anonymoususer_corrupted_feats.in_term3_list)
            else:
                import pdb; pdb.set_trace()
                
    # # Unfolded output
    hidden_states = in_term1 + in_term2 + torch.sum(in_term3_list, dim=0)
    
    if hasattr(self, "anonymoususer_trace_mode"):
        if self.anonymoususer_trace_mode:
            if self.anonymoususer_cond in ["path", "contingency"]:
                other_idx_term_len = [len(i) for i in other_idx_term[:3]] # since gpt neox use parallel residual, so we don't need the consideration of term4
                if sum(other_idx_term_len)==0:  # If the target subset is equal to the entire set, compensates for the error.
                    hidden_states = org_output
            elif self.anonymoususer_cond in ["counterfactual"]:
                hidden_states = copy.deepcopy(self.anonymoususer_corrupted_feats.org_output)
            else:
                import pdb; pdb.set_trace()
    
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def custom_attn_forward_gpt_neox(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):  
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)

    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Cache QKV values
    if layer_past is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "partial_rotation_size": self.rotary_ndims,
            "cache_position": cache_position,
        }
        key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Checking for fallbacks in case an unsupported feature is requested
    attention_type = self.config._attn_implementation
    if (output_attentions or head_mask is not None) and self.config._attn_implementation in [
        "sdpa",
        "flash_attention_2",
    ]:
        logger.warning_once(
            f"Setting `attention_type` to `eager` because `{attention_type}` does not support"
            f" `output_attentions=True` or `head_mask`."
        )
        attention_type = "eager"

    elif self.training and self.attention_dropout > 0 and self.config._attn_implementation == "flex_attention":
        logger.warning_once(
            f"Setting `attention_type` to `eager` because `dropout` is not supported in `{attention_type}`."
        )
        attention_type = "eager"

    attention_interface: Callable = eager_attention_forward_gpt_neox
    attention_interface = (
        ALL_ATTENTION_FUNCTIONS[attention_type] if attention_type != "eager" else attention_interface
    )

    # Compute attention
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        head_mask=head_mask,
        **kwargs,
    )
    head_wise_attn_output = copy.deepcopy(attn_output.permute(0, 2, 1, 3).contiguous().detach())
    
    # Reshape outputs and final projection
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)

    return attn_output, attn_weights, head_wise_attn_output

def custom_linear_forward_gpt_neox(x, layer, bias_divide=1, ignore_bias=False):
    size_out = x.size()[:-1]
    if ignore_bias:
        x = torch.addmm(torch.zeros_like(layer.bias), x.view(-1, x.size(-1)), layer.weight.T)
    else:
        x = torch.addmm(layer.bias/bias_divide, x.view(-1, x.size(-1)), layer.weight.T)
    x = x.view(size_out[0], size_out[1], -1)
    return x

def custom_mlp_forward_gpt_neox(self, hidden_states, custom_gelu=None, bias_divide=None, ignore_bias=False):
    if (bias_divide==None) & (ignore_bias==False):
        hidden_states = self.dense_h_to_4h(hidden_states)
    else:
        hidden_states = custom_linear_forward_gpt_neox(hidden_states, self.dense_h_to_4h, bias_divide=bias_divide, ignore_bias=ignore_bias)
        
    gelu_input_states = copy.deepcopy(hidden_states.detach())
    if custom_gelu is None:
        hidden_states = self.act(hidden_states)
    else:
        hidden_states = hidden_states * custom_gelu
        
    if (bias_divide==None) & (ignore_bias==False):
        hidden_states = self.dense_4h_to_h(hidden_states)
    else:
        hidden_states = custom_linear_forward_gpt_neox(hidden_states, self.dense_4h_to_h, bias_divide=bias_divide, ignore_bias=ignore_bias)
    return hidden_states, gelu_input_states

def org_block_forward_gpt_neox(
    self,
    hidden_states: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):  
    attn_output, attn_weights = self.attention(
        self.input_layernorm(hidden_states),
        attention_mask=attention_mask,
        position_ids=position_ids,
        layer_past=layer_past,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    attn_output = self.post_attention_dropout(attn_output)

    if self.use_parallel_residual:
        # pseudocode:
        # x = x + attn(ln1(x)) + mlp(ln2(x))
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output + hidden_states
    else:
        # pseudocode:
        # x = x + attn(ln1(x))
        # x = x + mlp(ln2(x))
        attn_output = attn_output + hidden_states
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs



def org_attn_forward_gpt_neox(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):  
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)

    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Cache QKV values
    if layer_past is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "partial_rotation_size": self.rotary_ndims,
            "cache_position": cache_position,
        }
        key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Checking for fallbacks in case an unsupported feature is requested
    attention_type = self.config._attn_implementation
    if (output_attentions or head_mask is not None) and self.config._attn_implementation in [
        "sdpa",
        "flash_attention_2",
    ]:
        logger.warning_once(
            f"Setting `attention_type` to `eager` because `{attention_type}` does not support"
            f" `output_attentions=True` or `head_mask`."
        )
        attention_type = "eager"

    elif self.training and self.attention_dropout > 0 and self.config._attn_implementation == "flex_attention":
        logger.warning_once(
            f"Setting `attention_type` to `eager` because `dropout` is not supported in `{attention_type}`."
        )
        attention_type = "eager"

    attention_interface: Callable = eager_attention_forward_gpt_neox
    attention_interface = (
        ALL_ATTENTION_FUNCTIONS[attention_type] if attention_type != "eager" else attention_interface
    )

    # Compute attention
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        head_mask=head_mask,
        **kwargs,
    )

    # Reshape outputs and final projection
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)

    return attn_output, attn_weights

def org_mlp_forward_gpt_neox(self, hidden_states):
    hidden_states = self.dense_h_to_4h(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.dense_4h_to_h(hidden_states)
    return hidden_states

FUNCTION_MAP = {
    "gpt2": {
        "org_block_forward": org_block_forward,
        "org_attn_forward": org_attn_forward,
        "org_mlp_forward": org_mlp_forward,
        "intervention_forward": path_level_intervention_forward,
        "custom_attn_forward": custom_attn_forward,
        "custom_mlp_forward": custom_mlp_forward,
        "pass_block_forward": pass_block_forward,
        "pass_corrupted_block_forward": pass_corrupted_block_forward
        },
    "gpt_neox": {
        "org_block_forward": org_block_forward_gpt_neox,
        "org_attn_forward": org_attn_forward_gpt_neox,
        "org_mlp_forward": org_mlp_forward_gpt_neox,
        "intervention_forward": path_level_intervention_forward_gpt_neox,
        "custom_attn_forward": custom_attn_forward_gpt_neox,
        "custom_mlp_forward": custom_mlp_forward_gpt_neox,
        "pass_block_forward": pass_block_forward_gpt_neox,
        "pass_corrupted_block_forward": pass_corrupted_block_forward_gpt_neox
        },
}

def return_forward_method_dict(args):
    if "gpt2" in args.model:
        return FUNCTION_MAP["gpt2"]
    elif "pythia" in args.model:
        return FUNCTION_MAP["gpt_neox"]


################################################################################
# search_func (CausalPathTracing_anonymous/CPT/language/lib/search_func.py)
################################################################################
import types
import time
import pickle
import itertools



def block_initializer(args, curr_block, symbolic_name="anonymoususer"):
    fwd_method_dict = return_forward_method_dict(args)
    
    ### INITIALIZE
    # restore the original function
    curr_block.forward = types.MethodType(fwd_method_dict["org_block_forward"], curr_block)
    curr_block.mlp.forward = types.MethodType(fwd_method_dict["org_mlp_forward"], curr_block.mlp)
    if hasattr(curr_block, "attn"):
        curr_block.attn.forward = types.MethodType(fwd_method_dict["org_attn_forward"], curr_block.attn)
    elif hasattr(curr_block, "attention"):
        curr_block.attention.forward = types.MethodType(fwd_method_dict["org_attn_forward"], curr_block.attention)
    else:
        import pdb; pdb.set_trace()
        
    # remove previous used values
    dellist = [k for k in curr_block.__dict__.keys() if "anonymoususer" in k]
    _ = [ delattr(curr_block, k) for k in dellist]
    dellist = [k for k in curr_block.mlp.__dict__.keys() if "anonymoususer" in k]
    _ = [ delattr(curr_block.mlp, k) for k in dellist]
    
    if hasattr(curr_block, "attn"):
        dellist = [k for k in curr_block.attn.__dict__.keys() if "anonymoususer" in k]
        _ = [ delattr(curr_block.attn, k) for k in dellist]
    elif hasattr(curr_block, "attention"):
        dellist = [k for k in curr_block.attention.__dict__.keys() if "anonymoususer" in k]
        _ = [ delattr(curr_block.attention, k) for k in dellist]
    else:
        import pdb; pdb.set_trace()
    ### INITIALIZE
    return curr_block

def get_detailed_corrupted_flow_tracer_one_block(
        args, curr_meta_arg, flow_tracer, detailed_flow_tracer_idx):
    fwd_method_dict = return_forward_method_dict(args)

    # just for saving corrupted features
    detailed_flow_tracer = CausalFlowTracer(args)
    detailed_flow_tracer.init_node()

    # Noise RUN
    for bidx in range(curr_meta_arg["total_block_num"]):
        if hasattr(curr_meta_arg["mt"].model, "transformer"):
            curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
        elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
            curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
        curr_block = block_initializer(args, curr_block)

        # change target function
        if bidx < detailed_flow_tracer_idx-1:
            curr_block.forward = types.MethodType(fwd_method_dict["pass_block_forward"], curr_block)
            curr_block.anonymoususer_feats_normal = flow_tracer.feats_normal[bidx]
            if flow_tracer.feats_k_normal is not None:
                curr_block.anonymoususer_feats_k_normal = flow_tracer.feats_k_normal[bidx]
            if flow_tracer.feats_k_normal is not None:
                curr_block.anonymoususer_feats_v_normal = flow_tracer.feats_v_normal[bidx]
        elif bidx == detailed_flow_tracer_idx-1:
            # prepare corrupted features
            curr_block.forward = types.MethodType(fwd_method_dict["pass_corrupted_block_forward"], curr_block)          

            curr_block.anonymoususer_feats_corrupted = flow_tracer.feats_corrupted[bidx]
            if flow_tracer.feats_k_corrupted is not None:
                curr_block.anonymoususer_feats_k_corrupted = flow_tracer.feats_k_corrupted[bidx]
            if flow_tracer.feats_v_corrupted is not None:
                curr_block.anonymoususer_feats_v_corrupted = flow_tracer.feats_v_corrupted[bidx]
        elif bidx == detailed_flow_tracer_idx:
            if bidx == 0: 
                # Since the process of preparing the corrupted feature is missing when bidx==0, replace it with this line instead.
                curr_block.anonymoususer_feats_corrupted_init = flow_tracer.feats_corrupted_init

            # save corrupted features
            curr_block.forward = types.MethodType(fwd_method_dict["intervention_forward"], curr_block)
            curr_block.mlp.forward = types.MethodType(fwd_method_dict["custom_mlp_forward"], curr_block.mlp)
            if hasattr(curr_block, "attn"):
                curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)
            elif hasattr(curr_block, "attention"):
                curr_block.attention.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attention)

            curr_block.anonymoususer_flow_tracer = detailed_flow_tracer
            curr_block.anonymoususer_save_mode = True
        else:
            curr_block.forward = types.MethodType(fwd_method_dict["pass_block_forward"], curr_block)

            curr_block.anonymoususer_feats_normal = flow_tracer.feats_normal[bidx]
            if flow_tracer.feats_k_normal is not None:
                curr_block.anonymoususer_feats_k_normal = flow_tracer.feats_k_normal[bidx]
            if flow_tracer.feats_v_normal is not None:
                curr_block.anonymoususer_feats_v_normal = flow_tracer.feats_v_normal[bidx]

    with torch.no_grad():
        _, _, _ = predict_from_input(curr_meta_arg["mt"].model, curr_meta_arg["inp"], multipred=(args.out_num!=1), end_symbol=args.end_symbol)

    # Restore orignal module
    for bidx in range(curr_meta_arg["total_block_num"]):
        if hasattr(curr_meta_arg["mt"].model, "transformer"):
            curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
        elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
            curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
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
            if hasattr(curr_meta_arg["mt"].model, "transformer"):
                curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
            elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
                curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
            curr_block = block_initializer(args, curr_block)
            
            fwd_method_dict = return_forward_method_dict(args)

            # change target function
            if bidx < curr_meta_arg["curr_block_idx"]:
                curr_block.forward = types.MethodType(fwd_method_dict['pass_block_forward'], curr_block)
                curr_block.anonymoususer_feats_normal = flow_tracer.feats_normal[bidx]
                if flow_tracer.feats_k_normal is not None:
                    curr_block.anonymoususer_feats_k_normal = flow_tracer.feats_k_normal[bidx]
                if flow_tracer.feats_v_normal is not None:
                    curr_block.anonymoususer_feats_v_normal = flow_tracer.feats_v_normal[bidx]
            elif bidx == curr_meta_arg["curr_block_idx"]:
                curr_block.forward = types.MethodType(fwd_method_dict['intervention_forward'], curr_block)
                curr_block.mlp.forward = types.MethodType(fwd_method_dict['custom_mlp_forward'], curr_block.mlp)
                if hasattr(curr_block, "attn"):
                    curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)
                elif hasattr(curr_block, "attention"):
                    curr_block.attention.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attention)

                curr_block.anonymoususer_trace_mode = True
                curr_block.anonymoususer_cond = cond
                curr_block.anonymoususer_corrupted_feats = curr_meta_arg["detailed_flow_tracer"][bidx]["corrupted"]
                curr_block.anonymoususer_curr_subset = curr_meta_arg["curr_subset"]
            elif bidx > curr_meta_arg["curr_block_idx"]:
                curr_block.forward = types.MethodType(fwd_method_dict['intervention_forward'], curr_block)
                curr_block.mlp.forward = types.MethodType(fwd_method_dict['custom_mlp_forward'], curr_block.mlp)
                if hasattr(curr_block, "attn"):
                    curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)
                elif hasattr(curr_block, "attention"):
                    curr_block.attention.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attention)

                curr_block.anonymoususer_trace_mode = True
                curr_block.anonymoususer_cond = "path"
                curr_block.anonymoususer_corrupted_feats = curr_meta_arg["detailed_flow_tracer"][bidx]["corrupted"]
                curr_block.anonymoususer_curr_subset = tuple(set().union(*flow_tracer.traced_paths[bidx]))

        with torch.no_grad():
            curr_pred, curr_p, raw_out = predict_from_input(
                curr_meta_arg["mt"].model, curr_meta_arg["inp"], 
                multipred=(args.out_num!=1), end_symbol=args.end_symbol, use_mean=True, stwd_mask=flow_tracer.stwd_mask)
        
        out = curr_meta_arg["mt"].model(**curr_meta_arg["inp"])["logits"]
        probs1 = torch.softmax(out[:, -1], dim=1).mean(dim=0).unsqueeze(0)
        probs2 = torch.softmax(out[:, -1], dim=1)[0:1]
        
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

    if args.except_stopword is False:
        normal_p, normal_pred = torch.max(flow_tracer.scores_normal, dim=0)
    else:
        desc_idx = torch.argsort(flow_tracer.scores_normal, dim=0, descending=True)
        sorted_stwd_mask = flow_tracer.stwd_mask[desc_idx]
        normal_pred = desc_idx[sorted_stwd_mask][0]
        normal_p = flow_tracer.scores_normal[normal_pred]

    curr_meta_arg["normal_p"] = normal_p
    curr_meta_arg["normal_pred"] = normal_pred
    if hasattr(curr_meta_arg['mt'].model.config, "use_parallel_residual"):
        curr_meta_arg["num_path_node"] = curr_meta_arg['mt'].model.config.num_attention_heads + 2
    else:
        curr_meta_arg["num_path_node"] = curr_meta_arg['mt'].model.config.n_head *2 + 2

    Xs_index = np.arange(curr_meta_arg["num_path_node"])
    min_search_memory = {}
    selected_subset = []
    step_subset = {}
    step_subset_cond_bool = {}
    stop_step = curr_meta_arg["num_path_node"]
    
    if args.debug_min_search:
        curr_meta_arg["logger"].info("\t [NOTE] This is the debug mode of Min. Search. It will pass 1~{} steps".format((curr_meta_arg["num_path_node"]-2)-1))

    minsear_iter = sorted(np.arange(curr_meta_arg["num_path_node"])+1)
    for curr_step in minsear_iter:
        if args.debug_min_search:
            # if curr_step < (curr_meta_arg["num_path_node"]-2):
            if curr_step < (curr_meta_arg["num_path_node"]):
                continue

        curr_step_subset = list(itertools.combinations(Xs_index, curr_step))

        if args.subset_search == "minimality":
            subset_candidate = exclude_subsets(curr_step_subset, selected_subset)
        else:
            subset_candidate = copy.deepcopy(curr_step_subset)
        
        if len(subset_candidate) == 0:
            stop_step = curr_step-1
        
        buffer_cond_bool = []
        if len(subset_candidate) != 0:
            curr_meta_arg["logger"].info("\t MinSearch Step: ({}/{}) -> Target Num.: {}".format(curr_step, max(minsear_iter), len(subset_candidate) ))
        curr_meta_arg["curr_step"] = curr_step
        
        if args.detailed_save:
            save_curr_step_causal_decision = {}
            save_curr_step_causal_subset = {}
            save_curr_step_causal_index = 0
        for subidx, curr_subset in enumerate(subset_candidate):
            curr_meta_arg["curr_subset"] = curr_subset

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
        stop_flag = True
        

    min_search_memory["selected_subset"] = selected_subset
    min_search_memory["stop_step"] = stop_step
    min_search_memory["step_subset"] = step_subset
    min_search_memory["step_subset_cond_bool"] = step_subset_cond_bool

    # Restore orignal module
    for bidx in range(curr_meta_arg["total_block_num"]):
        if hasattr(curr_meta_arg["mt"].model, "transformer"):
            curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
        elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
            curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
        curr_block = block_initializer(args, curr_block)

    return min_search_memory, stop_flag

def top_down_causal_flow_trace(
    args, curr_meta_arg, flow_tracer):
    
    inp = make_inputs(
        curr_meta_arg["mt"].tokenizer, [curr_meta_arg["prompt"]] * (args.num_noise_sample))
    
    curr_meta_arg["inp"] = inp

    stop_flag = False
    for curr_block_idx in tqdm.tqdm(range(curr_meta_arg["total_block_num"]-1, -1, -1)):
        start_block_time = time.time()
        curr_meta_arg["curr_block_idx"] = curr_block_idx
        curr_meta_arg["logger"].info("-----------Block Change----------- Block:{}".format( curr_meta_arg["curr_block_idx"]))
        
        min_search_memory, stop_flag = minimality_search(args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer)
        if stop_flag:
            curr_meta_arg["logger"].info("[STOP] Stopping early at block {} due to empty selection".format(curr_block_idx))
            break
        else:
            flow_tracer.traced_paths.update({curr_block_idx: copy.deepcopy(min_search_memory["selected_subset"])})
       

    return stop_flag


################################################################################
# path_forward (CausalPathTracing_anonymous/CPT/language/path_forward.py)
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
    if hasattr(mt.model.config, "use_parallel_residual"):
        return mt.model.config.num_attention_heads + 2
    else:
        return mt.model.config.n_head * 2 + 2


def path_forward(args, mt, n_lev, prompt, y, causal_path, interv_target="off_path"):
    """
    off_path: keep the causal path nodes and intervene (corrupt) all the other nodes.
    on_path : intervene the causal path nodes.
    causal_path: {block_idx(int): [node, ...]} (aggregate_paths output)
    Returns org/path decision and outputs (sorted by the original prediction order).
    """
    flow_tracer = CausalFlowTracer(args)
    flow_tracer, answer, curr_total_token_num, curr_total_block_num, normal_inp = predict_from_normal_and_noise_input(
        prompt, flow_tracer, y, None,
        mt, n_lev, end_symbol=[],
        out_num=1, num_normal_sample=args.num_noise_sample,
        num_noise_sample=args.num_noise_sample, noise_type=args.noise_type)

    curr_meta_arg = {}
    curr_meta_arg["total_block_num"] = curr_total_block_num
    curr_meta_arg["total_token_num"] = curr_total_token_num
    curr_meta_arg["prompt"] = prompt
    curr_meta_arg["mt"] = mt
    inp = make_inputs(
        curr_meta_arg["mt"].tokenizer, [curr_meta_arg["prompt"]] * (args.num_noise_sample))

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

    if args.except_stopword is False:
        normal_p, normal_pred = torch.max(flow_tracer.scores_normal, dim=0)
        desc_idx = torch.argsort(flow_tracer.scores_normal, dim=0, descending=True)
        normal_output = torch.index_select(flow_tracer.scores_normal, 0, desc_idx)
        pred_ord = desc_idx
    else:
        desc_idx = torch.argsort(flow_tracer.scores_normal, dim=0, descending=True)
        sorted_stwd_mask = flow_tracer.stwd_mask[desc_idx]
        preds = desc_idx[sorted_stwd_mask]
        normal_pred = preds[0]
        normal_p = flow_tracer.scores_normal[normal_pred]
        normal_output = torch.index_select(flow_tracer.scores_normal, 0, preds)
        pred_ord = preds

    curr_meta_arg["normal_p"] = normal_p
    curr_meta_arg["normal_pred"] = normal_pred
    curr_meta_arg["normal_output"] = normal_output
    curr_meta_arg["normal_pred_order"] = pred_ord
    curr_meta_arg["num_path_node"] = get_num_path_node(curr_meta_arg['mt'])

    for bidx in range(curr_meta_arg["total_block_num"]):
        if hasattr(curr_meta_arg["mt"].model, "transformer"):
            curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
        elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
            curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
        curr_block = block_initializer(args, curr_block)

        fwd_method_dict = return_forward_method_dict(args)

        curr_block.forward = types.MethodType(fwd_method_dict['intervention_forward'], curr_block)
        curr_block.mlp.forward = types.MethodType(fwd_method_dict['custom_mlp_forward'], curr_block.mlp)
        if hasattr(curr_block, "attn"):
            curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)
        elif hasattr(curr_block, "attention"):
            curr_block.attention.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attention)

        curr_block.anonymoususer_trace_mode = True
        curr_block.anonymoususer_cond = "path"
        curr_block.anonymoususer_corrupted_feats = curr_meta_arg["detailed_flow_tracer"][bidx]["corrupted"]

        if interv_target=="off_path":
            curr_block.anonymoususer_curr_subset = tuple(causal_path[bidx])
        elif interv_target=="on_path":
            curr_block.anonymoususer_curr_subset = tuple(np.setdiff1d(np.arange(curr_meta_arg["num_path_node"]), causal_path[bidx]).tolist())

    with torch.no_grad():
        curr_pred, curr_p, raw_out = predict_from_input(
            curr_meta_arg["mt"].model, curr_meta_arg["inp"],
            multipred=(args.out_num!=1), end_symbol=[], use_mean=True, stwd_mask=flow_tracer.stwd_mask)
        raw_out_sftmx = torch.softmax(raw_out[:, -1], dim=1).mean(dim=0).unsqueeze(0)
        desc_idx = torch.argsort(raw_out_sftmx, dim=1, descending=True)

        if args.except_stopword is False:
            output = torch.index_select(raw_out_sftmx, 1, desc_idx.squeeze(0)).squeeze(0)
            pred_order = desc_idx.squeeze(0)
        else:
            sorted_stwd_mask = flow_tracer.stwd_mask[desc_idx]
            preds = desc_idx[sorted_stwd_mask]
            output = torch.index_select(raw_out_sftmx, 1, preds).squeeze(0)

            pred_order = preds

    for bidx in range(curr_meta_arg["total_block_num"]):
        if hasattr(curr_meta_arg["mt"].model, "transformer"):
            curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
        elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
            curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
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
