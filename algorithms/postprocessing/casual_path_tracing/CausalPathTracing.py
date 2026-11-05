import os
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import copy
import json
import logging
import random
import tqdm
import torch
import numpy as np
from .util.cpt_language_function import (
    arg_func, build_arg_list, set_utils, get_model, get_noise_level, make_inputs,
    predict_from_normal_and_noise_input, CausalFlowTracer, top_down_causal_flow_trace,
    aggregate_paths, get_num_path_node, path_forward
)
from MAF.datamodule.casual_path_tracing.sample import KnownsDataset
from MAF.datamodule.casual_path_tracing.download_data import ensure_model, ensure_stopwords
from MAF.metric.casual_path_tracing import get_metric as common_utils

class CausalPathTracing:
    """Causal Path Tracing (CPT) for a language model (pythia-14m) on known_1000."""
    def __init__(self, dataset_name='known1000', model='pythia-14m', gpu='0', noise_type='other',
                 save_root=None, num_noise_sample=100, path_agg_type='union', arg_list=None, offline=False, **kwargs):
        self.dataset_name = dataset_name
        self.path_agg_type = path_agg_type  # aggregation of the traced paths in a block (original path_forward.py default: union)
        if arg_list is None:
            cpt_kwargs = dict(efficient_mode=True, slightly_quiet_mode=True, detailed_save=True, except_stopword=True)
            cpt_kwargs.update(kwargs)
            if save_root is not None:
                cpt_kwargs['save_root'] = save_root
            arg_list = build_arg_list(model=model, gpu=gpu, noise_type=noise_type,
                                      num_noise_sample=num_noise_sample, **cpt_kwargs)
        self.args = arg_func(arg_list)
        self.args.model = str(ensure_model(self.args.model, offline=offline))
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self.set_seed()
        self.logger, self.func_time_saver = set_utils(self.args)
        ensure_stopwords()
        self.load_and_preprocess_data()

    def set_seed(self, random_seed=0):
        torch.set_grad_enabled(False)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    def load_and_preprocess_data(self):
        if self.dataset_name == 'known1000':
            self.knowns = KnownsDataset(reverse=self.args.reverse)
            self.dataset = self.knowns.to_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.line_data = self.dataset['line_data']
        self.target_ids = [li for li in range(len(self.line_data)) if self.is_target(li)]

        self.mt = get_model(self.args)
        self.n_lev = get_noise_level(self.args, self.mt)
        self.func_time_saver.logging(self.logger, header="\n\n===============Prepare Function Log===============")

    def is_target(self, li):
        args = self.args
        if args.target_sample_index_list is not None:
            if li not in [int(i) for i in args.target_sample_index_list.split(", ")]:
                return False
        if args.target_sample_index is not None:
            spt = args.target_sample_index.split("_")
            if li < int(spt[0]):
                return False
            if spt[1] != "-":
                if li > int(spt[1]):
                    return False
        if args.debug_index is not None:
            if li != args.debug_index:
                return False
        return True

    def correct_check(self):
        args = self.args
        self.logger.info("[Correct Check Start]")
        self.func_time_saver.logging(self.logger, header="\n\n===============Correct Check Log===============")
        correct_ids = []
        for li, statement in tqdm.tqdm(enumerate(self.line_data), desc="Correct Check First"):
            flow_tracer = CausalFlowTracer(args)
            correctness = predict_from_normal_and_noise_input(
                statement["prompt"], flow_tracer, statement["attribute"], self.logger,
                self.mt, self.n_lev, end_symbol=args.end_symbol,
                out_num=1, num_normal_sample=args.num_noise_sample,
                num_noise_sample=args.num_noise_sample, noise_type=args.noise_type, correct_check_only=True)
            if correctness:
                correct_ids.append(li)
        self.logger.info("\t Correct Data Num.: {}".format(len(correct_ids)))
        np.savetxt(os.path.join(args.save_root, "correct_data_idx.txt"), np.asarray(correct_ids), fmt="%s")
        return correct_ids

    def baseline_fit(self):
        """Original model (all path nodes kept, no intervention) on the target samples."""
        args = self.args
        # keep the RNG state so that cpt_fit gives the same traced paths with or without baseline_fit
        rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())
        num_total_node = get_num_path_node(self.mt) * self.mt.num_layers
        baseline_results = {}
        for li in self.target_ids:
            flow_tracer = CausalFlowTracer(args)
            normal_inp = make_inputs(self.mt.tokenizer, [self.line_data[li]["prompt"]] * (args.num_noise_sample))
            org_decision = flow_tracer.trace_normal(self.mt.model, normal_inp).item()
            baseline_results[li] = {"num_selected_node": num_total_node, "num_total_node": num_total_node,
                                    "org_decision": org_decision, "path_decision": org_decision}
        torch.set_rng_state(rng_state[0])
        torch.cuda.set_rng_state_all(rng_state[1])
        np.random.set_state(rng_state[2])
        random.setstate(rng_state[3])
        return baseline_results

    def cpt_fit(self, li, statement):
        """Trace causal paths of one statement. Returns {block_idx: [path, ...]} or None."""
        args = self.args
        prompt = statement["prompt"]
        y = statement["attribute"]

        flow_tracer = CausalFlowTracer(args)
        flow_tracer, answer, curr_total_token_num, curr_total_block_num, normal_inp = predict_from_normal_and_noise_input(
            prompt, flow_tracer, y, self.logger,
            self.mt, self.n_lev, end_symbol=args.end_symbol,
            out_num=1, num_normal_sample=args.num_noise_sample,
            num_noise_sample=args.num_noise_sample, noise_type=args.noise_type)

        if (isinstance(answer, list) is False):
            if answer == -1:
                self.logger.info("--------------------")
                self.logger.info("[Incorrect Answer Pass] Index:{}, Prompt:{}".format(li, prompt))
                return None

        self.logger.info("--------------------")
        self.logger.info("[Searching Start] Index:{}, Prompt:{}, Ans:{}".format(li, prompt, y))
        args.curr_save_dir = os.path.join(args.save_result_root, "R{:04d}".format(li))
        os.makedirs(args.curr_save_dir, exist_ok=True)

        curr_meta_arg = {}
        curr_meta_arg["total_block_num"] = curr_total_block_num
        curr_meta_arg["total_token_num"] = curr_total_token_num
        curr_meta_arg["prompt"] = prompt
        curr_meta_arg["mt"] = self.mt
        curr_meta_arg["logger"] = self.logger

        stop_flag = top_down_causal_flow_trace(
            args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer)
        if stop_flag is not False:
            return None

        success_logger = ["li:{}\nprompt:{}\ny:{}".format(li, prompt, y)]
        np.savetxt(os.path.join(args.save_inpinfo_root, "I{:06d}.txt".format(li)), success_logger, fmt="%s")
        traced_paths = {
            str(k): [[int(i) for i in t] for t in v]
            for k, v in flow_tracer.traced_paths.items()
        }
        self.save_traced_paths(traced_paths, os.path.join(args.curr_save_dir, "C{:06d}.json".format(li)))
        return traced_paths

    def save_traced_paths(self, json_data, save_path):
        with open(save_path, "w") as f:
            f.write("{\n")
            for idx, (k, paths) in enumerate(json_data.items()):
                f.write(f'  "{k}": [\n')
                for i, p in enumerate(paths):
                    comma = "," if i < len(paths) - 1 else ""
                    f.write(f"    {p}{comma}\n")
                f.write("  ]" + ("," if idx < len(json_data) - 1 else "") + "\n")
            f.write("}\n")

    def evaluate(self, li, traced_paths):
        """Keep only the causal path nodes, intervene the others, and check the decision (original path_forward.py, off_path)."""
        # same settings as the original path_forward.py
        eval_args = copy.copy(self.args)
        eval_args.out_num = 1
        eval_args.end_symbol = []

        causal_path = aggregate_paths(traced_paths, self.path_agg_type)
        statement = self.line_data[li]
        result = path_forward(eval_args, self.mt, self.n_lev, statement["prompt"], statement["attribute"],
                              causal_path, interv_target="off_path")
        return {"num_selected_node": sum(len(nodes) for nodes in causal_path.values()),
                "num_total_node": result["num_path_node"] * result["total_block_num"],
                "org_decision": result["org_decision"], "path_decision": result["path_decision"]}

    def save_eval_results(self, eval_results, metrics_orig, metrics_causal_path):
        save_dir = os.path.join(self.args.save_root, "path_forward_results")
        os.makedirs(save_dir, exist_ok=True)
        sample_wise = {str(k): dict(v, path_node_ratio=v["num_selected_node"] / v["num_total_node"],
                                    decision_kept=v["org_decision"] == v["path_decision"])
                       for k, v in eval_results.items()}
        with open(os.path.join(save_dir, "sample_wise_{}.json".format(self.path_agg_type)), "w") as f:
            json.dump(sample_wise, f, indent=4)
        with open(os.path.join(save_dir, "metrics_{}.json".format(self.path_agg_type)), "w") as f:
            json.dump({"original": metrics_orig, "causal_path": metrics_causal_path}, f, indent=4)

    def compute_metrics(self, eval_results):
        return common_utils.compute_metrics(eval_results)

    def run(self):
        baseline_results = self.baseline_fit()

        if self.args.correct_check_first:
            self.correct_check()

        self.traced_results = {}
        for li in tqdm.tqdm(self.target_ids):
            traced_paths = self.cpt_fit(li, self.line_data[li])
            if traced_paths is not None:
                self.traced_results[li] = traced_paths

        self.eval_results = {}
        for li, traced_paths in tqdm.tqdm(self.traced_results.items(), desc="Path Forward"):
            self.eval_results[li] = self.evaluate(li, traced_paths)

        # evaluated on the traced samples only
        metrics_orig = self.compute_metrics({li: baseline_results[li] for li in self.eval_results})
        metrics_causal_path = self.compute_metrics(self.eval_results)
        self.save_eval_results(self.eval_results, metrics_orig, metrics_causal_path)
        return metrics_orig, metrics_causal_path


def mitigate_cpt():
    """MAF-DEMO용: 기본 호출(샘플 16·204)로 인과 경로만 남겼을 때의 두 지표를 돌려준다.

    장기 프로세스(DEMO)에서 불리므로 run()이 바꾼 grad 모드와 root logger 핸들러를 되돌린다.
    """
    root_logger = logging.getLogger()
    handlers_before = list(root_logger.handlers)
    grad_before = torch.is_grad_enabled()
    try:
        cpt = CausalPathTracing(dataset_name='known1000', model='pythia-14m', gpu='0',
                                target_sample_index_list='16, 204')
        _, metrics_causal_path = cpt.run()
    finally:
        torch.set_grad_enabled(grad_before)
        for handler in list(root_logger.handlers):
            if handler not in handlers_before:
                root_logger.removeHandler(handler)
                handler.close()
    return {"path_node_ratio": metrics_causal_path["path_node_ratio"],
            "decision_kept_ratio": metrics_causal_path["decision_kept_ratio"]}


def main(argv=None):
    # CLI flags are the same as the original CPT/language/main.py (see scripts/run_*.sh)
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv:
            cpt = CausalPathTracing(dataset_name='known1000', arg_list=argv)
        else:
            cpt = CausalPathTracing(dataset_name='known1000', model='pythia-14m', gpu='0', target_sample_index_list='16, 204')
        metrics_orig, metrics_causal_path = cpt.run()
    except (ValueError, OSError) as error:  # FileNotFoundError는 OSError
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for causal path only:")
    print(metrics_causal_path)
    print("\nTraced paths saved to: {}".format(cpt.args.save_result_root))
    print("Sample-wise evaluation saved to: {}".format(os.path.join(cpt.args.save_root, "path_forward_results")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
