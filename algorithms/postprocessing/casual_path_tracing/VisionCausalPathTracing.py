import os
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import copy
import json
import logging
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from .util.cpt_vision_function import (
    arg_func, build_arg_list, set_utils, get_model, DATA_ROOT,
    predict_from_normal_and_noise_input, CausalFlowTracer, top_down_causal_flow_trace,
    aggregate_paths, get_num_path_node, path_forward
)
from MAF.datamodule.casual_path_tracing.sample import ImageNetDataset, IMAGENET_ROOT, MAF_ROOT
from MAF.metric.casual_path_tracing import get_metric as common_utils

class VisionCausalPathTracing:
    """Causal Path Tracing (CPT) for a vision transformer (timm vit_tiny_patch16_224) on ImageNet-1k val."""
    def __init__(self, dataset_name='imagenet', model='vit_tiny_patch16_224', gpu='0', noise_type='other',
                 save_root=None, num_noise_sample=100, target_sample_idx=0,
                 data_root=IMAGENET_ROOT, path_agg_type='union', arg_list=None, **kwargs):
        self.dataset_name = dataset_name
        self.path_agg_type = path_agg_type  # aggregation of the traced paths in a block (original path_forward.py default: union)
        if arg_list is None:
            cpt_kwargs = dict(efficient_mode=True, slightly_quiet_mode=True, detailed_save=True)
            cpt_kwargs.update(kwargs)
            if save_root is not None:
                cpt_kwargs['save_root'] = save_root
            arg_list = build_arg_list(model=model, gpu=gpu, noise_type=noise_type,
                                      num_noise_sample=num_noise_sample, dataset=dataset_name, **cpt_kwargs)
        self.args = arg_func(arg_list)
        self.data_root = self.args.data_root if self.args.data_root is not None else data_root
        # int or list of ImageNet val indices (--target_sample_idx from CLI takes priority)
        if self.args.target_sample_idx is not None:
            target_sample_idx = self.args.target_sample_idx
        self.target_sample_idx_list = target_sample_idx if isinstance(target_sample_idx, (list, tuple)) else [target_sample_idx]
        self.set_seed()
        self.logger, self.func_time_saver = set_utils(self.args)
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
        if self.dataset_name == 'imagenet':
            DATA_ROOT[self.dataset_name] = self.data_root  # also used inside CausalFlowTracer.trace_corrupted
            self.imagenet = ImageNetDataset(root=self.data_root, split='val')
            self.dataset = self.imagenet.to_dataset()
            self.num_classes = self.dataset['num_classes']
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.mt = get_model(model_name=self.args.model, num_classes=self.num_classes, dataset_name=self.args.dataset)
        self.func_time_saver.logging(self.logger, header="\n\n===============Prepare Function Log===============")

    def load_sample(self, sample_idx):
        images, labels = next(iter(DataLoader(Subset(self.dataset['dataset'], [sample_idx]), batch_size=1, shuffle=False)))
        return images.cuda(), labels.cuda()

    def baseline_fit(self):
        """Original model (all path nodes kept, no intervention) on the target samples."""
        args = self.args
        # keep the RNG state so that cpt_fit gives the same traced paths with or without baseline_fit
        rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())
        num_total_node = get_num_path_node(self.mt) * len(self.mt.model.blocks)
        baseline_results = {}
        for sample_idx in self.target_sample_idx_list:
            images, labels = self.load_sample(sample_idx)
            flow_tracer = CausalFlowTracer()
            normal_inp = {"pixel_values": images.repeat(args.num_noise_sample, 1, 1, 1)}
            org_decision = flow_tracer.trace_normal(self.mt.model, normal_inp, args.model).item()
            baseline_results[sample_idx] = {"num_selected_node": num_total_node, "num_total_node": num_total_node,
                                            "org_decision": org_decision, "path_decision": org_decision}
        torch.set_rng_state(rng_state[0])
        torch.cuda.set_rng_state_all(rng_state[1])
        np.random.set_state(rng_state[2])
        random.setstate(rng_state[3])
        return baseline_results

    def cpt_fit(self, sample_idx):
        """Trace causal paths of one image. Returns {block_idx: [path, ...]} or None."""
        args = self.args
        args.target_sample_idx = sample_idx
        images, labels = self.load_sample(sample_idx)
        flow_tracer = CausalFlowTracer()
        flow_tracer, answer, curr_total_token_num, curr_total_block_num, normal_inp = predict_from_normal_and_noise_input(
            images, flow_tracer, labels, self.logger,
            self.mt, n_lev=None, end_symbol=None, out_num=1,
            model_name=args.model,
            dataset_name=args.dataset,
            num_normal_sample=args.num_noise_sample,
            num_noise_sample=args.num_noise_sample,
            noise_type=args.noise_type)

        if (isinstance(answer, list) is False):
            if answer == -1:
                self.logger.info("--------------------")
                self.logger.info("[Incorrect Answer Pass] Index:{}, Label:{}".format(sample_idx, labels[0].item()))
                return None

        self.logger.info("--------------------")
        self.logger.info("[Searching Start] Index:{}, Label:{}, Pred:{}".format(sample_idx, labels[0].item(), answer))

        args.curr_save_dir = os.path.join(args.save_result_root, "R{:04d}".format(sample_idx))
        os.makedirs(args.curr_save_dir, exist_ok=True)

        curr_meta_arg = {}
        curr_meta_arg["total_block_num"] = curr_total_block_num
        curr_meta_arg["total_token_num"] = curr_total_token_num
        curr_meta_arg["images"] = images
        curr_meta_arg["mt"] = self.mt
        curr_meta_arg["logger"] = self.logger

        stop_flag = top_down_causal_flow_trace(
            args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer)
        if stop_flag is not False:
            return None

        success_logger = ["idx:{}\nlabel:{}\npred:{}".format(sample_idx, labels[0].item(), answer[0])]
        np.savetxt(os.path.join(args.save_inpinfo_root, "I{:06d}.txt".format(sample_idx)), success_logger, fmt="%s")
        traced_paths = {
            str(k): [[int(i) for i in t] for t in v]
            for k, v in flow_tracer.traced_paths.items()
        }
        self.save_traced_paths(traced_paths, os.path.join(args.curr_save_dir, "C{:06d}.json".format(sample_idx)))
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

    def evaluate(self, sample_idx, traced_paths):
        """Keep only the causal path nodes, intervene the others, and check the decision (original path_forward.py, off_path)."""
        # same settings as the original path_forward.py
        eval_args = copy.copy(self.args)
        eval_args.out_num = 1
        eval_args.end_symbol = []

        causal_path = aggregate_paths(traced_paths, self.path_agg_type)
        images, labels = self.load_sample(sample_idx)
        result = path_forward(eval_args, self.mt, images, labels, causal_path, interv_target="off_path")
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

    def class_info(self, class_idx):
        """ImageNet-1k 클래스 인덱스 → (wnid, 사람이 읽는 이름); timm 메타데이터(오프라인) 사용."""
        from timm.data import ImageNetInfo
        info = ImageNetInfo()
        return info.index_to_label_name(class_idx), info.index_to_description(class_idx)

    def predict_sample(self, sample_idx):
        """원래 모델의 top-1 클래스와 라벨 일치 여부 — cpt_fit 의 correct check(out_num=1)와 같은 기준."""
        images, labels = self.load_sample(sample_idx)
        inp = {"pixel_values": images.repeat(self.args.num_noise_sample, 1, 1, 1)}
        with torch.no_grad():
            answer = CausalFlowTracer().trace_normal(self.mt.model, inp, self.args.model)
        pred = int(answer.reshape(-1)[0].item())
        label = int(labels[0].item())
        label_wnid, label_name = self.class_info(label)
        pred_wnid, pred_name = self.class_info(pred)
        return {"index": sample_idx, "label": label, "label_wnid": label_wnid, "label_name": label_name,
                "prediction": pred, "prediction_wnid": pred_wnid, "prediction_name": pred_name,
                "correct": pred == label}

    def sample_report(self, sample_idx):
        """DEMO용 샘플 1건 요약: 예측·라벨과, 추적됐으면 경로 지표·경로만 남긴 클래스."""
        report = dict(self.predict_sample(sample_idx), traced=False, path_node_ratio=None,
                      decision_kept=None, path_prediction=None, path_prediction_name=None)
        r = getattr(self, "eval_results", {}).get(sample_idx)
        if r is not None:
            _, path_name = self.class_info(int(r["path_decision"]))
            report.update(traced=True, path_node_ratio=r["num_selected_node"] / r["num_total_node"],
                          decision_kept=r["org_decision"] == r["path_decision"],
                          path_prediction=int(r["path_decision"]), path_prediction_name=path_name)
        return report

    def run(self):
        baseline_results = self.baseline_fit()

        self.traced_results = {}
        for sample_idx in self.target_sample_idx_list:
            traced_paths = self.cpt_fit(sample_idx)
            if traced_paths is not None:
                self.traced_results[sample_idx] = traced_paths

        self.eval_results = {}
        for sample_idx, traced_paths in self.traced_results.items():
            self.eval_results[sample_idx] = self.evaluate(sample_idx, traced_paths)

        # evaluated on the traced samples only
        metrics_orig = self.compute_metrics({idx: baseline_results[idx] for idx in self.eval_results})
        metrics_causal_path = self.compute_metrics(self.eval_results)
        self.save_eval_results(self.eval_results, metrics_orig, metrics_causal_path)
        return metrics_orig, metrics_causal_path


DEFAULT_VISION_SAMPLE = 5050
IMAGENET_VAL_SIZE = 50000
_PREDICT_SAVE_ROOT = os.path.join(str(MAF_ROOT), "result", "casual_path_tracing", "predict")


def _check_index(sample_idx):
    if type(sample_idx) is not int or not 0 <= sample_idx < IMAGENET_VAL_SIZE:
        raise ValueError(f"sample_idx must be an integer in 0..{IMAGENET_VAL_SIZE - 1}: {sample_idx!r}")


def _restoring_process_state(fn):
    """DEMO 같은 장기 프로세스에서 불리므로 grad 모드와 root logger 핸들러를 되돌린다."""
    root_logger = logging.getLogger()
    handlers_before = list(root_logger.handlers)
    grad_before = torch.is_grad_enabled()
    try:
        return fn()
    finally:
        torch.set_grad_enabled(grad_before)
        for handler in list(root_logger.handlers):
            if handler not in handlers_before:
                root_logger.removeHandler(handler)
                handler.close()


def predict_vision_cpt(sample_idx):
    """MAF-DEMO용(수 초): ImageNet val 샘플 1개의 top-1 예측·라벨·정답 여부. 결과 디렉터리는 predict_* 로 분리."""
    _check_index(sample_idx)

    def _run():
        cpt = VisionCausalPathTracing(dataset_name='imagenet', model='vit_tiny_patch16_224', gpu='0',
                                      target_sample_idx=sample_idx, save_root=_PREDICT_SAVE_ROOT)
        return cpt.predict_sample(sample_idx)
    return _restoring_process_state(_run)


def trace_vision_cpt(sample_idx):
    """MAF-DEMO용(약 15분): 샘플 1개 인과 경로 추적 + 평가. 결과는 run_* 디렉터리에 저장된다."""
    _check_index(sample_idx)

    def _run():
        cpt = VisionCausalPathTracing(dataset_name='imagenet', model='vit_tiny_patch16_224', gpu='0',
                                      target_sample_idx=sample_idx)
        _, metrics_causal_path = cpt.run()
        samples = sorted(int(k) for k in cpt.eval_results)
        return {"path_node_ratio": metrics_causal_path["path_node_ratio"],
                "decision_kept_ratio": metrics_causal_path["decision_kept_ratio"],
                "num_samples": len(samples), "samples": samples,
                "run": os.path.basename(os.path.normpath(cpt.args.save_root)),
                "details": [cpt.sample_report(sample_idx)]}
    return _restoring_process_state(_run)


def main(argv=None):
    # CLI flags are the same as the original CPT/vision/main.py (see scripts/run_vit.sh)
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv:
            cpt = VisionCausalPathTracing(dataset_name='imagenet', arg_list=argv)
        else:
            cpt = VisionCausalPathTracing(dataset_name='imagenet', model='vit_tiny_patch16_224', gpu='0', target_sample_idx=5050)
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
