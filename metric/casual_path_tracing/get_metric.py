from collections import OrderedDict
import numpy as np

def compute_metrics(eval_results,
                    disp = True):
    """ Compute the key metrics

    eval_results : {sample_idx: {"num_selected_node": int, "num_total_node": int,
                                 "org_decision": int, "path_decision": int}}
        num_selected_node : causal path nodes (union of the traced paths in each block, summed over blocks)
        num_total_node    : all path nodes (num_path_node x total_block_num)
        org_decision      : prediction of the original model
        path_decision     : prediction when only the causal path nodes are kept and the others are intervened
    """
    path_node_ratio = [r["num_selected_node"] / r["num_total_node"] for r in eval_results.values()]
    decision_kept = [r["org_decision"] == r["path_decision"] for r in eval_results.values()]

    metrics = OrderedDict()
    metrics["num_samples"] = len(eval_results)
    # sparsity: selected causal path nodes / all path nodes (sample-wise, averaged)
    metrics["path_node_ratio"] = float(np.mean(path_node_ratio)) if path_node_ratio else 0.0
    # faithfulness: decision is kept with the causal path only (= hit_rate of the original path_forward.py)
    metrics["decision_kept_ratio"] = float(np.mean(decision_kept)) if decision_kept else 0.0

    # if disp:
    #     for k in metrics:
    #         print("%s = %.4f" % (k, metrics[k]))

    return metrics
