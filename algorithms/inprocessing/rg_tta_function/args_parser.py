# args_parser.py

import argparse


def get_args():
    """
    Central experiment configuration.
    This is the single source of truth for hyperparameters and experiment settings.
    Import and call this in run_tta_experiment.py (or your notebook/script)
    to ensure experiments are reproducible and logged consistently.
    """

    parser = argparse.ArgumentParser(
        description="RLCF-style TTA for CLIP social debiasing (text->image retrieval)"
    )

    # -------------------------------------------------
    # Core TTA / RL hyperparameters
    # -------------------------------------------------
    parser.add_argument(
        "--tta_steps",
        type=int,
        default=30,
        help="Number of policy gradient update steps per query (episodic TTA length).",
    )
    parser.add_argument(
        "--sample_k",
        type=int,
        default=16,
        help="Top-K candidates sampled per query to compute reward / policy gradient.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for adapting the text encoder during TTA.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help=(
            "Momentum coefficient for EMA-style carryover between episodes. "
            "0.0 = no carryover, >0 means we update the internal momentum buffer "
            "after each query."
        ),
    )
    parser.add_argument(
        "--momentum_update_freq",
        type=int,
        default=64,
        help=(
            "Frequency of momentum buffer updates. "
            "1 = update after each query, N = update every N queries. "
            "Default: 1 (update every query)."
        ),
    )

    # -------------------------------------------------
    # Dataset / task configuration
    # -------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        default="fairface",
        choices=["fairface", "utkface", "facet"],
        help="Which dataset to evaluate on.",
    )
    parser.add_argument(
        "--attribute",
        type=str,
        default="race",
        choices=["race", "gender", "age", "all"],
        help=(
            "Which attribute(s) to focus on when sampling prompts / grouping stats. "
            "'all' means use all attribute-related prompts."
        ),
    )
    parser.add_argument(
        "--gallery_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split of the dataset to embed as the retrieval gallery.",
    )

    # -------------------------------------------------
    # Reward configuration
    # -------------------------------------------------
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="clip_plus_debias",
        choices=["clip_only", "clip_plus_debias"],
        help=(
            "Which reward signal to use.\n"
            " - clip_only: only CLIPScore-based reward\n"
            " - clip_plus_debias: CLIPScore + lambda * DebiasScore"
        ),
    )
    parser.add_argument(
        "--debias_lambda",
        type=float,
        default=1.0,
        help=(
            "Weight for DebiasScore when reward_mode='clip_plus_debias'. "
            "If 0.0, debias term is effectively disabled."
        ),
    )
    parser.add_argument(
        "--debias_score_base",
        type=str,
        default="instance_popularity",
        choices=["mu_norm", "soft_alignment_l2", "soft_alignment_kl", "instance_popularity", "instance_popularity_centered"],
        help=(
            "Base method for measuring bias.\n"
            " - mu_norm: Use L2 norm of mu (traditional method)\n"
            " - soft_alignment_l2: Use L2 distance from soft alignment class distribution\n"
            " - soft_alignment_kl: Use KL divergence from soft alignment class distribution"
        ),
    )
    parser.add_argument(
        "--debias_score_trace",
        type=str,
        default="none",
        choices=["none", "numerator", "denominator"],
        help=(
            "How to handle trace_sigma in score computation.\n"
            " - none: Do not use trace_sigma (equivalent to old l2_norm)\n"
            " - numerator: Multiply by trace_sigma term (equivalent to old product)\n"
            " - denominator: Divide by trace_sigma term (equivalent to old normalized)"
        ),
    )
    parser.add_argument(
        "--soft_alignment_gamma",
        type=float,
        default=1.0,
        help="Temperature parameter (gamma) for soft alignment computation. Higher values make the distribution sharper.",
    )
    parser.add_argument(
        "--subspace_mode",
        type=str,
        default="test",
        choices=["train", "test"],
        help=(
            "Which method to use for building subspace.\n"
            " - train: Use training data with actual labels to compute class means\n"
            " - test: Use test data with text query-based image retrieval to compute class means"
        ),
    )
    
    # test dataset으로 subspace를 구성할 때 사용되는 파라미터
    parser.add_argument(
        "--subspace_top_r",
        type=int,
        default=30,
        help="Number of top-R images to retrieve per class when building subspace (R in the paper). Only used when subspace_mode='test'.",
    )
    parser.add_argument(
        "--subspace_similarity_threshold",
        type=float,
        default=0.20,
        help="Minimum average cosine similarity threshold for retrieved images. If any class fails to meet this threshold, debias score will not be used. Only used when subspace_mode='test'.",
    )
    parser.add_argument(
        "--subspace_require_min_similarity",
        action="store_true",
        default=True,
        help="If set, debias score will be disabled if any class fails to meet the similarity threshold. Only used when subspace_mode='test'.",
    )
    parser.add_argument(
        "--debias_apply_threshold",
        type=float,
        default=0.02,
        help=(
            "Threshold for deciding whether to apply debiasing score. "
            "If (input_query_similarity - avg_class_query_similarity) < threshold, "
            "debiasing score will be applied. Default: 0.0 (always apply)."
        ),
    )

    # -------------------------------------------------
    # Prompt / text query configuration
    # -------------------------------------------------
    parser.add_argument(
        "--prompt_csv",
        type=str,
        default="/home/jaeho/VL_Debiasing/data/prompt_templates.csv",
        help="Path to the predefined prompt template CSV.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help=(
            "If set (int), limit the number of generated query prompts used in evaluation. "
            "Useful for debugging smaller subsets."
        ),
    )

    # -------------------------------------------------
    # Eval / logging configuration
    # -------------------------------------------------
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jaeho/VL_Debiasing/RLCF_Debiasing/outputs",
        help="Where to save score matrices, logs, and metadata.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="debug_run",
        help="String tag for this run (for logging / reproducibility).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for any stochastic components.",
    )
    parser.add_argument(
        "--use_amp",
        default=False,
        action="store_true",
        help="If set, use torch.cuda.amp.autocast() + GradScaler for mixed precision.",
    )

    # -------------------------------------------------
    # CLIP backbones (policy vs reward)
    # -------------------------------------------------
    # Backward-compat: legacy single-switch (now superseded by the two below)
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-B/16",
        help="[Deprecated] Legacy single CLIP backbone. Prefer policy/reward specific flags below.",
    )
    # New: explicit policy/reward separation
    parser.add_argument(
        "--policy_clip_model_name",
        type=str,
        default="ViT-B/16",
        help="Policy (trainable TTA) CLIP backbone, e.g., 'ViT-B/16'.",
    )
    parser.add_argument(
        "--reward_clip_model_name",
        type=str,
        default="ViT-L/14",
        help="Reward (frozen judge) CLIP backbone, e.g., 'ViT-L/14'.",
    )

    # -------------------------------------------------
    # Wandb (optional)
    # -------------------------------------------------
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="If set, log metrics and artifacts to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="RLCF_Debiasing",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        choices=["online", "offline", "disabled"],
        help="W&B mode override (e.g., offline by default).",
    )

    # -------------------------------------------------
    # ImageNet & Flickr evaluation
    # -------------------------------------------------
    parser.add_argument(
        "--eval_imagenet_flickr",
        action="store_true",
        default=False,
        help="If set, evaluate on ImageNet and Flickr30k after debiasing experiment.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for ImageNet evaluation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers.",
    )

    # -------------------------------------------------
    # Misc
    # -------------------------------------------------
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device identifier, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )

    parser.add_argument("--cache_data_dir", type=str, default="./cache")

    args = parser.parse_args()

    # Small convenience: if policy/reward names are missing, fall back to legacy
    if not getattr(args, "policy_clip_model_name", None):
        args.policy_clip_model_name = args.clip_model_name
    if not getattr(args, "reward_clip_model_name", None):
        # If user didn't specify, default to a stronger judge
        args.reward_clip_model_name = "ViT-L/14"

    return args
