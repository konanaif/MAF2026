# run_tta_experiment.py

import os
import torch
import random
import numpy as np
import clip
from typing import Optional, cast, Tuple, Dict, Any, List

from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW

from .args_parser import get_args
from .face_dataset_loader import load_dataset_and_embeddings
from .prompt_loader import load_prompts, tokenize_prompts
from .models.text_tta_model import TextTTAModel
from .rewards.reward_model import RewardModel
from .rewards.debias_score import DebiasScore 
from .RL.tta_rl_loop import episodic_tta_loop
from .eval.metrics_logger import save_results
from .eval.measure_bias import eval_bias_metrics


# 상수 정의
DEFAULT_POLICY_CLIP_MODEL = "ViT-B/16"
DEFAULT_REWARD_CLIP_MODEL = "ViT-L/14"
DEFAULT_CLIPSCORE_WEIGHT = 2.5
DEFAULT_OPTIMIZER_BETAS = (0.9, 0.999)
DEFAULT_OPTIMIZER_WEIGHT_DECAY = 0.0
DEFAULT_BIAS_MEASUREMENT_TOPN = 1000
DEFAULT_CACHE_DATA_DIR = "/home/jaeho/VL_Debiasing/RLCF_Debiasing/data"


def set_seed(seed: int) -> None:
    """재현성을 위한 시드 설정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_model_name(model_name: str) -> str:
    """CLIP 모델 이름을 파일명으로 사용 가능한 형태로 변환"""
    # "ViT-B/16" -> "ViT-B-16"
    return model_name.replace("/", "-")


def get_cache_path(data_dir: str, dataset: str, split: str, clip_model_name: str) -> str:
    """캐시 파일 경로 생성"""
    sanitized_name = sanitize_model_name(clip_model_name)
    cache_dir = os.path.join(data_dir, dataset, split)
    cache_path = os.path.join(cache_dir, f"{sanitized_name}.pt")
    return cache_path


def load_clip_model(
    args: Any, 
    device: torch.device, 
    model_name: Optional[str] = None
) -> Tuple[torch.nn.Module, Any, Any]:
    """
    CLIP 모델 로드.
    
    Args:
        args: 설정 객체
        device: 사용할 디바이스
        model_name: 모델명 (None이면 args에서 가져옴)
    
    Returns:
        (clip_model, image_preprocess, text_tokenizer) 튜플
    """
    if model_name is None:
        model_name = getattr(args, "clip_model_name", DEFAULT_POLICY_CLIP_MODEL)
    model_name = cast(str, model_name)

    clip_model, image_preprocess = clip.load(model_name, device=device, download_root=os.environ.get("RG_CLIP_DOWNLOAD_ROOT"))
    text_tokenizer = clip.tokenize

    clip_model.eval()
    clip_model = clip_model.to(device)
    return clip_model, image_preprocess, text_tokenizer


def build_final_queries(args: Any, prompt_spec: Dict[str, Any]) -> List[str]:
    """
    concepts를 사용하여 평가용 프롬프트를 생성.
    
    Args:
        args: 설정 객체
        prompt_spec: 프롬프트 스펙 딕셔너리 (concepts 키 포함)
    
    Returns:
        생성된 프롬프트 리스트
    """
    concepts = prompt_spec["concepts"]

    # 공백 정리 + 빈 문자열 제거
    clean_concepts = []
    for c in concepts:
        c_clean = " ".join(c.split())
        if len(c_clean) > 0:
            clean_concepts.append(c_clean)

    # 프롬프트 템플릿 적용
    final_prompts = []
    for c in clean_concepts:
        final_prompts.append(f"This person is {c}.")
        final_prompts.append(f"A {c} person.")

    # 중복 제거 (순서 유지)
    seen = set()
    deduped = []
    for p in final_prompts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    # max_prompts 제한 적용
    if getattr(args, "max_prompts", None) is not None:
        deduped = deduped[:int(args.max_prompts)]

    return deduped


def create_labels_from_dataset(
    dataset_meta: Dict[str, Any], 
    attribute_name: str
) -> np.ndarray:
    """
    데이터셋 메타데이터로부터 라벨 벡터 생성.
    
    Args:
        dataset_meta: 데이터셋 메타데이터
        attribute_name: 속성 이름
    
    Returns:
        라벨 배열 (numpy int64)
    """
    index_to_attr = dataset_meta["index_to_attr"]
    attr_values = [sample[attribute_name] for sample in index_to_attr]

    unique_vals = []
    val_to_id = {}
    for v in attr_values:
        if v not in val_to_id:
            val_to_id[v] = len(unique_vals)
            unique_vals.append(v)
    
    labels_list = np.array([val_to_id[v] for v in attr_values], dtype=np.int64)
    return labels_list


def create_prompt_embeddings(
    prompts: List[str],
    clip_model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device
) -> torch.Tensor:
    """
    프롬프트 텍스트를 임베딩 벡터로 변환.
    
    Args:
        prompts: 프롬프트 텍스트 리스트
        clip_model: CLIP 모델
        tokenizer: 텍스트 토크나이저
        device: 사용할 디바이스
    
    Returns:
        정규화된 프롬프트 임베딩 텐서
    """
    with torch.no_grad():
        token_batch = tokenizer(prompts).to(device)
        text_feats = clip_model.encode_text(token_batch).float()
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats.detach()


def initialize_clip_models(
    args: Any, 
    device: torch.device
) -> Dict[str, Any]:
    """
    정책 및 리워드용 CLIP 모델 초기화.
    
    Args:
        args: 설정 객체
        device: 사용할 디바이스
    
    Returns:
        모델 및 전처리 함수를 포함한 딕셔너리
    """
    policy_name = getattr(args, "policy_clip_model_name", DEFAULT_POLICY_CLIP_MODEL)
    reward_name = getattr(args, "reward_clip_model_name", DEFAULT_REWARD_CLIP_MODEL)
    
    clip_model_policy, policy_preprocess, policy_tokenizer = load_clip_model(
        args, device, model_name=policy_name
    )
    
    clip_model_reward, reward_preprocess, reward_tokenizer = load_clip_model(
        args, device, model_name=reward_name
    )
    
    return {
        "policy": {
            "model": clip_model_policy,
            "preprocess": policy_preprocess,
            "tokenizer": policy_tokenizer,
            "name": policy_name,
        },
        "reward": {
            "model": clip_model_reward,
            "preprocess": reward_preprocess,
            "tokenizer": reward_tokenizer,
            "name": reward_name,
        }
    }


def load_image_embeddings(
    args: Any,
    clip_models: Dict[str, Any],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    정책 및 리워드 공간에서 이미지 임베딩 로드 (캐시 파일 우선 사용).
    
    Args:
        args: 설정 객체
        clip_models: CLIP 모델 딕셔너리
        device: 사용할 디바이스
    
    Returns:
        (policy_embeddings, reward_embeddings, dataset_meta) 튜플
    """
    # 캐시 파일 경로 설정
    data_dir = getattr(args, "cache_data_dir", DEFAULT_CACHE_DATA_DIR)
    dataset = args.dataset.lower()
    split = getattr(args, "gallery_split", "test")
    
    policy_model_name = clip_models["policy"]["name"]
    reward_model_name = clip_models["reward"]["name"]
    
    policy_cache_path = get_cache_path(data_dir, dataset, split, policy_model_name)
    reward_cache_path = get_cache_path(data_dir, dataset, split, reward_model_name)
    
    # 캐시 파일 존재 여부 확인
    missing_caches = []
    if not os.path.exists(policy_cache_path):
        missing_caches.append(("Policy", policy_model_name, policy_cache_path))
    if not os.path.exists(reward_cache_path):
        missing_caches.append(("Reward", reward_model_name, reward_cache_path))
    
    if missing_caches:
        error_msg = "캐시 파일을 찾을 수 없습니다.\n\n"
        for model_type, model_name, cache_path in missing_caches:
            error_msg += f"{model_type} 모델용 캐시: {cache_path}\n"
        error_msg += "\n다음 명령어로 캐시 파일을 생성하세요:\n"
        for model_type, model_name, cache_path in missing_caches:
            error_msg += f"python -m RLCF_Debiasing.cache_image_embeddings --dataset {dataset} --split {split} --clip_model_name {model_name}\n"
        raise RuntimeError(error_msg)
    
    # 캐시 파일 로드
    print(f"[Cache] Policy 모델용 캐시 로드: {policy_cache_path}")
    policy_cache = torch.load(policy_cache_path, map_location=device)
    
    print(f"[Cache] Reward 모델용 캐시 로드: {reward_cache_path}")
    reward_cache = torch.load(reward_cache_path, map_location=device)
    
    # 메타데이터 검증
    policy_meta = policy_cache.get("dataset_meta", {})
    reward_meta = reward_cache.get("dataset_meta", {})
    
    # Policy 캐시의 메타데이터를 사용 (dataset_meta는 동일해야 함)
    dataset_meta = policy_meta
    
    # 검증: dataset, split, clip_model_name 확인
    if policy_cache.get("dataset", "").lower() != dataset:
        raise RuntimeError(
            f"Policy 캐시 파일의 dataset이 일치하지 않습니다. "
            f"예상: {dataset}, 실제: {policy_cache.get('dataset')}"
        )
    if policy_cache.get("split", "") != split:
        raise RuntimeError(
            f"Policy 캐시 파일의 split이 일치하지 않습니다. "
            f"예상: {split}, 실제: {policy_cache.get('split')}"
        )
    if policy_cache.get("clip_model_name", "") != policy_model_name:
        raise RuntimeError(
            f"Policy 캐시 파일의 clip_model_name이 일치하지 않습니다. "
            f"예상: {policy_model_name}, 실제: {policy_cache.get('clip_model_name')}"
        )
    
    if reward_cache.get("dataset", "").lower() != dataset:
        raise RuntimeError(
            f"Reward 캐시 파일의 dataset이 일치하지 않습니다. "
            f"예상: {dataset}, 실제: {reward_cache.get('dataset')}"
        )
    if reward_cache.get("split", "") != split:
        raise RuntimeError(
            f"Reward 캐시 파일의 split이 일치하지 않습니다. "
            f"예상: {split}, 실제: {reward_cache.get('split')}"
        )
    if reward_cache.get("clip_model_name", "") != reward_model_name:
        raise RuntimeError(
            f"Reward 캐시 파일의 clip_model_name이 일치하지 않습니다. "
            f"예상: {reward_model_name}, 실제: {reward_cache.get('clip_model_name')}"
        )
    
    # 임베딩 추출 및 디바이스 이동
    image_embeddings_policy = policy_cache["image_embeddings"].to(device)
    image_embeddings_reward = reward_cache["image_embeddings"].to(device)
    
    print(f"[Cache] Policy 임베딩 로드 완료: {image_embeddings_policy.shape}")
    print(f"[Cache] Reward 임베딩 로드 완료: {image_embeddings_reward.shape}")
    
    return image_embeddings_policy, image_embeddings_reward, dataset_meta


def create_tta_model(
    clip_model: torch.nn.Module,
    image_embeddings: torch.Tensor,
    device: torch.device,
    momentum: float
) -> TextTTAModel:
    """
    TTA 모델 생성 및 초기화.
    
    Args:
        clip_model: 정책 CLIP 모델
        image_embeddings: 이미지 임베딩
        device: 사용할 디바이스
        momentum: 모멘텀 파라미터
    
    Returns:
        초기화된 TTA 모델
    """
    tta_model = TextTTAModel(
        clip_model=clip_model,
        device=device,
        momentum=momentum,
    ).to(device)
    tta_model.set_image_features(image_embeddings)
    return tta_model


def create_reward_model(
    args: Any,
    clip_model_reward: torch.nn.Module,
    reward_tokenizer: Any,
    image_embeddings_reward: torch.Tensor,
    device: torch.device,
    policy_clip_model: torch.nn.Module,
    policy_preprocess: Any,
    policy_tokenizer: Optional[Any] = None,
) -> RewardModel:
    """
    리워드 모델 생성 및 초기화.
    
    Args:
        args: 설정 객체
        clip_model_reward: 리워드 CLIP 모델
        reward_tokenizer: 리워드 토크나이저
        image_embeddings_reward: 리워드 공간 이미지 임베딩
        device: 사용할 디바이스
        policy_clip_model: subspace 구성을 위한 정책 CLIP 모델
        policy_preprocess: 정책 CLIP 전처리 함수
    
    Returns:
        초기화된 리워드 모델
    """
    reward_model = RewardModel(
        device=device,
        sample_k=args.sample_k,
        clipscore_weight=DEFAULT_CLIPSCORE_WEIGHT,
        process_batch=False,
        reward_mode=args.reward_mode,
        debias_lambda=args.debias_lambda,
    )
    
    # 이미지 임베딩 설정
    # 이미지 임베딩이 미리 계산된 경우(예: retrieval 실험)에는 캐시에 등록하고,
    # 그렇지 않은 경우(예: ImageNet TTA)에는 나중에 동적으로 설정/사용한다.
    if image_embeddings_reward is not None:
        if hasattr(reward_model, "set_image_features_with_dataloader"):
            reward_model.set_image_features_with_dataloader(image_embeddings_reward)
        elif hasattr(reward_model, "set_image_features"):
            reward_model.set_image_features(image_embeddings_reward)
        else:
            raise RuntimeError("RewardModel missing set_image_features* method.")
    
    # 리워드 CLIP 설정
    if hasattr(reward_model, "set_reward_clip"):
        reward_model.set_reward_clip(clip_model_reward, reward_tokenizer)
    
    # Debiaser 설정
    auto_build_subspace = (
        args.reward_mode == "clip_plus_debias" and args.debias_lambda > 0.0
    )
    policy_clip_model_name = getattr(args, "policy_clip_model_name", DEFAULT_POLICY_CLIP_MODEL)
    cache_data_dir = getattr(args, "cache_data_dir", DEFAULT_CACHE_DATA_DIR)
    
    # 하위 호환성: debias_score_mode가 있으면 사용, 없으면 새로운 파라미터 사용
    score_mode = getattr(args, "debias_score_mode", None)
    score_base = getattr(args, "debias_score_base", "mu_norm")
    score_trace = getattr(args, "debias_score_trace", "none")
    soft_alignment_gamma = getattr(args, "soft_alignment_gamma", 1.0)
    
    debiaser = DebiasScore(
        device=device,
        lambda_weight=args.debias_lambda,
        score_mode=score_mode,  # 하위 호환성을 위해 유지
        score_base=score_base,
        score_trace=score_trace,
        soft_alignment_gamma=soft_alignment_gamma,
        dataset_name=args.dataset if auto_build_subspace else None,
        attribute_name=args.attribute if auto_build_subspace else None,
        subspace_mode=getattr(args, "subspace_mode", "test"),  # "train" 또는 "test"
        subspace_split=None,  # None이면 각 데이터셋의 기본 split 사용 (mode에 따라 결정)
        subspace_equal_split=getattr(args, "equal_split", False),
        subspace_batch_size=getattr(args, "gallery_batch_size", 64),
        subspace_clip_model=policy_clip_model if auto_build_subspace else None,
        subspace_image_preprocess=policy_preprocess if auto_build_subspace else None,
        auto_build_subspace=auto_build_subspace,
        subspace_clip_model_name=policy_clip_model_name if auto_build_subspace else None,
        subspace_cache_data_dir=cache_data_dir if auto_build_subspace else None,
        subspace_top_r=getattr(args, "subspace_top_r", 10),
        subspace_similarity_threshold=getattr(args, "subspace_similarity_threshold", 0.3),
        subspace_require_min_similarity=getattr(args, "subspace_require_min_similarity", False),
        policy_clip_model=policy_clip_model if auto_build_subspace else None,
        policy_tokenizer=policy_tokenizer if auto_build_subspace else None,
        apply_threshold=getattr(args, "debias_apply_threshold", 0.0),
    )
    reward_model.set_debiaser(debiaser)
    
    return reward_model


def create_optimizer_and_scaler(
    model: TextTTAModel,
    learning_rate: float,
    use_amp: bool
) -> Tuple[AdamW, GradScaler]:
    """
    옵티마이저 및 그라디언트 스케일러 생성.
    
    Args:
        model: 최적화할 모델
        learning_rate: 학습률
        use_amp: AMP 사용 여부
    
    Returns:
        (optimizer, scaler) 튜플
    """
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=DEFAULT_OPTIMIZER_BETAS,
        weight_decay=DEFAULT_OPTIMIZER_WEIGHT_DECAY,
    )
    scaler = GradScaler('cuda', enabled=use_amp)
    return optimizer, scaler


def maybe_init_wandb(args: Any, run_meta: Dict[str, Any]) -> Optional[Any]:
    """
    Weights & Biases 초기화 (옵션).
    
    Args:
        args: 설정 객체
        run_meta: 실행 메타데이터
    
    Returns:
        wandb 객체 (사용하지 않으면 None)
    """
    if not getattr(args, "use_wandb", False):
        return None

    import wandb
    
    # wandb_mode 확인 (기본값: "offline")
    wandb_mode = getattr(args, "wandb_mode", "offline")
    
    # offline 또는 disabled 모드가 아닐 때만 login 시도
    if wandb_mode not in ("offline", "disabled"):
        wandb.login(timeout=60)
    
    policy_name = getattr(args, "policy_clip_model_name", DEFAULT_POLICY_CLIP_MODEL)
    reward_name = getattr(args, "reward_clip_model_name", DEFAULT_REWARD_CLIP_MODEL)
    
    wandb_settings = wandb.Settings(
        mode=wandb_mode,
        init_timeout=120,
    )
    
    # 모든 args를 wandb config에 기록
    # vars(args)를 사용하여 모든 파라미터를 자동으로 포함
    config_dict = vars(args).copy()
    
    # policy/reward 모델 이름은 명시적으로 설정 (이미 계산된 값 사용)
    config_dict["policy_clip_model_name"] = policy_name
    config_dict["reward_clip_model_name"] = reward_name
    
    # run_meta 추가
    config_dict.update(run_meta)
    
    wandb.init(
        project=getattr(args, "wandb_project", "RLCF_Debiasing_11.24"),
        name=args.run_name,
        config=config_dict,
        settings=wandb_settings,
    )
    return wandb


def save_experiment_results(
    args: Any,
    score_matrix: torch.Tensor,
    dataset_meta: Dict[str, Any],
    final_query_texts: List[str],
    run_meta: Dict[str, Any],
    policy_name: str,
    reward_name: str
) -> Dict[str, str]:
    """
    실험 결과 저장.
    
    Args:
        args: 설정 객체
        score_matrix: 점수 행렬
        dataset_meta: 데이터셋 메타데이터
        final_query_texts: 최종 쿼리 텍스트
        run_meta: 실행 메타데이터
        policy_name: 정책 모델명
        reward_name: 리워드 모델명
    
    Returns:
        저장된 파일 경로 딕셔너리
    """
    return save_results(
        output_dir=args.output_dir,
        run_name=args.run_name,
        score_matrix=score_matrix,
        meta={
            "dataset_meta": dataset_meta,
            "prompts": final_query_texts,
            "run_meta": run_meta,
        },
        args_dict={
            "dataset": args.dataset,
            "attribute": args.attribute,
            "tta_steps": args.tta_steps,
            "sample_k": args.sample_k,
            "lr": args.lr,
            "momentum": args.momentum,
            "reward_mode": args.reward_mode,
            "debias_lambda": args.debias_lambda,
            "seed": args.seed,
            "use_amp": args.use_amp,
            "prompt_csv": args.prompt_csv,
            "max_prompts": args.max_prompts,
            "run_name": args.run_name,
            "policy_clip_model_name": policy_name,
            "reward_clip_model_name": reward_name,
        },
    )


def finalize_wandb_run(
    wandb_run: Any,
    num_queries: int,
    num_gallery_images: int,
    output_dir: str,
    run_name: str,
    save_paths: Dict[str, str]
) -> None:
    """
    Weights & Biases 실행 마무리.
    
    Args:
        wandb_run: wandb 실행 객체
        num_queries: 쿼리 개수
        num_gallery_images: 갤러리 이미지 개수
        output_dir: 출력 디렉토리
        run_name: 실행 이름
        save_paths: 저장된 파일 경로 딕셔너리
    """
    if wandb_run is None:
        return
    
    wandb_run.log({
        "num_queries": num_queries,
        "num_gallery_images": num_gallery_images,
    })
    
    cfg_dump_path = os.path.join(output_dir, run_name, "config.json")
    wandb_run.save(cfg_dump_path)
    wandb_run.save(save_paths["summary_path"])
    wandb_run.finish()


def prepare_prompts_and_embeddings(
    args: Any,
    clip_models: Dict[str, Any],
    device: torch.device
) -> Tuple[List[str], List[Any], torch.Tensor]:
    """
    프롬프트 생성 및 임베딩 준비.
    
    Args:
        args: 설정 객체
        clip_models: CLIP 모델 딕셔너리
        device: 사용할 디바이스
    
    Returns:
        (final_query_texts, query_token_list, prompts_embeddings_policy) 튜플
    """
    # 프롬프트 로드 및 생성
    prompt_spec = load_prompts(args)
    final_query_texts = build_final_queries(args, prompt_spec)

    # RL 루프용 토큰 (정책 tokenizer 사용)
    query_token_list = tokenize_prompts(
        prompt_list=final_query_texts,
        text_tokenizer=clip_models["policy"]["tokenizer"],
        device=device,
    )

    # bias 측정용 프롬프트 임베딩 (정책 텍스트 인코더 기준)
    prompts_embeddings_policy = create_prompt_embeddings(
        prompts=final_query_texts,
        clip_model=clip_models["policy"]["model"],
        tokenizer=clip_models["policy"]["tokenizer"],
        device=device,
    )

    return final_query_texts, query_token_list, prompts_embeddings_policy


def run_tta_training(
    query_token_list: List[Any],
    tta_model: TextTTAModel,
    reward_model: RewardModel,
    optimizer: AdamW,
    scaler: GradScaler,
    args: Any,
    final_query_texts: List[str],
    labels_list: np.ndarray
) -> torch.Tensor:
    """
    TTA 학습 실행 및 결과 반환.
    
    Args:
        query_token_list: 쿼리 토큰 리스트
        tta_model: TTA 모델
        reward_model: 리워드 모델
        optimizer: 옵티마이저
        scaler: 그라디언트 스케일러
        args: 설정 객체
        final_query_texts: 최종 쿼리 텍스트
        labels_list: 라벨 리스트
    
    Returns:
        점수 행렬 [2, T]
    """
    result_dict = episodic_tta_loop(
        query_token_list=query_token_list,
        model=tta_model,
        reward_model=reward_model,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        query_text_list=final_query_texts,
        labels_list=labels_list,
    )

    avg_maxskew_per_epoch = torch.tensor(
        result_dict["avg_maxskew_per_epoch"], 
        dtype=torch.float32
    )
    avg_ndkl_per_epoch = torch.tensor(
        result_dict["avg_ndkl_per_epoch"], 
        dtype=torch.float32
    )
    score_matrix = torch.stack([avg_maxskew_per_epoch, avg_ndkl_per_epoch], dim=0)
    
    return score_matrix


def evaluate_bias(
    labels_list: np.ndarray,
    image_embeddings: torch.Tensor,
    prompts_embeddings: torch.Tensor,
    device: torch.device,
    topn: int = DEFAULT_BIAS_MEASUREMENT_TOPN
) -> Dict[str, Any]:
    """
    편향 평가 메트릭 계산.
    
    Args:
        labels_list: 라벨 리스트
        image_embeddings: 이미지 임베딩
        prompts_embeddings: 프롬프트 임베딩
        device: 사용할 디바이스
        topn: 상위 N개 고려
    
    Returns:
        편향 평가 결과 딕셔너리
    """
    image_embeddings = image_embeddings.to(device)
    return eval_bias_metrics(
        labels_list=labels_list,
        image_embeddings=image_embeddings,
        prompts_embeddings=prompts_embeddings,
        topn=topn,
    )


def main() -> None:
    """메인 실행 함수."""
    # 0) 초기 설정 및 재현성
    args = get_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) CLIP 모델 초기화 (정책 및 리워드)
    clip_models = initialize_clip_models(args, device)

    # 2) 이미지 임베딩 로드
    image_embeddings_policy, image_embeddings_reward, dataset_meta = load_image_embeddings(
        args, clip_models, device
    )

    # 3) 라벨 벡터 생성
    labels_list = create_labels_from_dataset(dataset_meta, args.attribute)

    # 4) 프롬프트 생성 및 임베딩
    final_query_texts, query_token_list, prompts_embeddings_policy = prepare_prompts_and_embeddings(
        args, clip_models, device
    )

    # 5) TTA 모델 생성
    tta_model = create_tta_model(
        clip_model=clip_models["policy"]["model"],
        image_embeddings=image_embeddings_policy,
        device=device,
        momentum=args.momentum,
    )

    # 6) 리워드 모델 생성
    reward_model = create_reward_model(
        args=args,
        clip_model_reward=clip_models["reward"]["model"],
        reward_tokenizer=clip_models["reward"]["tokenizer"],
        image_embeddings_reward=image_embeddings_reward,
        device=device,
        policy_clip_model=clip_models["policy"]["model"],
        policy_preprocess=clip_models["policy"]["preprocess"],
        policy_tokenizer=clip_models["policy"]["tokenizer"],
    )

    # 7) 옵티마이저 및 스케일러 생성
    optimizer, scaler = create_optimizer_and_scaler(
        model=tta_model,
        learning_rate=args.lr,
        use_amp=args.use_amp,
    )

    # 8) Weights & Biases 초기화 (옵션)
    run_meta = {
        "num_queries": len(query_token_list),
        "num_gallery_images": image_embeddings_policy.shape[0],
    }
    wandb_run = maybe_init_wandb(args, run_meta)

    # 9) RL episodic TTA 학습 실행
    score_matrix = run_tta_training(
        query_token_list=query_token_list,
        tta_model=tta_model,
        reward_model=reward_model,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        final_query_texts=final_query_texts,
        labels_list=labels_list,
    )

    # 10) 결과 저장
    save_paths = save_experiment_results(
        args=args,
        score_matrix=score_matrix,
        dataset_meta=dataset_meta,
        final_query_texts=final_query_texts,
        run_meta=run_meta,
        policy_name=clip_models["policy"]["name"],
        reward_name=clip_models["reward"]["name"],
    )

    # 11) 편향 평가 (정책 공간 기준)
    bias_result = evaluate_bias(
        labels_list=labels_list,
        image_embeddings=image_embeddings_policy,
        prompts_embeddings=prompts_embeddings_policy,
        device=device,
    )

    # 12) Weights & Biases 마무리
    finalize_wandb_run(
        wandb_run=wandb_run,
        num_queries=len(query_token_list),
        num_gallery_images=int(image_embeddings_policy.shape[0]),
        output_dir=args.output_dir,
        run_name=args.run_name,
        save_paths=save_paths,
    )

    # 13) ImageNet & Flickr 평가 (옵션)
    if getattr(args, "eval_imagenet_flickr", False):
        print("\n" + "="*50)
        print("Starting ImageNet & Flickr evaluation...")
        print("="*50 + "\n")
        
        try:
            from .eval.imagenet_tta_eval import run_imagenet_eval
            imagenet_results = run_imagenet_eval(args, clip_models)
            
            # Log to wandb if available
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "imagenet/top1": imagenet_results["top1"],
                    "imagenet/top5": imagenet_results["top5"],
                })
            
            print("\n[ImageNet] Evaluation completed successfully.")
        except Exception as e:
            print(f"\n[ImageNet] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            from .eval.flickr_tta_eval import run_flickr_eval
            flickr_results = run_flickr_eval(args, clip_models)
            
            # Log to wandb if available
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "flickr/img_r1": flickr_results["img_r1"],
                    "flickr/img_r5": flickr_results["img_r5"],
                    "flickr/img_r10": flickr_results["img_r10"],
                    "flickr/txt_r1": flickr_results["txt_r1"],
                    "flickr/txt_r5": flickr_results["txt_r5"],
                    "flickr/txt_r10": flickr_results["txt_r10"],
                    "flickr/r_mean": flickr_results["r_mean"],
                })
            
            print("\n[Flickr] Evaluation completed successfully.")
        except Exception as e:
            print(f"\n[Flickr] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)
        print("ImageNet & Flickr evaluation finished.")
        print("="*50 + "\n")



if __name__ == "__main__":
    main()
