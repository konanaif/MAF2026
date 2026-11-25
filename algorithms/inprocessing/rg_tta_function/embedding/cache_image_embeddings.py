# cache_image_embeddings.py
# 이미지 임베딩을 사전 계산하여 캐시하는 스크립트

import os
import argparse
import torch
import clip
from types import SimpleNamespace
from typing import Optional

from ..face_dataset_loader import load_dataset_and_embeddings


def get_cache_args():
    """캐싱 스크립트용 argument parser"""
    parser = argparse.ArgumentParser(
        description="Pre-compute and cache image embeddings for CLIP models"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fairface", "utkface", "facet"],
        help="Dataset name",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split to cache",
    )
    
    parser.add_argument(
        "--clip_model_name",
        type=str,
        required=True,
        help="CLIP model name (e.g., 'ViT-B/16', 'ViT-L/14')",
    )
    
    parser.add_argument(
        "--attribute",
        type=str,
        default="race",
        choices=["race", "gender", "age"],
        help="Attribute to use for dataset building (does not affect embeddings)",
    )
    
    parser.add_argument(
        "--gallery_batch_size",
        type=int,
        default=64,
        help="Batch size for image encoding",
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/jaeho/VL_Debiasing/RLCF_Debiasing/data",
        help="Base directory for saving cached embeddings",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu, etc.)",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files",
    )
    
    return parser.parse_args()


def sanitize_model_name(model_name: str) -> str:
    """CLIP 모델 이름을 파일명으로 사용 가능한 형태로 변환"""
    # "ViT-B/16" -> "ViT-B-16"
    return model_name.replace("/", "-")


def get_cache_path(data_dir: str, dataset: str, split: str, clip_model_name: str) -> str:
    """캐시 파일 경로 생성"""
    sanitized_name = sanitize_model_name(clip_model_name)
    cache_dir = os.path.join(data_dir, dataset, split)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{sanitized_name}.pt")
    return cache_path


def cache_image_embeddings(
    dataset: str,
    split: str,
    clip_model_name: str,
    attribute: str = "race",
    gallery_batch_size: int = 64,
    data_dir: str = "/home/jaeho/VL_Debiasing/RLCF_Debiasing/data",
    device: str = "cuda",
    overwrite: bool = False,
) -> str:
    """
    이미지 임베딩을 계산하고 캐시 파일로 저장
    
    Args:
        dataset: 데이터셋 이름
        split: 데이터셋 split (train/val/test)
        clip_model_name: CLIP 모델 이름
        attribute: 데이터셋 빌드에 사용할 attribute (임베딩에는 영향 없음)
        gallery_batch_size: 배치 크기
        data_dir: 캐시 저장 디렉토리
        device: 사용할 디바이스
        overwrite: 기존 파일 덮어쓰기 여부
    
    Returns:
        저장된 캐시 파일 경로
    """
    # 캐시 경로 확인
    cache_path = get_cache_path(data_dir, dataset, split, clip_model_name)
    
    if os.path.exists(cache_path) and not overwrite:
        print(f"[Cache] 이미 존재하는 캐시 파일을 건너뜁니다: {cache_path}")
        return cache_path
    
    print(f"[Cache] 이미지 임베딩 계산 시작: {dataset}/{split}/{clip_model_name}")
    
    # 디바이스 설정
    device_obj = torch.device(device)
    
    # CLIP 모델 로드
    print(f"[Cache] CLIP 모델 로드 중: {clip_model_name}")
    clip_model, image_preprocess = clip.load(clip_model_name, device=device_obj, download_root=os.environ.get("RG_CLIP_DOWNLOAD_ROOT"))
    clip_model.eval()
    clip_model = clip_model.to(device_obj)
    
    # args 객체 생성 (load_dataset_and_embeddings에 필요한 형태)
    args = SimpleNamespace(
        dataset=dataset,
        attribute=attribute,
        gallery_split=split,
        gallery_batch_size=gallery_batch_size,
        equal_split=False,
    )
    
    # 이미지 임베딩 계산
    print(f"[Cache] 이미지 임베딩 계산 중...")
    image_embeddings, dataset_meta = load_dataset_and_embeddings(
        args=args,
        clip_model=clip_model,
        device=device_obj,
        image_preprocess=image_preprocess,
    )
    
    # 캐시 데이터 구성
    cache_data = {
        "image_embeddings": image_embeddings,  # [N, D] tensor
        "dataset_meta": dataset_meta,  # dict with index_to_path, index_to_attr, etc.
        "clip_model_name": clip_model_name,
        "dataset": dataset,
        "split": split,
    }
    
    # 저장
    print(f"[Cache] 캐시 파일 저장 중: {cache_path}")
    torch.save(cache_data, cache_path)
    
    print(f"[Cache] 완료! 저장된 임베딩 수: {len(image_embeddings)}")
    print(f"[Cache] 임베딩 차원: {image_embeddings.shape[1]}")
    
    return cache_path


def main():
    """메인 함수"""
    args = get_cache_args()
    
    cache_path = cache_image_embeddings(
        dataset=args.dataset,
        split=args.split,
        clip_model_name=args.clip_model_name,
        attribute=args.attribute,
        gallery_batch_size=args.gallery_batch_size,
        data_dir=args.data_dir,
        device=args.device,
        overwrite=args.overwrite,
    )
    
    print(f"\n[Cache] 최종 저장 경로: {cache_path}")


if __name__ == "__main__":
    main()

