# data/face_dataset_loader.py

import os
import torch
import torch.nn.functional as F

# 네가 준 datasets.py에 있는 클래스들을 import한다고 가정
# 실제 경로에 맞게 조정해줘야 해.
# 예: from .datasets import FairFace, UTKface, FACET
from .datasets import FairFace, UTKface, FACET


def build_dataset(args, image_preprocess):
    """
    args 정보와 주입된 image_preprocess(transform)를 이용해
    FairFace / UTKface / FACET 중 하나의 dataset 인스턴스를 만든다.

    반환:
        dataset: torch.utils.data.Dataset
            - __getitem__(i) 호출 시 dict-like 객체를 돌려주고,
              그 안에 변환된 이미지 텐서가 'img' (또는 sample.img)로 들어 있어야 한다.
        meta_attr_names: list[str]
            - 나중에 로그에 남길 속성 키들(라벨들).
              예: ["race", "gender", "age"] 등.
              지금 단계에서는 주로 기록용/디버깅용.
    """

    ds_name = args.dataset.lower()
    attr = args.attribute.lower()  # "race", "gender", "age", "skin_tone", "joint", ...

    # 어떤 split을 쓸지 (갤러리로 쓸 subset)
    # FairFace:  mode="test", "val", "train" 등이 가능
    # UTKface:   mode="test", "val", "train"
    # FACET:     FACET은 mode를 인자로 안 받으므로 무시 가능
    split_mode = getattr(args, "gallery_split", "test")

    # 갤러리 다운샘플링 등 옵션
    n_samples = None
    equal_split = getattr(args, "equal_split", False)

    if ds_name == "fairface":
        dataset = FairFace(
            iat_type=attr,
            lazy=True,
            mode=split_mode,
            _n_samples=n_samples,
            transforms=image_preprocess,
            equal_split=equal_split,
        )
        meta_attr_names = ["race", "gender", "age"]

    elif ds_name == "utkface" or ds_name == "utk":
        dataset = UTKface(
            iat_type=attr,
            lazy=True,
            mode=split_mode,
            _n_samples=n_samples,
            transforms=image_preprocess,
            equal_split=equal_split,
        )
        meta_attr_names = ["race", "gender", "age"]

    elif ds_name == "facet":
        dataset = FACET(
            iat_type=attr,
            lazy=True,
            _n_samples=n_samples,
            transforms=image_preprocess,
            equal_split=equal_split,
        )
        # FACET의 labels에는 skin_tone, age, gender 등이 파생돼 있음
        meta_attr_names = ["race", "gender", "age", "skin_tone"]

    else:
        raise ValueError(f"[build_dataset] Unsupported dataset: {ds_name}")

    return dataset, meta_attr_names


def _gather_sample_meta(dataset, row, idx):
    """
    dataset.labels.iloc[idx]에서 뽑은 row (pandas Series)를 기준으로
    이미지 경로, race/gender/age 등 메타데이터를 깔끔하게 dict로 정리한다.

    - FairFace:
        row["file"], row["race"], row["gender"], row["age"]
        이미지 경로는 dataset.DATA_PATH/imgs/train_val/<file>
    - UTKface:
        row["filename"], row["race"], row["gender"], row["age"]
        경로는 dataset.DATA_PATH/<filename>
    - FACET:
        row["filename"], row["gender"], row["age"], row["skin_tone"], (race는 없을 수도)
        경로는 FACET._search_dir(filename)
    """

    # 1) 경로 추출
    if "file" in row:
        # FairFace 스타일
        img_path = os.path.join(dataset.DATA_PATH, "imgs", "train_val", row["file"])
    elif "filename" in row:
        # UTKface or FACET 스타일
        if hasattr(dataset, "_search_dir"):
            # FACET: 이미지가 imgs_1 / imgs_2 / imgs_3 등 흩어져 있어서 _search_dir 필요
            img_path = dataset._search_dir(row["filename"])
        else:
            # UTKface: filename이 곧 파일명
            img_path = os.path.join(dataset.DATA_PATH, row["filename"])
    else:
        img_path = None  # 안전장치

    # 2) 속성(라벨) 추출
    attr_dict = {
        "race": row["race"] if "race" in row else None,
        "gender": row["gender"] if "gender" in row else None,
        "age": row["age"] if "age" in row else None,
        "skin_tone": row["skin_tone"] if "skin_tone" in row else None,
    }

    return img_path, attr_dict


def extract_image_embeddings(args, clip_model, device, dataset):
    """
    dataset 전체를 순회하면서:
      - dataset[i]에서 transform된 이미지 텐서 꺼내고
      - clip_model.encode_image()로 임베딩을 얻고 (정규화)
      - index별 이미지 경로 및 라벨 메타를 수집

    반환:
        image_embeddings: torch.Tensor [N_img, D]  (float32, L2-normalized)
        index_to_path: list[str] length N_img
        index_to_attr: list[dict] length N_img
    """

    batch_size = getattr(args, "gallery_batch_size", 64)

    all_feats = []
    index_to_path = []
    index_to_attr = []

    batch_imgs = []
    batch_paths = []
    batch_attrs = []

    def flush_batch():
        """지금 모은 batch_imgs를 한번에 CLIP에 태우고 결과를 쌓는다."""
        if len(batch_imgs) == 0:
            return

        imgs_tensor = torch.stack(batch_imgs, dim=0).to(device)  # [B, C, H, W]

        with torch.no_grad():
            feats = clip_model.encode_image(imgs_tensor)  # [B, D]
            feats = feats.float()
            feats = F.normalize(feats, dim=-1)            # cosine space에 맞게 정규화

        all_feats.append(feats.detach().cpu())

        # 메타 append
        index_to_path.extend(batch_paths)
        index_to_attr.extend(batch_attrs)

        batch_imgs.clear()
        batch_paths.clear()
        batch_attrs.clear()

    # dataset.labels는 pandas DataFrame이라 index로 메타 접근 가능
    # dataset[i]는 변환된 이미지 텐서를 포함한 샘플을 반환
    # (FairFace.__getitem__은 Dotdict / dict 형태로 img와 iat_label 등을 넣어줬어)
    for idx in range(len(dataset)):
        sample = dataset[idx]  # dict-like

        # 이미지 텐서 꺼내기
        if hasattr(sample, "img"):
            img_tensor = sample.img
        else:
            img_tensor = sample["img"]

        # 메타정보 추출
        row = dataset.labels.iloc[idx]
        img_path, attr_dict = _gather_sample_meta(dataset, row, idx)

        # 버퍼에 추가
        batch_imgs.append(img_tensor)
        batch_paths.append(img_path)
        batch_attrs.append(attr_dict)

        # 배치 사이즈만큼 쌓였으면 한 번에 인코딩
        if len(batch_imgs) >= batch_size:
            flush_batch()

    # 마지막 남은 배치 처리
    flush_batch()

    # [N_img, D]
    image_embeddings = torch.cat(all_feats, dim=0)

    return image_embeddings, index_to_path, index_to_attr


def load_dataset_and_embeddings(args, clip_model, device, image_preprocess):
    """
    이 함수는 run_tta_experiment.py에서 호출될 엔트리 포인트.

    1. 주어진 image_preprocess (CLIP에 맞는 transform)로 얼굴 데이터셋(FairFace/UTKface/FACET)을 구성
    2. 그 데이터셋 전체를 encode_image 해서 갤러리 임베딩을 만든다
    3. 각 인덱스에 어떤 이미지(경로)와 어떤 속성(race/gender/age/skin_tone)이 대응되는지 메타데이터를 만든다

    Args:
        args: 실험 세팅(argparse 결과)
        clip_model: 로드된 CLIP 모델 (encode_image 사용 가능해야 함)
        device: torch.device ('cuda', 'cuda:0', 'cpu', ...)
        image_preprocess: PIL -> CLIP tensor 변환하는 transform (모델 로더에서 가져온 것)

    Returns:
        image_embeddings: torch.Tensor [N_img, D]  # 전체 갤러리 임베딩
        dataset_meta: dict                         # 인덱스별 메타데이터 등
    """

    # 1) 데이터셋 생성
    dataset, meta_attr_names = build_dataset(
        args=args,
        image_preprocess=image_preprocess,
    )

    # 2) 전체 이미지 임베딩 계산 & 메타 수집
    (
        image_embeddings,
        index_to_path,
        index_to_attr,
    ) = extract_image_embeddings(
        args=args,
        clip_model=clip_model,
        device=device,
        dataset=dataset,
    )

    # 3) 메타 패키징
    dataset_meta = {
        "dataset_name": args.dataset,
        "attribute": args.attribute,
        "split": getattr(args, "gallery_split", "test"),
        "num_images": len(index_to_path),
        "meta_attr_names": meta_attr_names,  # 기록용
        "index_to_path": index_to_path,      # [N_img] 파일경로(또는 None)
        "index_to_attr": index_to_attr,      # [N_img] {race, gender, age, ...}
    }

    return image_embeddings, dataset_meta
