# rewards/subspace.py

import os
import torch
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any, List, Tuple
from types import SimpleNamespace

import clip

from ..face_dataset_loader import build_dataset, extract_image_embeddings
from ..datasets import IATDataset


class DebiasSubspace:
    """
    Debiasing 데이터셋(FairFace, UTKFace, FACET)의 training 데이터를 사용하여
    편향 관련 subspace를 정의하는 모듈.
    
    각 클래스 c에 대한 평균 임베딩 μ_c를 계산하고,
    기준 벡터 μ_ref와의 차이 v_c = μ_c - μ_ref를 구한 뒤,
    상위 k개 주성분을 추출하여 subspace를 구성합니다.
    
    사용 예시:
        subspace_comp = DebiasSubspace(
            dataset_name="fairface",
            attribute="gender",
            device=device,
            reference="mean",
            n_components=1
        )
        subspace = subspace_comp.compute_subspace(class_means)  # [C, D]
    """
    
    def __init__(
        self,
        dataset_name: str,
        attribute: str,
        device: torch.device,
        reference: Union[str, int] = "mean",
        n_components: int = 1,
    ):
        """
        Args:
            dataset_name: 데이터셋 이름 ("fairface", "utkface", "facet")
            attribute: 분석할 속성 ("gender", "race", "age", "skin_tone")
            device: 연산에 사용할 디바이스
            reference: 기준 벡터 선택
                - "mean": 모든 클래스 평균의 평균 (전체 평균)
                - int: 특정 클래스 인덱스 (예: 0, 1, ...)
            n_components: 추출할 주성분 개수 (k)
        """
        self.dataset_name = dataset_name.lower()
        self.attribute = attribute.lower()
        self.device = device
        self.reference = reference
        self.n_components = n_components
        
        # 유효한 데이터셋 및 속성 검증
        valid_datasets = {"fairface", "utkface", "utk", "facet"}
        valid_attributes = {"gender", "race", "age", "skin_tone", "joint"}
        
        if self.dataset_name not in valid_datasets:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Must be one of {valid_datasets}"
            )
        
        if self.attribute not in valid_attributes:
            raise ValueError(
                f"Unsupported attribute: {attribute}. "
                f"Must be one of {valid_attributes}"
            )
        
        # 계산 결과 저장
        self.class_means: Optional[torch.Tensor] = None  # [C, D]
        self.reference_vector: Optional[torch.Tensor] = None  # [D]
        self.difference_vectors: Optional[torch.Tensor] = None  # [C, D]
        self.subspace: Optional[torch.Tensor] = None  # [D, k]
    
    def compute_subspace(
        self,
        class_means: torch.Tensor,
        auto_n_components: bool = True,
    ) -> torch.Tensor:
        """
        클래스별 평균 임베딩을 사용하여 debiasing subspace를 계산합니다.
        
        Args:
            class_means: 클래스별 평균 임베딩 [C, D] (C: 클래스 개수, D: 임베딩 차원)
            auto_n_components: True이면 n_components를 클래스 개수 - 1로 자동 설정
        
        Returns:
            subspace: 주성분 벡터들 [D, k] (k: n_components)
                     각 열이 하나의 주성분 벡터
        """
        class_means = class_means.to(self.device)
        self.class_means = class_means  # [C, D]
        
        # 클래스 개수 확인 및 n_components 자동 설정
        num_classes = class_means.shape[0]
        if num_classes == 0:
            raise ValueError("class_means is empty. Cannot compute subspace.")
        if auto_n_components:
            # 주성분 개수를 클래스 개수 - 1로 설정
            self.n_components = max(1, num_classes - 1)  # 최소 1개
        
        # 2. 기준 벡터 선택
        reference_vector = self._select_reference(class_means)
        self.reference_vector = reference_vector  # [D]
        
        # 3. 차이 벡터 계산: v_c = μ_c - μ_ref
        difference_vectors = self._compute_difference_vectors(
            class_means, reference_vector
        )
        self.difference_vectors = difference_vectors  # [C, D]
        
        # 4. 주성분 추출
        subspace = self._extract_principal_components(difference_vectors)
        self.subspace = subspace  # [D, k]
        
        return subspace
    
    def _compute_class_means(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        각 클래스별 평균 임베딩을 계산합니다.
        
        Args:
            embeddings: [N, D]
            labels: [N]
        
        Returns:
            class_means: [C, D] (C: 클래스 개수)
        """
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        embedding_dim = embeddings.shape[1]
        
        class_means = torch.zeros(num_classes, embedding_dim, device=self.device)
        
        for i, label in enumerate(unique_labels):
            # 해당 클래스의 모든 임베딩 선택
            mask = (labels == label)
            class_embeddings = embeddings[mask]  # [N_c, D]
            
            # 평균 계산
            class_means[i] = torch.mean(class_embeddings, dim=0)
        
        return class_means
    
    def _select_reference(
        self,
        class_means: torch.Tensor,
    ) -> torch.Tensor:
        """
        기준 벡터를 선택합니다.
        
        Args:
            class_means: [C, D]
        
        Returns:
            reference_vector: [D]
        """
        if self.reference == "mean":
            # 모든 클래스 평균의 평균
            reference_vector = torch.mean(class_means, dim=0)
        elif isinstance(self.reference, int):
            # 특정 클래스의 평균
            if self.reference < 0 or self.reference >= class_means.shape[0]:
                raise ValueError(
                    f"Reference class index {self.reference} is out of range. "
                    f"Valid range: [0, {class_means.shape[0] - 1}]"
                )
            reference_vector = class_means[self.reference]
        else:
            raise ValueError(
                f"Invalid reference: {self.reference}. "
                f"Must be 'mean' or an integer class index."
            )
        
        return reference_vector
    
    def _compute_difference_vectors(
        self,
        class_means: torch.Tensor,
        reference_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        각 클래스 평균과 기준 벡터의 차이를 계산합니다.
        
        Args:
            class_means: [C, D]
            reference_vector: [D]
        
        Returns:
            difference_vectors: [C, D] (v_c = μ_c - μ_ref)
        """
        # 브로드캐스팅: [C, D] - [D] = [C, D]
        difference_vectors = class_means - reference_vector.unsqueeze(0)
        return difference_vectors
    
    def _extract_principal_components(
        self,
        difference_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        차이 벡터들로부터 SVD를 사용하여 주성분을 추출합니다.
        
        차이 벡터들을 행 벡터로 쌓은 행렬 U = [v_1, v_2, ..., v_C]^T [C, D]에 대해
        SVD를 수행하고 상위 k개 주성분을 반환합니다.
        
        Args:
            difference_vectors: [C, D]
        
        Returns:
            subspace: [D, k] (각 열이 하나의 주성분 벡터)
        """
        # U = [v_1^T, v_2^T, ..., v_C^T] = difference_vectors [C, D]
        U = difference_vectors  # [C, D]
        
        # SVD: U = U_svd @ S @ V^T
        # 여기서 V의 열들이 주성분 방향 (right singular vectors)
        # U는 [C, D]이므로, C < D인 경우를 고려해야 함
        
        # 행 중심화 (각 행의 평균을 빼서 중앙화)
        U_centered = U - torch.mean(U, dim=0, keepdim=True)
        
        # 공분산 행렬 계산: (1/(C-1)) * U_centered^T @ U_centered [D, D]
        # 또는 직접 SVD 수행: U_centered = U_svd @ S @ V^T
        # 더 효율적인 방법: U_centered를 직접 SVD
        
        # SVD 수행 (V는 [D, D], 상위 k개 열만 필요)
        _, _, Vt = torch.linalg.svd(U_centered, full_matrices=False)
        # Vt는 [min(C, D), D] 형태
        
        # Vt의 행들이 주성분 방향 (각 행이 하나의 주성분)
        # 상위 k개를 선택하여 전치: [k, D] -> [D, k]
        num_available = min(U_centered.shape[0], U_centered.shape[1])
        k = min(self.n_components, num_available)
        
        if k <= 0:
            raise ValueError(
                f"Cannot extract {self.n_components} components. "
                f"Available: {num_available}"
            )
        
        # Vt의 첫 k개 행을 선택하고 전치
        principal_components = Vt[:k, :].T  # [D, k]
        
        return principal_components
    
    def project_to_subspace(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        임베딩을 subspace에 투영합니다.
        
        Args:
            embeddings: [N, D] 또는 [D]
        
        Returns:
            projected: [N, k] 또는 [k] (subspace로 투영된 좌표)
        """
        if self.subspace is None:
            raise RuntimeError(
                "Subspace has not been computed yet. "
                "Call compute_subspace() first."
            )
        
        embeddings = embeddings.to(self.device)
        original_shape = embeddings.shape
        
        # [D] 또는 [N, D] -> [N, D]로 정규화
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # 투영: [N, D] @ [D, k] = [N, k]
        projected = embeddings @ self.subspace
        
        # 원래 shape 복원
        if len(original_shape) == 1:
            projected = projected.squeeze(0)
        
        return projected
    
    def remove_subspace_component(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        임베딩에서 subspace 성분을 제거합니다 (debiasing).
        
        Args:
            embeddings: [N, D] 또는 [D]
        
        Returns:
            debiased: [N, D] 또는 [D] (subspace 성분이 제거된 임베딩)
        """
        if self.subspace is None:
            raise RuntimeError(
                "Subspace has not been computed yet. "
                "Call compute_subspace() first."
            )
        
        embeddings = embeddings.to(self.device)
        original_shape = embeddings.shape
        
        # [D] 또는 [N, D] -> [N, D]로 정규화
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # subspace로 투영
        projected = embeddings @ self.subspace  # [N, k]
        
        # 투영된 성분을 원래 임베딩에서 제거
        # [N, k] @ [k, D] = [N, D]
        subspace_component = projected @ self.subspace.T  # [N, D]
        debiased = embeddings - subspace_component
        
        # 원래 shape 복원
        if len(original_shape) == 1:
            debiased = debiased.squeeze(0)
        
        return debiased
    
    def get_subspace_info(self) -> Dict[str, Any]:
        """
        계산된 subspace 정보를 반환합니다.
        
        Returns:
            info: subspace 메타데이터 딕셔너리
        """
        info = {
            "dataset_name": self.dataset_name,
            "attribute": self.attribute,
            "reference": self.reference,
            "n_components": self.n_components,
            "subspace_shape": None,
            "num_classes": None,
        }
        
        if self.subspace is not None:
            info["subspace_shape"] = list(self.subspace.shape)
        
        if self.class_means is not None:
            info["num_classes"] = self.class_means.shape[0]
        
        return info


def _labels_from_index_attrs(
    index_to_attr: List[Dict[str, Any]],
    attribute: str,
    device: torch.device,
) -> torch.Tensor:
    """
    index_to_attr 리스트에서 원하는 attribute 값을 라벨 ID 텐서로 변환한다.
    """
    values: List[str] = []
    for entry in index_to_attr:
        raw_val = entry.get(attribute)
        # None이면 'unknown'으로 치환하여 별도 클래스 취급
        values.append("unknown" if raw_val is None else str(raw_val))

    unique_vals: Dict[str, int] = {}
    labels: List[int] = []
    for val in values:
        if val not in unique_vals:
            unique_vals[val] = len(unique_vals)
        labels.append(unique_vals[val])

    return torch.tensor(labels, dtype=torch.long, device=device)


def _sanitize_model_name(model_name: str) -> str:
    """CLIP 모델 이름을 파일명으로 사용 가능한 형태로 변환"""
    return model_name.replace("/", "-")


def _get_eval_split_for_dataset(dataset_name: str) -> str:
    """
    각 데이터셋의 평가에 사용되는 split을 반환합니다.
    
    Args:
        dataset_name: 데이터셋 이름 ("fairface", "utkface", "facet")
    
    Returns:
        평가 split 이름 ("test" 또는 "val")
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "fairface":
        # FairFace는 mode='test'일 때 partition='val'을 사용
        return "test"  # build_dataset에서 mode='test'로 전달하면 내부적으로 'val' partition 사용
    elif dataset_name in ("utkface", "utk"):
        return "test"
    elif dataset_name == "facet":
        return "test"  # FACET은 항상 test split (frac=0.1 샘플링)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _get_class_names_from_attribute(attribute: str, dataset_name: str) -> List[str]:
    """
    attribute와 dataset_name에 따라 클래스 이름 목록을 반환합니다.
    
    Args:
        attribute: 속성 이름 ("gender", "race", "age", "skin_tone")
        dataset_name: 데이터셋 이름 ("fairface", "utkface", "facet")
    
    Returns:
        클래스 이름 리스트
    """
    attribute = attribute.lower()
    dataset_name = dataset_name.lower()
    
    if attribute == "gender":
        # GENDER_ENCODING의 키들
        return list(IATDataset.GENDER_ENCODING.keys())
    
    elif attribute == "race":
        # dataset에 따라 다른 encoding 사용
        if dataset_name in ("utkface", "utk"):
            class_names = list(IATDataset.RACE_ENCODING_UTK.keys())
            # UTKFace 데이터셋에서는 "Others" 레이블이 필터링되므로 쿼리에서도 제외
            if "Others" in class_names:
                class_names.remove("Others")
            return class_names
        else:
            return list(IATDataset.RACE_ENCODING.keys())
    
    elif attribute == "age":
        # AGE_ENCODING의 값(0,1,2)에 따라 매핑
        # 0 -> young, 1 -> middle-aged, 2 -> old
        age_mapping = {0: "young", 1: "middle-aged", 2: "old"}
        # ENCODING에서 값들을 추출하고 고유값만 가져옴
        unique_values = sorted(set(IATDataset.AGE_ENCODING.values()))
        return [age_mapping[val] for val in unique_values]
    
    elif attribute == "skin_tone":
        return list(IATDataset.SKIN_ENCODING.keys())
    
    else:
        raise ValueError(f"Unknown attribute: {attribute}")


def _generate_class_queries(attribute: str, class_names: List[str]) -> List[str]:
    """
    클래스 이름들로부터 텍스트 쿼리를 생성합니다.
    
    Args:
        attribute: 속성 이름 (참고용)
        class_names: 클래스 이름 리스트
    
    Returns:
        텍스트 쿼리 리스트 (형식: "a photo of a {class_name}")
    """
    queries = []
    for class_name in class_names:
        # 소문자로 변환
        class_name_lower = class_name.lower()
        query = f"{class_name_lower} person"
        queries.append(query)
    
    return queries


def _encode_text_fp32(clip_model: torch.nn.Module, text_tokens: torch.Tensor) -> torch.Tensor:
    """
    CLIP 모델의 텍스트 인코딩을 float32로 수행합니다.
    
    CLIP의 encode_text는 내부에서 .type(self.dtype)로 half 캐스팅을 강제하는데,
    여기서는 텍스트 경로를 직접 따라가며 FP32로 계산해 dtype mismatch를 방지합니다.
    text_tta_model.py의 encode_text 메서드와 동일한 로직입니다.
    
    Args:
        clip_model: CLIP 모델
        text_tokens: 텍스트 토큰 [B, L]
    
    Returns:
        text_embeddings: 텍스트 임베딩 [B, D] (정규화됨)
    """
    cm = clip_model  # 줄여쓰기

    # 1) 임베딩 + 위치임베딩 (FP32)
    x = cm.token_embedding(text_tokens).float()  # [B, L, D]
    if hasattr(cm, "positional_embedding"):
        x = x + cm.positional_embedding.float()

    # 2) 트랜스포머 (CLIP은 [L, B, D] 형식으로 받음)
    x = x.permute(1, 0, 2)  # [L, B, D]
    x = cm.transformer(x)  # [L, B, D]
    x = x.permute(1, 0, 2)  # [B, L, D]

    # 3) 최종 LayerNorm (FP32)
    x = cm.ln_final(x.float())  # [B, L, D]

    # 4) 엔코딩 선택(토큰 argmax 위치) + 투영 (FP32)
    #   OpenAI CLIP: x[batch_idx, text.argmax(dim=-1)] @ text_projection
    idx = text_tokens.argmax(dim=-1)
    x = x[torch.arange(x.shape[0]), idx] @ cm.text_projection.float()  # [B, D]

    # 5) 정규화
    x = F.normalize(x.float(), dim=-1)
    return x


def _search_images_by_query(
    text_embedding: torch.Tensor,
    image_embeddings: torch.Tensor,
    top_r: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    텍스트 임베딩과 이미지 임베딩 간 코사인 유사도를 계산하여 top-R 이미지를 검색합니다.
    
    Args:
        text_embedding: 텍스트 임베딩 [D]
        image_embeddings: 이미지 임베딩들 [N, D]
        top_r: 검색할 이미지 개수 R
    
    Returns:
        (selected_embeddings, selected_indices, mean_similarity)
        - selected_embeddings: 선택된 이미지 임베딩 [R, D]
        - selected_indices: 선택된 인덱스 [R]
        - mean_similarity: 선택된 이미지들의 평균 코사인 유사도
    """
    num_images = image_embeddings.shape[0]
    if num_images == 0:
        raise ValueError("image_embeddings is empty. Cannot search images.")
    
    if top_r <= 0:
        raise ValueError(f"top_r must be positive, got {top_r}")
    
    # 코사인 유사도 계산: [D] @ [N, D]^T = [N]
    # 이미 정규화된 임베딩이라고 가정
    similarities = F.cosine_similarity(
        text_embedding.unsqueeze(0),  # [1, D]
        image_embeddings,  # [N, D]
        dim=1
    )  # [N]
    
    # top-R 인덱스 선택 (이미지 개수보다 많을 수 없음)
    actual_top_r = min(top_r, num_images)
    top_values, top_indices = torch.topk(similarities, k=actual_top_r, dim=0)
    
    # 선택된 이미지 임베딩
    selected_embeddings = image_embeddings[top_indices]  # [R, D]
    
    # 평균 similarity 계산
    mean_similarity = top_values.mean().item()
    
    return selected_embeddings, top_indices, mean_similarity


def _get_cache_path(data_dir: str, dataset: str, split: str, clip_model_name: str) -> str:
    """캐시 파일 경로 생성"""
    sanitized_name = _sanitize_model_name(clip_model_name)
    cache_dir = os.path.join(data_dir, dataset, split)
    cache_path = os.path.join(cache_dir, f"{sanitized_name}.pt")
    return cache_path


def build_subspace_from_dataset(
    dataset_name: str,
    attribute: str,
    device: torch.device,
    clip_model: torch.nn.Module,
    image_preprocess,
    split: Optional[str] = None,
    reference: Union[str, int] = "mean",
    n_components: int = 1,
    equal_split: bool = False,
    gallery_batch_size: int = 64,
    clip_model_name: Optional[str] = None,
    cache_data_dir: Optional[str] = None,
    top_r: int = 10,
    similarity_threshold: float = 0.3,
    require_min_similarity: bool = False,
) -> Tuple[Optional[DebiasSubspace], bool]:
    """
    주어진 dataset/attribute 조합에 대해 텍스트 쿼리 기반 이미지 검색으로 subspace를 구성한다.
    
    Returns:
        (DebiasSubspace 인스턴스 또는 None, similarity_check_passed: bool)
        - similarity_check_passed: 모든 클래스가 similarity 임계값을 만족했는지 여부
    """
    dataset_name = dataset_name.lower()
    attribute = attribute.lower()
    
    # 평가 split 자동 선택
    if split is None:
        split = _get_eval_split_for_dataset(dataset_name)
    
    # 클래스 이름 추출
    class_names = _get_class_names_from_attribute(attribute, dataset_name)
    
    if len(class_names) == 0:
        raise ValueError(f"No classes found for attribute '{attribute}' in dataset '{dataset_name}'")
    
    # 텍스트 쿼리 생성
    text_queries = _generate_class_queries(attribute, class_names)
    
    print(f"[DebiasSubspace] Building subspace for {dataset_name}/{attribute} using split='{split}'")
    print(f"[DebiasSubspace] Classes: {class_names}")
    print(f"[DebiasSubspace] Text queries: {text_queries}")
    
    # 이미지 갤러리 로드 (캐시 우선 사용)
    print(f"[DebiasSubspace] Loading image gallery from {split} split...")
    
    # 캐시 파일 경로 확인
    image_embeddings = None
    if cache_data_dir is not None and clip_model_name is not None:
        cache_path = _get_cache_path(cache_data_dir, dataset_name, split, clip_model_name)
        if os.path.exists(cache_path):
            print(f"[DebiasSubspace] 캐시 파일에서 이미지 임베딩 로드: {cache_path}")
            try:
                cache_data = torch.load(cache_path, map_location=device)
                image_embeddings = cache_data.get("image_embeddings")
                
                # 캐시 검증
                if image_embeddings is None:
                    print(f"[DebiasSubspace] WARNING: 캐시 파일에 image_embeddings가 없습니다. 새로 생성합니다.")
                    image_embeddings = None
                elif cache_data.get("dataset", "").lower() != dataset_name:
                    print(f"[DebiasSubspace] WARNING: 캐시 파일의 dataset이 일치하지 않습니다. 새로 생성합니다.")
                    image_embeddings = None
                elif cache_data.get("split", "") != split:
                    print(f"[DebiasSubspace] WARNING: 캐시 파일의 split이 일치하지 않습니다. 새로 생성합니다.")
                    image_embeddings = None
                elif cache_data.get("clip_model_name", "") != clip_model_name:
                    print(f"[DebiasSubspace] WARNING: 캐시 파일의 clip_model_name이 일치하지 않습니다. 새로 생성합니다.")
                    image_embeddings = None
                else:
                    # 캐시에서 성공적으로 로드됨
                    image_embeddings = image_embeddings.to(device)
                    print(f"[DebiasSubspace] 캐시에서 임베딩 로드 완료: {image_embeddings.shape}")
            except Exception as e:
                print(f"[DebiasSubspace] WARNING: 캐시 파일 로드 실패 ({e}). 새로 생성합니다.")
                image_embeddings = None
    
    # 캐시가 없거나 로드 실패 시 새로 생성
    if image_embeddings is None:
        print(f"[DebiasSubspace] 이미지 임베딩을 새로 생성합니다...")
        subspace_args = SimpleNamespace(
            dataset=dataset_name,
            attribute=attribute,
            gallery_split=split,
            equal_split=equal_split,
            gallery_batch_size=gallery_batch_size,
        )
        
        dataset, _ = build_dataset(subspace_args, image_preprocess)
        image_embeddings, _, _ = extract_image_embeddings(
            args=subspace_args,
            clip_model=clip_model,
            device=device,
            dataset=dataset,
        )
    
    # 이미지 갤러리가 비어있는지 확인
    if image_embeddings.shape[0] == 0:
        raise ValueError(f"No images found in {split} split for dataset {dataset_name}")
    
    # 이미지 임베딩 정규화 및 디바이스로 이동
    # (캐시에서 로드한 경우 이미 정규화되어 있을 수 있지만, 안전을 위해 다시 정규화)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    image_embeddings = image_embeddings.to(device)  # 디바이스 일치를 위해 명시적으로 이동
    
    # 텍스트 쿼리 임베딩 생성 (FP32로 직접 인코딩하여 dtype mismatch 방지)
    text_tokens = clip.tokenize(text_queries).to(device)
    with torch.no_grad():
        text_embeddings = _encode_text_fp32(clip_model, text_tokens)  # [C, D] (이미 정규화됨)
    
    # 각 클래스별로 이미지 검색 및 평균 임베딩 계산
    class_means_list = []
    similarity_check_passed = True
    
    for c_idx, (class_name, text_emb) in enumerate(zip(class_names, text_embeddings)):
        # 이미지 검색
        selected_embeddings, selected_indices, mean_similarity = _search_images_by_query(
            text_embedding=text_emb,
            image_embeddings=image_embeddings,
            top_r=top_r,
        )
        
        print(f"[DebiasSubspace] Class '{class_name}': retrieved {len(selected_embeddings)} images, mean similarity={mean_similarity:.4f}")
        
        # Similarity 체크
        if require_min_similarity and mean_similarity < similarity_threshold:
            print(f"[DebiasSubspace] WARNING: Class '{class_name}' failed similarity check "
                  f"(mean={mean_similarity:.4f} < threshold={similarity_threshold})")
            similarity_check_passed = False
        
        # 클래스별 평균 임베딩 계산
        class_mean = torch.mean(selected_embeddings, dim=0)  # [D]
        class_means_list.append(class_mean)
    
    # Similarity 체크 실패 시 None 반환
    if require_min_similarity and not similarity_check_passed:
        print(f"[DebiasSubspace] Similarity check failed. Returning None.")
        return None, False
    
    # 클래스별 평균 임베딩이 비어있는지 확인
    if len(class_means_list) == 0:
        raise RuntimeError("No class means computed. This should not happen.")
    
    # 클래스별 평균 임베딩을 행렬로 변환
    class_means = torch.stack(class_means_list, dim=0)  # [C, D]
    
    # DebiasSubspace 생성 및 계산
    debias_subspace = DebiasSubspace(
        dataset_name=dataset_name,
        attribute=attribute,
        device=device,
        reference=reference,
        n_components=n_components,
    )
    debias_subspace.compute_subspace(
        class_means=class_means,
        auto_n_components=True,
    )
    
    return debias_subspace, True


