# rewards/subspace.py

import os
import torch
from typing import Optional, Union, Dict, Any, List
from types import SimpleNamespace

from ..face_dataset_loader import build_dataset, extract_image_embeddings


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
        subspace = subspace_comp.compute_subspace(embeddings, labels)
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
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        auto_n_components: bool = True,
    ) -> torch.Tensor:
        """
        주어진 임베딩과 라벨을 사용하여 debiasing subspace를 계산합니다.
        
        Args:
            embeddings: 임베딩 텐서 [N, D] (N: 샘플 수, D: 임베딩 차원)
            labels: 클래스 라벨 [N] (0 ~ C-1 범위의 정수)
            auto_n_components: True이면 n_components를 클래스 개수 - 1로 자동 설정
        
        Returns:
            subspace: 주성분 벡터들 [D, k] (k: n_components)
                     각 열이 하나의 주성분 벡터
        """
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        
        # 1. 클래스별 평균 임베딩 계산
        class_means = self._compute_class_means(embeddings, labels)
        self.class_means = class_means  # [C, D]
        
        # 클래스 개수 확인 및 n_components 자동 설정
        num_classes = class_means.shape[0]
        if auto_n_components:
            # 주성분 개수를 클래스 개수 - 1로 설정
            original_n_components = self.n_components
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
    split: str = "train",
    reference: Union[str, int] = "mean",
    n_components: int = 1,
    equal_split: bool = False,
    gallery_batch_size: int = 64,
    clip_model_name: Optional[str] = None,
    cache_data_dir: Optional[str] = None,
) -> DebiasSubspace:
    """
    주어진 dataset/attribute 조합에 대해 지정한 split (기본 train)으로 subspace를 구성한다.
    캐시된 이미지 임베딩이 있으면 사용하고, 없으면 계산한다.
    """
    dataset_name = dataset_name.lower()
    attribute = attribute.lower()

    # 캐시 파일 확인 및 로드
    image_embeddings = None
    index_to_attr = None
    
    if clip_model_name is not None and cache_data_dir is not None:
        cache_path = _get_cache_path(cache_data_dir, dataset_name, split, clip_model_name)
        
        if os.path.exists(cache_path):
            print(f"[DebiasScore] 캐시된 임베딩 로드: {cache_path}")
            try:
                cache_data = torch.load(cache_path, map_location=device)
                
                # 캐시 검증
                if cache_data.get("dataset", "").lower() != dataset_name:
                    raise RuntimeError(
                        f"캐시 파일의 dataset이 일치하지 않습니다. "
                        f"예상: {dataset_name}, 실제: {cache_data.get('dataset')}"
                    )
                if cache_data.get("split", "") != split:
                    raise RuntimeError(
                        f"캐시 파일의 split이 일치하지 않습니다. "
                        f"예상: {split}, 실제: {cache_data.get('split')}"
                    )
                if cache_data.get("clip_model_name", "") != clip_model_name:
                    raise RuntimeError(
                        f"캐시 파일의 clip_model_name이 일치하지 않습니다. "
                        f"예상: {clip_model_name}, 실제: {cache_data.get('clip_model_name')}"
                    )
                
                # 캐시에서 데이터 로드
                image_embeddings = cache_data["image_embeddings"].to(device)
                dataset_meta = cache_data.get("dataset_meta", {})
                index_to_attr = dataset_meta.get("index_to_attr", None)
                
                if index_to_attr is None:
                    print(f"[DebiasScore] 경고: 캐시 파일에 index_to_attr가 없습니다. 메타데이터를 다시 계산합니다.")
                    image_embeddings = None  # 메타데이터 재계산 필요
                else:
                    print(f"[DebiasScore] 캐시 로드 완료: {image_embeddings.shape[0]}개 임베딩")
                    
            except Exception as e:
                print(f"[DebiasScore] 캐시 로드 실패: {e}. 임베딩을 새로 계산합니다.")
                image_embeddings = None
                index_to_attr = None

    # 캐시가 없거나 로드 실패 시 계산
    if image_embeddings is None:
        print(f"[DebiasScore] 이미지 임베딩 계산 중...")
        # subspace 전용 args 네임스페이스 구성
        subspace_args = SimpleNamespace(
            dataset=dataset_name,
            attribute=attribute,
            gallery_split=split,
            equal_split=equal_split,
            gallery_batch_size=gallery_batch_size,
        )

        dataset, _ = build_dataset(subspace_args, image_preprocess)
        image_embeddings, _, index_to_attr = extract_image_embeddings(
            args=subspace_args,
            clip_model=clip_model,
            device=device,
            dataset=dataset,
        )

    labels = _labels_from_index_attrs(index_to_attr, attribute, device)

    debias_subspace = DebiasSubspace(
        dataset_name=dataset_name,
        attribute=attribute,
        device=device,
        reference=reference,
        n_components=n_components,
    )
    debias_subspace.compute_subspace(
        embeddings=image_embeddings,
        labels=labels,
        auto_n_components=True,
    )
    return debias_subspace


