# rewards/debias_score.py

import torch
from typing import Optional, Dict, Any, Tuple, Union


def _get_subspace_modules(subspace_mode: str = "test"):
    """subspace_mode에 따라 적절한 모듈을 동적으로 로드"""
    if subspace_mode == "train":
        from .subspace_train import DebiasSubspace, build_subspace_from_dataset
    elif subspace_mode == "test":
        from .subspace_test import DebiasSubspace, build_subspace_from_dataset
    else:
        raise ValueError(f"Invalid subspace_mode: {subspace_mode}. Must be 'train' or 'test'.")
    return DebiasSubspace, build_subspace_from_dataset


class DebiasScore:
    """
    편향(인종, 성별 등) 관련 페널티/보상을 계산하는 모듈.
    
    Subspace 기반 점수 계산 방식:
    - score_base: mu_norm, soft_alignment_l2, soft_alignment_kl
    - score_trace: none, numerator, denominator
    - 총 9가지 조합 지원 (3 base × 3 trace)
    
    사용: total_reward = clip_reward - lambda_weight * debias_score
    """

    def __init__(
        self,
        device: torch.device,
        lambda_weight: float = 0.0,
        subspace: Optional[Any] = None,
        score_mode: Optional[str] = None,  # Deprecated
        score_base: str = "mu_norm",
        score_trace: str = "none",
        soft_alignment_gamma: float = 1.0,
        epsilon: float = 1e-8,
        dataset_name: Optional[str] = None,
        attribute_name: Optional[str] = None,
        subspace_mode: str = "test",
        subspace_split: Optional[str] = None,
        subspace_equal_split: bool = False,
        subspace_batch_size: int = 64,
        subspace_clip_model: Optional[torch.nn.Module] = None,
        subspace_image_preprocess: Optional[Any] = None,
        auto_build_subspace: bool = True,
        subspace_reference: Union[str, int] = "mean",
        subspace_n_components: Optional[int] = None,
        subspace_clip_model_name: Optional[str] = None,
        subspace_cache_data_dir: Optional[str] = None,
        subspace_top_r: int = 10,
        subspace_similarity_threshold: float = 0.3,
        subspace_require_min_similarity: bool = False,
        policy_clip_model: Optional[torch.nn.Module] = None,
        policy_tokenizer: Optional[Any] = None,
        apply_threshold: float = 0.0,
        **kwargs
    ):
        self.device = device
        self.lambda_weight = lambda_weight
        self.subspace = subspace
        self.epsilon = epsilon
        self.soft_alignment_gamma = soft_alignment_gamma
        self.meta: Dict[str, Any] = dict(kwargs)
        self.auto_build_subspace = auto_build_subspace
        self.subspace_mode = subspace_mode
        self.policy_clip_model = policy_clip_model
        self.policy_tokenizer = policy_tokenizer
        self.apply_threshold = apply_threshold
        
        DebiasSubspace, build_subspace_from_dataset = _get_subspace_modules(subspace_mode)
        self.DebiasSubspace = DebiasSubspace
        self.build_subspace_from_dataset = build_subspace_from_dataset
        
        score_base, score_trace = self._convert_score_mode(score_mode, score_base, score_trace)
        self._validate_score_config(score_base, score_trace)
        self.score_base = score_base
        self.score_trace = score_trace
        
        self._auto_subspace_cfg = {
            "dataset_name": dataset_name.lower() if dataset_name else None,
            "attribute_name": attribute_name.lower() if attribute_name else None,
            "split": subspace_split,
            "equal_split": subspace_equal_split,
            "batch_size": subspace_batch_size,
            "clip_model": subspace_clip_model,
            "image_preprocess": subspace_image_preprocess,
            "reference": subspace_reference,
            "n_components": subspace_n_components or 1,
            "clip_model_name": subspace_clip_model_name,
            "cache_data_dir": subspace_cache_data_dir,
            "top_r": subspace_top_r,
            "similarity_threshold": subspace_similarity_threshold,
            "require_min_similarity": subspace_require_min_similarity,
        }
    def _validate_score_config(self, score_base: str, score_trace: str) -> None:
        """score_base와 score_trace 유효성 검증"""
        valid_bases = {
            "mu_norm", 
            "soft_alignment_l2", 
            "soft_alignment_kl",
            "instance_popularity", "instance_popularity_centered"
        }
        valid_traces = {"none", "numerator", "denominator"}
        
        if score_base not in valid_bases:
            raise ValueError(f"Invalid score_base: {score_base}. Must be one of {valid_bases}")
        if score_trace not in valid_traces:
            raise ValueError(f"Invalid score_trace: {score_trace}. Must be one of {valid_traces}")

    def compute_score(
        self,
        text_index: Optional[torch.Tensor] = None,
        image_index: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Debiasing 보상/페널티 점수 계산
        
        Returns:
            debias_scores: [B] 점수 텐서 (편향이 클수록 더 큰 값)
        """
        self._ensure_subspace()

        if not self._is_subspace_available():
            batch_size = self._get_batch_size(text_index, image_index, image_features)
            return self.lambda_weight * torch.zeros(batch_size, device=self.device)
        
        if image_features is None:
            raise ValueError("image_features must be provided for subspace-based score computation.")
        
        image_features = image_features.to(self.device)
        batch_size = image_features.shape[0]
        
        if batch_size == 0:
            return torch.zeros(0, device=self.device)
        
        projected = self._compute_subspace_projection(image_features)
        projected_coords = self._compute_subspace_coordinates(image_features)
        mu_s, sigma_s = self._compute_mean_and_covariance(projected_coords)
        
        # [변경] base_score는 스칼라일 수도 있고, [B] 벡터일 수도 있음
        base_score = self._compute_base_score(mu_s, sigma_s, projected)
        
        # Trace 적용 (Vector인 경우에도 scalar sigma가 브로드캐스팅되어 곱해짐)
        final_score = self._apply_trace_sigma(base_score, sigma_s)
        
        # [변경] final_score가 이미 벡터([B])라면 ones를 곱하지 않음
        if final_score.ndim > 0 and final_score.shape[0] == batch_size:
            debias_scores = final_score
        else:
            debias_scores = final_score * torch.ones(batch_size, device=self.device)
            
        return self.lambda_weight * debias_scores

    def _compute_base_score(
        self,
        mu_s: torch.Tensor,
        sigma_s: torch.Tensor,
        projected_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """score_base에 따른 기본 점수 계산"""
        
        # 1. 기존 Logic: Global Mean Norm
        if self.score_base == "mu_norm":
            return torch.norm(mu_s, p=2)
            
        # 2. [신규 Logic]: Instance-wise Popularity Penalty
        elif self.score_base in {"instance_popularity", "instance_popularity_centered"}:
            if projected_embeddings is None:
                raise ValueError(f"projected_embeddings required for {self.score_base}")

            # (1) 각 샘플별 Soft Alignment 계산 -> [Batch, Class]
            prototypes = self._compute_class_prototypes()
            soft_assignments = self._compute_soft_alignment(projected_embeddings, prototypes)
            
            # (2) 배치 전체의 평균 클래스 분포 (인기도) -> [Class]
            #     alpha_bar = (1/K) * sum(alpha_i)
            popularity_dist = torch.mean(soft_assignments, dim=0) 
            
            # (3) 개별 패널티: 내적(alpha_i, alpha_bar)
            #     [Batch, Class] * [1, Class] -> [Batch, Class] -> sum -> [Batch]
            #     자신이 속한 클래스가 인기 많을수록 값이 커짐 (High Penalty)
            if self.score_base == "instance_popularity_centered":
                popularity_dist = popularity_dist - 1.0 / soft_assignments.shape[1]
            instance_scores = torch.sum(soft_assignments * popularity_dist.unsqueeze(0), dim=1)
            
            return instance_scores
        
        # 3. 기존 Logic: Distribution Distance
        elif self.score_base in {"soft_alignment_l2", "soft_alignment_kl"}:
            if projected_embeddings is None:
                raise ValueError(f"projected_embeddings required for {self.score_base}")
            
            batch_distribution = self._compute_soft_alignment_distribution(projected_embeddings)
            num_classes = batch_distribution.shape[0]
            target_distribution = torch.ones(num_classes, device=self.device) / num_classes
            
            if self.score_base == "soft_alignment_l2":
                return torch.norm(batch_distribution - target_distribution, p=2)
            else:  # soft_alignment_kl
                return self._compute_kl_divergence(batch_distribution, target_distribution)
        
        else:
            raise ValueError(f"Unknown score_base: {self.score_base}")
    def _convert_score_mode(
        self, 
        score_mode: Optional[str], 
        score_base: str, 
        score_trace: str
    ) -> Tuple[str, str]:
        """하위 호환성: score_mode를 score_base와 score_trace로 변환"""
        if score_mode is None:
            return score_base, score_trace
        
        conversion_map = {
            "l2_norm": ("mu_norm", "none"),
            "normalized": ("mu_norm", "denominator"),
            "product": ("mu_norm", "numerator"),
        }
        
        if score_mode not in conversion_map:
            raise ValueError(f"Invalid score_mode: {score_mode}")
        
        return conversion_map[score_mode]

    def update_metadata(self, **kwargs) -> None:
        """메타데이터 등록/업데이트"""
        self.meta.update(kwargs)


    def _is_subspace_available(self) -> bool:
        """Subspace 사용 가능 여부 확인"""
        return self.subspace is not None and self.subspace.subspace is not None

    def _get_batch_size(
        self,
        text_index: Optional[torch.Tensor],
        image_index: Optional[torch.Tensor],
        image_features: Optional[torch.Tensor],
    ) -> int:
        """배치 크기를 추론"""
        if image_features is not None:
            return image_features.shape[0]
        if image_index is not None:
            return len(image_index)
        if text_index is not None:
            return len(text_index)
        return 1

    def _compute_subspace_coordinates(self, embeddings: torch.Tensor) -> torch.Tensor:
        """임베딩을 subspace 좌표로 변환: U^T (z - μ_ref)"""
        self._check_subspace_initialized()
        subspace = self.subspace.subspace
        reference_vector = self.subspace.reference_vector
        centered = embeddings - reference_vector.unsqueeze(0)
        return centered @ subspace
    
    def _compute_subspace_projection(self, embeddings: torch.Tensor) -> torch.Tensor:
        """임베딩을 subspace에 투영: U U^T (z - μ_ref)"""
        self._check_subspace_initialized()
        subspace = self.subspace.subspace
        reference_vector = self.subspace.reference_vector
        embeddings = embeddings.to(self.device)
        centered = embeddings - reference_vector.unsqueeze(0)
        projected_coords = centered @ subspace
        return projected_coords @ subspace.T

    def _check_subspace_initialized(self) -> None:
        """Subspace 초기화 상태 확인"""
        if self.subspace is None or self.subspace.subspace is None:
            raise RuntimeError("Subspace is not initialized.")
        if self.subspace.reference_vector is None:
            raise RuntimeError("Reference vector is not initialized.")
    
    def _compute_mean_and_covariance(
        self,
        projected_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """투영된 좌표들의 평균과 공분산 계산"""
        batch_size = projected_coords.shape[0]
        mu_s = torch.mean(projected_coords, dim=0)
        
        if batch_size == 1:
            k = projected_coords.shape[1]
            sigma_s = torch.zeros(k, k, device=self.device)
        else:
            centered = projected_coords - mu_s.unsqueeze(0)
            sigma_s = (1.0 / (batch_size - 1)) * (centered.T @ centered)
        
        return mu_s, sigma_s
    

    def _compute_soft_alignment_distribution(
        self,
        projected_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Soft alignment를 통해 배치의 클래스 분포 계산"""
        prototypes = self._compute_class_prototypes()
        soft_assignments = self._compute_soft_alignment(projected_embeddings, prototypes)
        return self._compute_batch_class_distribution(soft_assignments)

    def _compute_class_prototypes(self) -> torch.Tensor:
        """클래스 프로토타입 계산: t_c = U U^T (μ_c - μ_ref)"""
        self._check_subspace_initialized()
        
        if self.subspace.class_means is None:
            raise RuntimeError("Class means are not initialized.")
        
        subspace = self.subspace.subspace
        class_means = self.subspace.class_means
        reference_vector = self.subspace.reference_vector
        
        centered = class_means - reference_vector.unsqueeze(0)
        projected_coords = centered @ subspace
        return projected_coords @ subspace.T
    
    def _compute_soft_alignment(
        self,
        projected_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Soft assignment 계산: α_c^(i) = exp(-γ ||s^(i) - t_c||_2^2) / sum"""
        distances = torch.cdist(projected_embeddings, prototypes, p=2) ** 2
        exp_distances = torch.exp(-self.soft_alignment_gamma * distances)
        return exp_distances / exp_distances.sum(dim=1, keepdim=True)
    
    def _compute_batch_class_distribution(
        self,
        soft_assignments: torch.Tensor,
    ) -> torch.Tensor:
        """배치 평균 클래스 분포 계산: ᾱ_c = (1/|B|) Σ_i α_c^(i)"""
        return torch.mean(soft_assignments, dim=0)

    def _compute_kl_divergence(
        self,
        batch_distribution: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence 계산: KL(ᾱ || π) = Σ_c ᾱ_c log(ᾱ_c / π_c)"""
        batch_dist = batch_distribution + self.epsilon
        target_dist = target_distribution + self.epsilon
        return torch.sum(batch_dist * torch.log(batch_dist / target_dist))
    
    def _apply_trace_sigma(
        self,
        base_score: torch.Tensor,
        sigma_s: torch.Tensor,
    ) -> torch.Tensor:
        """trace_sigma 처리 방법에 따라 점수 조정"""
        if self.score_trace == "none":
            return base_score
        
        trace_sigma = torch.trace(sigma_s)
        sigma_term = torch.sqrt(trace_sigma + self.epsilon) + self.epsilon
        
        if self.score_trace == "numerator":
            return base_score * sigma_term
        elif self.score_trace == "denominator":
            return base_score / sigma_term
        else:
            raise ValueError(f"Unknown score_trace: {self.score_trace}")

    def _ensure_subspace(self) -> None:
        """필요 시 자동으로 subspace 구성"""
        if self._is_subspace_available():
            return
        if not self.auto_build_subspace or self.lambda_weight <= 0.0:
            return

        cfg = self._auto_subspace_cfg
        dataset_name = cfg.get("dataset_name")
        attribute_name = cfg.get("attribute_name")
        clip_model = cfg.get("clip_model")
        image_preprocess = cfg.get("image_preprocess")

        if not dataset_name or not attribute_name:
            return
        if clip_model is None or image_preprocess is None:
            raise RuntimeError(
                "자동 subspace 생성을 위해 clip_model과 image_preprocess가 필요합니다."
            )

        split_to_use = cfg.get("split")
        if split_to_use is None:
            split_to_use = "train" if self.subspace_mode == "train" else None
        
        print(
            f"[DebiasScore] Building debias subspace ({dataset_name}/{attribute_name}) "
            f"using mode='{self.subspace_mode}', split='{split_to_use or 'auto'}'."
        )
        
        was_training = clip_model.training
        try:
            clip_model.eval()
            
            if self.subspace_mode == "train":
                subspace_result = self.build_subspace_from_dataset(
                    dataset_name=dataset_name,
                    attribute=attribute_name,
                    device=self.device,
                    clip_model=clip_model,
                    image_preprocess=image_preprocess,
                    split=split_to_use or "train",
                    reference=cfg.get("reference", "mean"),
                    n_components=cfg.get("n_components", 1),
                    equal_split=cfg.get("equal_split", False),
                    gallery_batch_size=cfg.get("batch_size", 64),
                    clip_model_name=cfg.get("clip_model_name"),
                    cache_data_dir=cfg.get("cache_data_dir"),
                )
                similarity_passed = True
            else:
                subspace_result, similarity_passed = self.build_subspace_from_dataset(
                    dataset_name=dataset_name,
                    attribute=attribute_name,
                    device=self.device,
                    clip_model=clip_model,
                    image_preprocess=image_preprocess,
                    split=split_to_use,
                    reference=cfg.get("reference", "mean"),
                    n_components=cfg.get("n_components", 1),
                    equal_split=cfg.get("equal_split", False),
                    gallery_batch_size=cfg.get("batch_size", 64),
                    clip_model_name=cfg.get("clip_model_name"),
                    cache_data_dir=cfg.get("cache_data_dir"),
                    top_r=cfg.get("top_r", 10),
                    similarity_threshold=cfg.get("similarity_threshold", 0.3),
                    require_min_similarity=cfg.get("require_min_similarity", False),
                )
            
            if not similarity_passed:
                print(f"[DebiasScore] Similarity check failed. Disabling debias score.")
                self.subspace = None
            else:
                self.subspace = subspace_result
        finally:
            if was_training:
                clip_model.train()

    def should_apply_debias(
        self,
        input_query_text_embedding: torch.Tensor,  # [D] 입력 쿼리 임베딩
        top_k_image_embeddings: torch.Tensor,      # [K, D] Top K 이미지 임베딩
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Debiasing score를 적용할지 여부를 판단합니다.
        
        판단 기준:
        - Top K 이미지 중 가장 유사도가 높은 이미지 선택
        - 입력 쿼리와 해당 이미지의 similarity 계산
        - 모든 attribute class 쿼리들과 해당 이미지의 similarity 평균 계산
        - (입력 쿼리 similarity - 평균 similarity) < 임계값이면 True 반환
        
        Args:
            input_query_text_embedding: [D] 입력 쿼리 텍스트 임베딩 (policy 공간)
            top_k_image_embeddings: [K, D] Top K 이미지 임베딩 (policy 공간)
            extra_info: 추가 정보
        
        Returns:
            bool: True이면 debiasing score를 적용, False이면 적용하지 않음
        """
        # 1. Policy 모델 및 attribute 정보 확인
        if self.policy_clip_model is None or self.policy_tokenizer is None:
            return True  # 기본값: 항상 적용
        
        if not self._is_subspace_available():
            return True  # Subspace가 없으면 기본값
        
        # 2. Attribute의 class 이름 가져오기
        from .subspace_test import _get_class_names_from_attribute, _generate_class_queries
        
        dataset_name = self._auto_subspace_cfg.get("dataset_name")
        attribute_name = self._auto_subspace_cfg.get("attribute_name")
        
        if not dataset_name or not attribute_name:
            return True  # 정보가 없으면 기본값
        
        class_names = _get_class_names_from_attribute(attribute_name, dataset_name)
        if len(class_names) == 0:
            return True
        
        # 3. Class 쿼리 생성 및 임베딩
        class_queries = _generate_class_queries(attribute_name, class_names)
        # "a photo of a [class]" 형태로 변환 (현재는 "{class} person" 형태)
        class_queries = [f"a photo of a {q}" for q in class_queries]
        
        with torch.no_grad():
            # dtype 불일치 문제 해결: _encode_text_fp32 함수 사용
            from .subspace_test import _encode_text_fp32
            class_query_tokens = self.policy_tokenizer(class_queries).to(self.device)
            class_query_embeddings = _encode_text_fp32(self.policy_clip_model, class_query_tokens)
        
        # 4. Top K 이미지 중 가장 유사도가 높은 이미지 찾기
        # input_query_text_embedding과 top_k_image_embeddings의 유사도 계산
        # input_query_text_embedding이 [D] 또는 [1, D] 형태일 수 있으므로 처리
        if input_query_text_embedding.dim() > 1:
            input_query_text_embedding = input_query_text_embedding.squeeze(0)  # [D]
        input_query_norm = input_query_text_embedding / input_query_text_embedding.norm(dim=-1, keepdim=True)  # [D]
        similarities = (top_k_image_embeddings @ input_query_norm.unsqueeze(-1)).squeeze(-1)  # [K]
        best_image_idx = torch.argmax(similarities)
        best_image_embedding = top_k_image_embeddings[best_image_idx]  # [D]
        best_image_embedding = best_image_embedding / best_image_embedding.norm(dim=-1, keepdim=True)  # [D]
        
        # 5. 입력 쿼리와 최고 이미지의 similarity
        input_similarity = (input_query_norm @ best_image_embedding).item()
        
        # 6. 모든 class 쿼리들과 최고 이미지의 similarity 평균
        class_similarities = (class_query_embeddings @ best_image_embedding.T).squeeze(-1)  # [num_classes]
        avg_class_similarity = class_similarities.mean().item()
        
        # 7. 판단: (입력 쿼리 similarity - 평균 similarity) < 임계값
        similarity_diff = input_similarity - avg_class_similarity
        should_apply = similarity_diff < self.apply_threshold
        
        # 8. 로깅 (쿼리마다 적용 여부 기록)
        if extra_info is None:
            extra_info = {}
        extra_info['debias_applied'] = should_apply
        extra_info['similarity_diff'] = similarity_diff
        extra_info['input_similarity'] = input_similarity
        extra_info['avg_class_similarity'] = avg_class_similarity
        
        return should_apply
