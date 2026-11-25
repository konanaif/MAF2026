# rewards/reward_model.py
import torch
import torch.nn.functional as F
from typing import List, Optional, Union


class RewardModel:
    """
    CLIP 기반 리워드 모델(심판).
    역할:
      - (1) (선택) 내부에 '리워드용 CLIP'을 보관해 텍스트/이미지를 직접 인코딩
      - (2) 텍스트 임베딩 캐시(self.text_features) 유지
      - (3) 이미지 임베딩 캐시(self.image_features) 유지
      - (4) 주어진 top-K 인덱스를 기준으로 CLIPScore (= w * max(cos, 0)) 계산
      - (5) baseline(평균) 제거하여 advantage 형태의 리워드 반환

    지원 모드:
      A) 외부 임베딩 모드: set_text_features(), set_image_features()로 임베딩을 직접 주입
      B) 내부 인코딩 모드: set_reward_clip()으로 심판 CLIP을 등록한 뒤
         set_text_by_strings()/set_text_by_tokens() 등으로 원문 텍스트/토큰을 넘기면
         내부에서 encode_text로 임베딩을 생성/정규화/저장.

    주의:
      - self.image_features는 '리워드 CLIP' 공간에서 인코딩된 갤러리 임베딩이어야 함.
      - cosine을 dot으로 쓰기 위해 텍스트/이미지 임베딩은 항상 L2 normalize 한다.
    """

    def __init__(
        self,
        device: torch.device,
        sample_k: int = 16,
        clipscore_weight: float = 2.5,
        process_batch: bool = False,
        reward_mode: str = "clip_only", # debiasing 실험 시에는 "clip_plus_debias"
        debias_lambda: float = 0.0,
    ):
        """
        Args:
            device: torch.device
            sample_k: K (top-K 샘플 개수). RL 샘플링과 동일하게 맞춰야 함
            clipscore_weight: w 계수 (CLIPScore 스케일링)
            process_batch:
                - True  -> (bs, K) 텐서를 그대로 받아 baseline 계산
                - False -> (bs*K,) 텐서를 (bs,K)로 간주하여 baseline 계산
        """
        self.device = device
        self.sample_k = sample_k
        self.clipscore_weight = clipscore_weight
        self.process_batch = process_batch

        # 캐시(항상 L2 정규화된 벡터를 담는다)
        self.text_features: Optional[torch.Tensor] = None   # [N_text or 1, D]
        self.image_features: Optional[torch.Tensor] = None  # [N_img, D]

        # 내부 리워드 CLIP (선택)
        self._reward_clip = None
        self._reward_tokenizer = None

        # Debias 관련
        self.reward_mode = reward_mode
        self.debias_lambda = debias_lambda
        self.debiaser = None  # DebiasScore 인스턴스
    # ------------------------------------------------------------------
    # 리워드 CLIP 등록 / 내부 인코딩 유틸
    # ------------------------------------------------------------------
    def set_reward_clip(self, clip_model, tokenizer=None):
        """
        리워드 CLIP(예: ViT-L/14)과 tokenizer를 주입.
        이후 set_text_by_strings/set_text_by_tokens에서 내부 인코딩 가능.
        """
        self._reward_clip = clip_model
        self._reward_tokenizer = tokenizer
        # 안전을 위해 eval 모드 + 디바이스 정렬
        if self._reward_clip is not None:
            self._reward_clip.eval()
            self._reward_clip = self._reward_clip.to(self.device)

    def _encode_text_with_reward_clip(
        self,
        texts: Optional[List[str]] = None,
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        리워드 CLIP으로 텍스트 임베딩을 생성하여 L2 normalize 후 반환.
        texts 또는 tokens 중 하나만 제공하면 된다.
        """
        assert self._reward_clip is not None, (
            "Reward CLIP is not set. Call set_reward_clip(clip_model, tokenizer) first."
        )

        with torch.no_grad():
            if tokens is None:
                assert texts is not None, "Either texts or tokens must be provided."
                assert self._reward_tokenizer is not None, (
                    "Tokenizer not set. Provide tokenizer in set_reward_clip or pass tokens directly."
                )
                tokens = self._reward_tokenizer(texts).to(self.device)
            else:
                tokens = tokens.to(self.device)

            feats = self._reward_clip.encode_text(tokens).float()
            feats = F.normalize(feats, dim=-1)
            return feats.detach()

    # ------------------------------------------------------------------
    # 세팅 함수들 (외부 임베딩 / 내부 인코딩 모두 지원)
    # ------------------------------------------------------------------
    def set_debiaser(self, debiaser):
        self.debiaser = debiaser

    def set_text_features(self, text_features: torch.Tensor):
        """
        외부에서 계산된 텍스트 임베딩을 직접 주입.
        shape: [D] or [1,D] or [N_text, D]
        """
        text_features = text_features.to(self.device)
        self.text_features = F.normalize(text_features, dim=-1).detach()

    def set_many_text_features(self, all_text_features: torch.Tensor):
        """
        전체 텍스트 갤러리 임베딩을 한 번에 주입.
        shape: [N_text, D]
        """
        all_text_features = all_text_features.to(self.device)
        self.text_features = F.normalize(all_text_features, dim=-1).detach()

    def set_text_by_strings(self, texts: Union[str, List[str]]):
        """
        문자열(하나 또는 리스트)을 받아 내부 리워드 CLIP으로 임베딩 생성.
        """
        if isinstance(texts, str):
            texts = [texts]
        feats = self._encode_text_with_reward_clip(texts=texts, tokens=None)
        self.text_features = feats  # 이미 정규화됨

    def set_text_by_tokens(self, tokens: torch.Tensor):
        """
        토큰 텐서를 받아 내부 리워드 CLIP으로 임베딩 생성.
        tokens: [N, L] 또는 [1, L]
        """
        feats = self._encode_text_with_reward_clip(texts=None, tokens=tokens)
        self.text_features = feats  # 이미 정규화됨

    def set_image_features(self, image_features: torch.Tensor):
        """
        리워드 CLIP 공간에서 인코딩된 '전체 이미지 갤러리 임베딩'을 저장.
        shape: [N_img, D]
        """
        image_features = image_features.to(self.device)
        self.image_features = F.normalize(image_features, dim=-1).detach()

    # 하위호환용 별칭
    def set_image_features_with_dataloader(self, all_image_features: torch.Tensor):
        self.set_image_features(all_image_features)

    # ------------------------------------------------------------------
    # CLIPScore 계산
    # ------------------------------------------------------------------
    def CLIPScore(self, text_index: Optional[torch.Tensor] = None,
                  images_index: Optional[torch.Tensor] = None,
                  pairwise: bool = False) -> torch.Tensor:
        """
        CLIPScore 정의:  w * max(cos, 0)
        cosine은 L2 normalize 되어 있으므로 dot = cosine.

        Args:
            text_index:  [bs*K] or [K] 또는 None
            images_index:[bs*K] or [K] 또는 None
            pairwise:
                True  -> 모든 조합 유사도 행렬 (사용 권장X)
                False -> index별 1:1 매칭 (일반적인 RLCF retrieval에서 사용)

        Returns:
            scores:
                pairwise=False -> shape [bs*K] (또는 [K])
                pairwise=True  -> shape [*,*]
        """
        assert self.image_features is not None, "image_features not set."
        assert self.text_features is not None, "text_features not set."

        # 텍스트 선택
        if text_index is not None:
            text_sel = self.text_features.index_select(0, text_index.to(self.device))
        else:
            # 현재 쿼리 1개를 sample_k 번 반복
            text_sel = torch.repeat_interleave(self.text_features, self.sample_k, dim=0)

        # 이미지 선택
        if images_index is not None:
            img_sel = self.image_features.index_select(0, images_index.to(self.device))
        else:
            img_sel = torch.repeat_interleave(self.image_features, self.sample_k, dim=0)

        if pairwise:
            sim = self.clipscore_weight * (text_sel @ img_sel.t())
        else:
            sim = self.clipscore_weight * torch.sum(text_sel * img_sel, dim=-1)

        scores = torch.maximum(sim, torch.zeros_like(sim)).squeeze()
        return scores

    # ------------------------------------------------------------------
    # baseline 제거 (advantage)
    # ------------------------------------------------------------------
    def rewards_post_process(self, clip_scores: torch.Tensor) -> torch.Tensor:
        """
        RLCF Eq.(5): R = CLIPScore - mean(CLIPScore over K)
        """
        if self.process_batch:
            # clip_scores: [bs, K]
            baseline = torch.mean(clip_scores, dim=1, keepdim=True)   # [bs,1]
            advantages = clip_scores - baseline                        # [bs,K]
            rewards = advantages.reshape(-1).detach()                  # [bs*K]
        else:
            # clip_scores: [bs*K] (보통 bs=1 → [K])
            baseline = torch.mean(clip_scores)
            rewards = (clip_scores - baseline).detach()
        return rewards
    
    # 총 리워드 계산: CLIPScore + DebiasScore
    def total_reward(
        self,
        images_index: torch.Tensor,          # [K]
        policy_text_feat: torch.Tensor,      # [D]
        policy_image_feats: torch.Tensor,    # [K, D]
    ) -> torch.Tensor:
        # 1) CLIPScore (리워드 CLIP 공간, baseline 제거)
        clip_scores = self.CLIPScore(
            text_index=None,
            images_index=images_index,
            pairwise=False,
        )  # [K]
        rewards = self.rewards_post_process(clip_scores)  # [K]

        # 2) (옵션) DebiasScore — extra_info는 전달하지 않음
        if (
            self.reward_mode == "clip_plus_debias"
            and self.debiaser is not None
            and self.debias_lambda > 0.0
        ):
            # Debiasing score 적용 여부 판단
            extra_info = {}
            should_apply = self.debiaser.should_apply_debias(
                input_query_text_embedding=policy_text_feat,  # [D]
                top_k_image_embeddings=policy_image_feats,   # [K, D]
                extra_info=extra_info,
            )
            
            if should_apply:
                debias_scores = self.debiaser.compute_score(
                    text_index=None,
                    image_index=images_index,
                    text_features=policy_text_feat,     # [D] or [1,D]
                    image_features=policy_image_feats,  # [K, D]
                )  # [K]를 기대 (내부에서 lambda_weight까지 곱해 리턴) # lambda가 일종의 scale 역할까지 할 것으로 예상.
                rewards = rewards - debias_scores  # [K] total_reward = clip_score - lambda * debiasing_score
                
                # 로깅: debiasing score가 적용되었음을 기록 및 출력
                similarity_diff = extra_info.get('similarity_diff', 0.0)
                input_sim = extra_info.get('input_similarity', 0.0)
                avg_class_sim = extra_info.get('avg_class_similarity', 0.0)
                # print(
                #     f"[DebiasScore] ✓ APPLIED | "
                #     f"diff={similarity_diff:.4f} (input={input_sim:.4f}, avg_class={avg_class_sim:.4f}) | "
                #     f"threshold={self.debiaser.apply_threshold:.4f}"
                # )
                self.debiaser.update_metadata(
                    last_debias_applied=True,
                    last_similarity_diff=similarity_diff,
                )
            else:
                # 로깅: debiasing score가 적용되지 않았음을 기록 및 출력
                similarity_diff = extra_info.get('similarity_diff', 0.0)
                input_sim = extra_info.get('input_similarity', 0.0)
                avg_class_sim = extra_info.get('avg_class_similarity', 0.0)
                # print(
                #     f"[DebiasScore] ✗ NOT APPLIED | "
                #     f"diff={similarity_diff:.4f} (input={input_sim:.4f}, avg_class={avg_class_sim:.4f}) | "
                #     f"threshold={self.debiaser.apply_threshold:.4f}"
                # )
                self.debiaser.update_metadata(
                    last_debias_applied=False,
                    last_similarity_diff=similarity_diff,
                )

        return rewards

