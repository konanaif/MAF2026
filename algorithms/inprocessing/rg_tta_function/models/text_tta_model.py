# models/text_tta_model.py

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextTTAModel(nn.Module):
    """
    TextTTAModel
    - Text encoder만 업데이트 가능한 형태로 감싼 CLIP-style 모델 래퍼.
    - Image encoder는 freeze된 상태에서 미리 구해둔 image_features만 사용.
    - forward()는 텍스트 vs 이미지 갤러리의 similarity(logits)를 계산해서
      retrieval에 쓰일 logits_per_text를 반환한다.

    또한,
    - momentum_update(): 에피소드가 끝났을 때 현재 가중치로 slow buffer를 업데이트
    - reset_initial(): 다음 에피소드 시작 전에 파라미터를 초기값(또는 momentum 반영된 버전)으로 되돌림
    """

    def __init__(
        self,
        clip_model,
        device,
        momentum=0.0,
    ):
        """
        Args:
            clip_model: 사전학습된 CLIP 모델 (text_encode / image_encode / logit_scale 등을 제공한다고 가정)
                        - text encoder: 학습 대상
                        - image encoder: 동결
            device: torch.device
            momentum: float in [0,1]. 0이면 momentum 없이 그냥 리셋만.
                      >0이면 episode 사이에서 EMA처럼 초기버퍼를 갱신.
        """
        super().__init__()
        self.device = device
        self.momentum = momentum

        # CLIP 전체를 들고 있지만, 학습은 text encoder 쪽만 허용해줄 거야.
        self.clip_model = clip_model

        # 캐시된 갤러리 이미지 임베딩 (정규화된 상태)
        self.register_buffer("image_features_cache", None, persistent=False)

        # 학습 가능한지 여부를 세팅
        self._freeze_image_encoder()
        self._enable_text_encoder()
        self._cast_text_encoder_to_float32() 
        # 모델 파라미터 스냅샷(초기 상태 / momentum 상태)
        #   initial_state_params: 에피소드 시작 시점으로 복원할 기준
        #   momentum_state_params: EMA 누적 상태
        self.initial_state_params = self._get_current_state()
        self.momentum_state_params = copy.deepcopy(self.initial_state_params)

    def _cast_text_encoder_to_float32(self):
        """
        텍스트 인코더 경로를 모두 FP32로 통일:
        - 모듈 파라미터들(transformer, token_embedding, ln_final)
        - text_projection (Parameter 재등록)
        - positional_embedding (Parameter 재등록)
        - self.clip_model.dtype 플래그
        """
        dev = self.device

        # 1) 모듈 단위 파라미터 FP32
        if hasattr(self.clip_model, "transformer"):
            self.clip_model.transformer.float()
        if hasattr(self.clip_model, "token_embedding"):
            self.clip_model.token_embedding.float()
        if hasattr(self.clip_model, "ln_final"):
            self.clip_model.ln_final.float()

        # 2) text_projection: Parameter 재등록(캐스팅)
        if hasattr(self.clip_model, "text_projection"):
            tp = self.clip_model.text_projection
            if isinstance(tp, torch.nn.Parameter):
                new_tp = torch.nn.Parameter(tp.detach().to(device=dev, dtype=torch.float32),
                                            requires_grad=tp.requires_grad)
                self.clip_model.register_parameter("text_projection", new_tp)
            else:
                new_tp = torch.nn.Parameter(torch.as_tensor(tp, device=dev, dtype=torch.float32),
                                            requires_grad=True)
                self.clip_model.register_parameter("text_projection", new_tp)

        # 3) positional_embedding: CLIP 텍스트 경로에 필수 버퍼(파라미터)
        if hasattr(self.clip_model, "positional_embedding"):
            pe = self.clip_model.positional_embedding
            if isinstance(pe, torch.nn.Parameter):
                new_pe = torch.nn.Parameter(pe.detach().to(device=dev, dtype=torch.float32),
                                            requires_grad=pe.requires_grad)
                self.clip_model.register_parameter("positional_embedding", new_pe)

        # 4) CLIP 내부에서 dtype property는 setter가 없음 → 대신 버퍼를 FP32로 바꿔줌
        for buf_name, buf in self.clip_model.named_buffers():
            if torch.is_floating_point(buf):
                buf.data = buf.data.to(dtype=torch.float32)


    # ------------------------------------------------------------------
    # Internal helpers for requires_grad control
    # ------------------------------------------------------------------
    def _freeze_image_encoder(self):
        """
        이미지 인코더 쪽은 gradient 안 흐르게 완전히 고정.
        """
        if hasattr(self.clip_model, "visual"):
            for p in self.clip_model.visual.parameters():
                p.requires_grad = False

    def _enable_text_encoder(self):
        """
        텍스트 인코더 쪽은 학습 가능하도록 열어줌.
        """
        if hasattr(self.clip_model, "transformer"):
            for p in self.clip_model.transformer.parameters():
                p.requires_grad = True
        if hasattr(self.clip_model, "token_embedding"):
            for p in self.clip_model.token_embedding.parameters():
                p.requires_grad = True
        if hasattr(self.clip_model, "ln_final"):
            for p in self.clip_model.ln_final.parameters():
                p.requires_grad = True
        if hasattr(self.clip_model, "text_projection"):
            # text_projection은 보통 nn.Parameter
            if isinstance(self.clip_model.text_projection, torch.nn.Parameter):
                self.clip_model.text_projection.requires_grad = True

        # logit_scale은 CLIP에서 학습 가능한 파라미터지만
        # test-time에는 보통 고정해도 되고(optional).
        if hasattr(self.clip_model, "logit_scale"):
            self.clip_model.logit_scale.requires_grad = False

    # ------------------------------------------------------------------
    # Feature cache setters
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_image_features(self, image_features: torch.Tensor):
        normed = F.normalize(image_features, dim=-1)

        self.image_features_cache = normed.to(self.device)


    # ------------------------------------------------------------------
    # Forward: text -> image retrieval 쿼리
    # ------------------------------------------------------------------
    def encode_text(self, text_tokens: torch.Tensor):
        """
        CLIP의 encode_text는 내부에서 .type(self.dtype)로 half 캐스팅을 강제한다.
        여기서는 텍스트 경로를 직접 따라가며 FP32로 계산해 dtype mismatch를 방지한다.
        """
        cm = self.clip_model  # 줄여쓰기

        # 1) 임베딩 + 위치임베딩 (FP32)
        x = cm.token_embedding(text_tokens).float()                          # [B, L, D]
        if hasattr(cm, "positional_embedding"):
            x = x + cm.positional_embedding.float()

        # 2) 트랜스포머 (CLIP은 [L, B, D] 형식으로 받음)
        x = x.permute(1, 0, 2)                                               # [L, B, D]
        x = cm.transformer(x)                                                # [L, B, D]
        x = x.permute(1, 0, 2)                                               # [B, L, D]

        # 3) 최종 LayerNorm (FP32)
        x = cm.ln_final(x.float())                                           # [B, L, D]

        # 4) 엔코딩 선택(토큰 argmax 위치) + 투영 (FP32)
        #   OpenAI CLIP: x[batch_idx, text.argmax(dim=-1)] @ text_projection
        idx = text_tokens.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), idx] @ cm.text_projection.float()    # [B, D]

        # 5) 정규화
        x = F.normalize(x.float(), dim=-1)
        return x


    def forward(self, text_tokens: torch.Tensor):
        assert self.image_features_cache is not None, \
            "image_features_cache is not set. Call set_image_features() first."

        text_feat = self.encode_text(text_tokens)          # [B, D]
        image_feat = self.image_features_cache             # [N_img, D]

        if hasattr(self.clip_model, "logit_scale"):
            logit_scale = self.clip_model.logit_scale.exp()
        else:
            logit_scale = torch.tensor(1.0, device=self.device)

        logits_per_text = logit_scale * text_feat @ image_feat.t()

        return logits_per_text


    # ------------------------------------------------------------------
    # Momentum / Reset logic for episodic TTA
    # ------------------------------------------------------------------
    def _get_current_state(self):
        """
        현재 text encoder 파라미터(+ 기타 trainable 파라미터들)를 딕셔너리 형태로 복사해서 반환.
        image encoder는 requires_grad=False라서 여기 안 들어와도 상관없음.
        """
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().clone()
        return state

    @torch.no_grad()
    def _load_state(self, state_dict):
        """
        내부 파라미터 중 requires_grad=True인 것들만 state_dict에서 덮어쓴다.
        """
        for name, param in self.named_parameters():
            if param.requires_grad and name in state_dict:
                param.copy_(state_dict[name])

    @torch.no_grad()
    def reset_initial(self):
        """
        한 에피소드(한 쿼리 텍스트에 대한 TTA) 이후,
        text encoder 파라미터를 initial_state_params로 되돌린다.
        즉, query-level episodic TTA 보장.
        """
        self._load_state(self.initial_state_params)

    @torch.no_grad()
    def momentum_update_model(self):
        """
        momentum이 0보다 크면,
        현재(튜닝된) 파라미터를 momentum_state_params에 EMA 방식으로 반영하고,
        그걸 initial_state_params로 갱신한다.

        아이디어:
        - 에피소드 끝날 때마다 조금씩 '좋아진' 방향을 누적시키고
        - 다음 쿼리의 초기 상태(initial_state_params)를 그 누적본으로 바꿔준다.

        momentum == 0 이면 아무것도 안 하고,
        그냥 현재 initial_state_params를 유지해도 된다.
        """
        if self.momentum <= 0.0:
            # 그냥 현재 initial_state_params 유지
            return

        new_initial = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                old_val = self.momentum_state_params[name]  # EMA buffer
                cur_val = param.detach()
                # 일반적인 EMA 공식: momentum이 클수록 이전 값 유지
                ema_val = self.momentum * old_val + (1.0 - self.momentum) * cur_val
                new_initial[name] = ema_val

        # momentum_state_params 업데이트
        self.momentum_state_params = copy.deepcopy(new_initial)
        # 다음 에피소드의 기준(initial_state_params)도 갱신
        self.initial_state_params = copy.deepcopy(new_initial)

    @torch.no_grad()
    def save_initial_state(self):
        """
        (옵션) 외부에서 명시적으로 호출해서
        '지금 상태를 새 에피소드 시작점으로 쓰겠다' 라고 선언할 수도 있게 제공.
        """
        self.initial_state_params = self._get_current_state()
        self.momentum_state_params = copy.deepcopy(self.initial_state_params)

    # ------------------------------------------------------------------
    # Utility flags (optional, for later hooks)
    # ------------------------------------------------------------------
    @property
    def only_text(self):
        """
        RLCF 원본 코드에 only_visual / not only_visual 플래그가 있던 것처럼,
        여기서는 텍스트 인코더만 업데이트한다는 걸 명시적으로 알려주는 헬퍼.
        추후 로깅/디버깅용.
        """
        return True
