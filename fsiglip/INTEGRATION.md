# 통합 가이드 — serve_fsiglip_knn.py에 multi-expert 검증 라우트 붙이기

연구 보고서의 핵심 권고("검색 백본은 FashionSigLIP 유지, 검증 단계를 분해")를
**기존 서버를 건드리지 않고 라우트만 추가**하는 방식으로 구현했습니다. 모델은
재로드하지 않고 이미 떠 있는 `_state`와 헬퍼 클로저를 재사용합니다.

## 1) 파일 배치
```
fsiglip/
  serve_fsiglip_knn.py      # 기존 (KNN + /patch-coverage + /patch-color-coverage)
  color_expert.py           # 신규 — CIELAB k-means + ΔE2000 (CPU, torch 불필요)
  segmentation_expert.py    # 신규 — SAM3/Grounded-SAM, fail-open heuristic
  serve_expert_routes.py    # 신규 — /seg-color-coverage /seg-pattern-coverage /verify-option
```

## 2) serve_fsiglip_knn.py 맨 아래 (app.run 직전)에 추가

기존 서버에 있는 헬퍼 이름에 맞춰 인자만 매핑하면 됩니다. 보통 다음 이름들이
이미 존재합니다(없으면 실제 함수명으로 교체):

- `_load_pil_cached(src)` : URL/경로 → PIL (base64 LRU 캐시) → `load_pil`
- `_encode_norm(texts)`   : 텍스트 리스트 → L2 정규화 (N,D) → `encode_norm`
- `_img_feats_np(tiles)`  : PIL 리스트 → L2 정규화 (P,D) → `img_feats`
- `_make_tiles(img, grid, drop_white=True)` : grid×grid 타일 (near-white drop) → `make_tiles`

```python
import os
from fsiglip.segmentation_expert import GarmentSegmenter
from fsiglip.serve_expert_routes import register_expert_routes

# 세그멘터 1회 생성 (기본 heuristic; 서버에 SAM3/Grounded-SAM 설치 시 env로 전환)
#   SEG_BACKEND=sam3         + SAM3_CHECKPOINT=/path/to/sam3.pt
#   SEG_BACKEND=grounded_sam + GDINO_CONFIG / GDINO_CKPT / SAM_CKPT
_state["segmenter"] = GarmentSegmenter(
    backend=os.environ.get("SEG_BACKEND", "heuristic"),
    device=os.environ.get("SEG_DEVICE", "cuda"),
    sam3_checkpoint=os.environ.get("SAM3_CHECKPOINT"),
    gdino_config=os.environ.get("GDINO_CONFIG"),
    gdino_checkpoint=os.environ.get("GDINO_CKPT"),
    sam_checkpoint=os.environ.get("SAM_CKPT"),
)

register_expert_routes(
    app, _state,
    load_pil=_load_pil_cached,     # ← 실제 서버 헬퍼명으로
    encode_norm=_encode_norm,      # ← 실제 서버 헬퍼명으로
    img_feats=_img_feats_np,       # ← 실제 서버 헬퍼명으로
    make_tiles=_make_tiles,        # ← 실제 서버 헬퍼명으로
)
```

세그멘터 백본 로드가 실패하면(미설치 등) 자동으로 `heuristic`으로 fail-open되며
`_state["segmenter"].load_error`에 사유가 기록됩니다. 즉 SAM3가 없어도 서버는
정상 기동하고, 라우트는 near-white/near-black 전경 휴리스틱으로 degrade됩니다.

## 3) 신규 라우트 요약

| 라우트 | 역할 | 기존 대체 대상 |
|---|---|---|
| `POST /seg-color-coverage` | 마스크 픽셀 → CIELAB k-means + ΔE2000 색명 argmax → coverage | `/patch-color-coverage` (Quick-Win #1) |
| `POST /seg-pattern-coverage` | 마스크 안에 중심이 든 타일만 SigLIP {pattern}vs plain argmax | `/patch-coverage` (Medium-Term #1) |
| `POST /verify-option` | 한 번에 conjunctive `score = pcov_t × ccov^β` (collector 재랭킹용) | 재랭킹 fusion |

요청/응답 예:
```bash
# 색상만
curl -s :1235/seg-color-coverage -H 'Content-Type: application/json' -d '{
  "images": ["https://.../a.jpg"], "color": "navy", "garment": "coat"}'
# -> [{"coverage":0.93,"primary":"navy","top_other":"black","mask_used":true,...}]

# 선지 1개 종합 검증 (collector가 후보별로 호출)
curl -s :1235/verify-option -H 'Content-Type: application/json' -d '{
  "image": "https://.../b.jpg",
  "color": "navy", "pattern": "striped", "garment": "shirt", "beta": 1.0}'
# -> [{"pcov":0.86,"ccov":0.91,"score":0.78,"primary_color":"navy",...}]
```

## 4) collector 통합 (collect_images_*.py)

후보 다운로드/디코드 후, 기존 `/patch-coverage`·`/patch-color-coverage` 호출을
`/verify-option` 한 번으로 대체하고 `score`로 재랭킹합니다(유사도는 타이브레이크):

```python
def rerank_options(client, base_url, candidates, color, pattern, garment, beta=1.0):
    # candidates: [{"url":..., "sim": float}, ...]  (FashionSigLIP KNN 결과)
    payload = {"images": [c["url"] for c in candidates],
               "color": color, "pattern": pattern, "garment": garment, "beta": beta}
    res = client.post(f"{base_url}/verify-option", json=payload).json()
    for c, r in zip(candidates, res):
        c["score"] = r.get("score", 0.0)        # conjunctive; pcov=0 or ccov=0 -> 0
        c["pcov"], c["ccov"] = r.get("pcov"), r.get("ccov")
    # 1순위 score, 동점 시 retrieval sim (보고서: 유사도=타이브레이크)
    candidates.sort(key=lambda c: (c["score"], c["sim"]), reverse=True)
    return candidates
```

conjunctive(비보상)이므로 "패턴은 맞지만 색이 틀린"(또는 그 반대) 후보는
score≈0으로 자동 탈락합니다 — 예: `black camouflage`는 camo 패턴 AND 검정색을
동시에 요구하며, 곱 점수라야 한쪽만 맞는 후보를 걸러냅니다.

## 5) 설계 철학 보존 체크리스트 (보고서 대비)
- [x] 검색 백본 미변경 (FashionSigLIP KNN 그대로, 검증만 분해)
- [x] conjunctive·non-compensatory 스코어 `pcov_t × ccov^β`
- [x] 유사도는 타이브레이크로만
- [x] 모든 게이트 fail-open (세그 실패 → 전경 휴리스틱 → whole-image)
- [x] frozen·HTTP 서빙, 모델 재로드 없음
- [x] mock-testable (torch/SAM/faiss 없이 16개 체크 통과)
- [x] 색상: CIELAB k-means + ΔE2000 (CPU, navy↔blue 결정론적 분리)
- [ ] (Future) 동점 한정 pairwise VLM 타이브레이크 — 미구현(보고서도 "좁게 검토")
- [ ] (Future) SCHP 부위 파싱 — region_pixels로 upper/lower 근사만 제공
