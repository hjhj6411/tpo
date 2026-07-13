# fsiglip2 — zooclaw-FashionSigLIP2 corpus pipeline

`fsiglip/`의 extract → build → serve 3단계를 [srpone/zooclaw-fashionsiglip2](https://huggingface.co/srpone/zooclaw-fashionsiglip2)
(transformers, SigLIP2-base-patch16-**384** wise-ft 파인튠)용으로 미러링한 것.

**중요**: 임베딩 공간이 Marqo FashionSigLIP(v1)과 완전히 다르다.
`fashionsiglip_corpus`와 `fashionsiglip2_corpus`의 shard/index를 절대 섞지 말 것.
쿼리 인코딩과 corpus 인코딩은 반드시 같은 모델이어야 한다.

## v1과의 차이

| | fsiglip (v1) | fsiglip2 |
|---|---|---|
| 모델 | `hf-hub:Marqo/marqo-fashionSigLIP` (open_clip) | `srpone/zooclaw-fashionsiglip2` (transformers) |
| 입력 | 224px | 384px (배치 기본값 256→128로 하향) |
| 텍스트 인코딩 | open_clip tokenizer | `padding="max_length"` (SigLIP 표준 레시피) |
| corpus | `data/fashionsiglip_corpus` | `data/fashionsiglip2_corpus` |
| 서버 포트 | 1235 | **1236** (v1과 나란히 실행 가능) |

HTTP 라우트/요청/응답 계약은 v1과 동일 → 기존 collector들은 `--client-url`
포트만 1236으로 바꾸면 그대로 작동.

## 실행 순서

```bash
# 1) corpus 재인코딩 (clip_corpus tar 재사용, shard 단위 resumable)
#    GPU 1개:
python fsiglip2/extract_fsiglip2.py --gpu 0 --num-gpus 1
#    GPU 4개 병렬:
for G in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$G python fsiglip2/extract_fsiglip2.py --gpu $G --num-gpus 4 &
done; wait

# 2) FAISS 인덱스 빌드
python fsiglip2/build_faiss_fsiglip2.py

# 3) 서버 (v1의 1235와 나란히)
python fsiglip2/serve_fsiglip2_knn.py --port 1236 --gpu 0
```

## corpus 재빌드 전 저비용 검증 (권장)

`score-image-files`는 corpus와 무관(요청 시점 인코딩)하다. 인덱스가 없어도
채점만 검증하려면 — extract 1개 shard만 돌려 미니 인덱스를 만들거나, 기존
저장된 패치에 재채점 실험을 돌려 모델 품질을 먼저 비교:

```bash
python fsiglip/rescore_pattern_patches_experiment.py \
  --conditions scale --score-url http://127.0.0.1:1236/score-image-files
```
