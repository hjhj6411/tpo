# 작업 지시: garment 어휘 통일 및 `long_coat` / `pea_coat` 재수집

Stage 4(이미지 수집) 작업이다. `wacv_scenario_v5` 생성(Stage 1–3)과는 별개이나,
`option_planner.py`의 검색어 매핑 수정은 **v5 생성 전에** 적용되어야 한다(§3-1).

---

## 1. 무엇이 잘못되었나

`long_coat` 셀의 검색어와 VLM 판정 라벨로 **`"wool coat"`** 이 쓰였다. 당시 판단은
"long coat보다 wool coat이 FAISS 검색 범주가 넓다"였고, 회수율만 놓고 보면 맞는
판단이었다. 그러나 **`wool coat`은 `long_coat`의 동의어가 아니라 상위 범주다.**
피코트도 울코트이고, 더플코트도 울코트다.

### 오염 경로

```
configs/config.py                long_coat            (정식 어휘)
  ↓ option_planner.py 287행이 치환
search_query                     "gray wool coat"
  ↓ collector 가 문자열에서 garment 추출
query_garment                    "wool coat"
  ↓ VLM 선택지로 그대로 제시
VLM 이 본 선택지                  "wool coat"          ("long coat" 은 본 적 없음)
  ↓ target 도 "wool coat" 이라 정확히 일치
판정                              PASS
```

**핵심:** VLM은 피코트 사진을 보고 "wool coat"이라 답할 수 있다. 피코트도 울코트이기
때문이다. 그리고 그 답은 `long_coat` 셀에서 PASS 처리된다. 라벨이 상위어인 한
모델이 아무리 정확해도 이 오분류를 막을 수 없다.

### 증거

**① VLM 프롬프트가 자기모순이다.** `VLM_GARMENT_PROMPT` 978–979행:

> "a pea coat **is a short double-breasted wool coat**; a wool coat reaches
> roughly the knee or below"

앞 절은 포함 관계, 뒤 절은 배타 관계를 말한다. 모델에게 "A는 B의 일종인데 A와 B를
구별하라"고 요구하는 셈이며, 구별 기준이 재질(둘 다 울)이라 성립할 수 없다.

**② 상위어와 하위어가 같은 선택지 목록에 형제로 들어 있다.**
`TEXT_QUERY_GARMENT_VOCAB`에 `"pea coat"`과 `"wool coat"`이 나란히 있다.

**③ 확보율이 극단적으로 비대칭이다.**

| | available | `no_such_garment` |
|---|---:|---:|
| `long_coat` | 64/75 (**85%**, 23종 중 12위) | 6 |
| `pea_coat`  | 22/64 (**34%**, 23종 중 22위) | **30** |

`pea_coat`의 제외 사유는 `no_such_garment`가 압도적이다. 상위어 검색이 두 범주를
모두 끌어와 `long_coat` 셀이 후보를 독식하고 `pea_coat` 셀에는 부적합한 것만 남는
패턴과 일치한다.

**④ 어휘 불일치 탐지 장치가 무력화되어 있었다.**
`assert_vocab_covers_targets()`(952행)는 target garment가 VLM 어휘 밖이면
`SystemExit`으로 중단시키는 안전장치다. 그런데 이 함수가
`garment_equivalence_set(g, allow_equivalence=True)`로 검사하기 때문에,
`GARMENT_EQUIV_GROUPS`가 `wool coat`을 `long coat`의 동의어로 인정해 통과시켰다.
**어휘 불일치를 잡아야 할 유일한 장치가 별칭 테이블 때문에 침묵했다.**

### 사람 검수로 걸러졌을 가능성

`long_coat` 셀은 사람이 전수 검수했으나, 검수자에게 제시된 라벨이 "wool coat"이었다면
무릎 길이 경계 판단이 흔들렸을 수 있고, 애초에 **후보 풀 자체가 오염된 상태**였다.
검수는 주어진 후보 중 고르는 작업이지 없는 후보를 만들어내지 못한다.

---

## 2. 영향 범위

| | 값 |
|---|---:|
| 재수집 대상 셀 | **139** (`long_coat` 75 + `pea_coat` 64) |
| 전체 셀 대비 | 8.4% (139 / 1,661) |
| v5 출제가능 문항 중 두 옷을 쓰는 문항 | **381 / 2,571 (14.8%)** |
| — `long_coat` 포함 | 283 |
| — `pea_coat` 포함 | 101 |

나머지 21종은 검색어가 전부 정식 명칭의 항등 매핑이므로 **영향이 없다.**

---

## 3. 코드 수정

핵심 원칙은 하나다:

> **garment 어휘는 `configs/config.py`의 정식 23종 하나뿐이다.
> 별칭도, 상위어도, 동의어 계층도 두지 않는다.**

### 3-1. `construction/option_planner.py` — 검색어 별칭 제거 ★v5 생성 전 필수

`GARMENT_SEARCH_PHRASE` 287행. 이 테이블은 23개 중 22개가 언더스코어→공백 항등
매핑이고, 이 항목만 다른 단어로 치환한다.

```diff
-    "long_coat": "wool coat",
+    # `wool coat` is a HYPERNYM — a pea coat is also a wool coat. Using it as
+    # long_coat's search phrase pulled pea coats into long_coat cells and
+    # starved the pea_coat cells (34% availability, lowest of all 23).
+    # Every other entry is the canonical name; this one must be too.
+    "long_coat": "long coat",
```

`long coat`을 택한 근거는 `configs/scenarios.py` 100행의 개정 이력이다:
*"Replace trench_coat with **length-specific** pea_coat / long_coat labels."*
두 코트는 처음부터 길이로 구별하도록 설계된 쌍이며, 시나리오 제약도 그 물리적
속성에 의존한다. `overcoat`은 그 자체가 또 다른 상위어이므로 대안이 되지 못한다.

**이 수정은 `search_query` 필드에 반영되므로 v5 옵션 플랜 생성 전에 적용해야 한다.**
적용 후 v5를 재생성하면 642개 옵션의 `search_query`가 바뀌어
`option_plans.jsonl` 해시가 달라진다. 이는 정상이다.

### 3-2. 수집기 — garment 어휘를 `configs.config`에서 import

현재 `collector_sam3.py`에 어휘가 두 벌 하드코딩되어 있다:

- `TEXT_QUERY_GARMENT_VOCAB` (23개) — **VLM에게 제시되는 닫힌 선택지**이자 검색어
  문자열에서 garment를 추출하는 사전
- `DEFAULT_GARMENTS` (38개) — 별칭·광의어가 섞인 파싱 사전.
  `GARMENT_QUERY_TERMS = TEXT_QUERY_GARMENT_VOCAB + DEFAULT_GARMENTS`로만 쓰인다

**실측 근거:** 실제 옵션 12,108개의 `search_query`를 정식 23종만으로 파싱한 결과
**target과 불일치한 사례가 0건**이다. `search_query`는 항상
`"{pattern} {color} {garment}"` 형태로 자동 생성되므로 별칭이 등장할 여지가 없고,
등장하더라도 `extract_query_garment`가 fallback(=실제 target)을 반환한다.
즉 `DEFAULT_GARMENTS`의 별칭 16개는 **한 번도 사용된 적이 없다.**

두 목록을 하나로 합치고, 하드코딩 대신 import 한다:

```python
# Canonical garment vocabulary — single source of truth.
# The pattern axis already imports from configs.config (see PATTERN_VOCAB
# below); garment stayed hardcoded and drifted, which is how `wool coat`
# entered as a stand-in for `long_coat`.
_GARMENT_AXIS_FALLBACK = [
    "t shirt", "tank top", "formal shirt", "polo shirt",
    "sweatshirt", "sweater", "hoodie", "cardigan",
    "blazer", "suit vest", "windbreaker", "leather jacket", "puffer jacket",
    "fleece jacket", "pea coat", "long coat",
    "jeans", "slacks", "shorts", "leggings",
    "dress", "mini skirt", "long skirt",
]
try:
    from configs.config import FASHION_ATTRIBUTE_AXES as _AXES
    GARMENT_VOCAB = [g.replace("_", " ") for g in _AXES["garment_category"]]
except Exception as _e:   # a SILENT fallback would re-hide a stale vocabulary
    print(f"  [warn] configs.config unavailable ({type(_e).__name__}: {_e}); "
          f"GARMENT_VOCAB falls back to the frozen list, which may be stale.")
    GARMENT_VOCAB = list(_GARMENT_AXIS_FALLBACK)

# No aliases, no hypernyms, no umbrella terms. Verified against all 12,108
# real search queries: these 23 terms parse every one of them correctly.
TEXT_QUERY_GARMENT_VOCAB = GARMENT_VOCAB
DEFAULT_GARMENTS = GARMENT_VOCAB
GARMENT_QUERY_TERMS = GARMENT_VOCAB
```

패턴 축이 이미 이 형태다(`collector_sam3.py` 773–781행). 주석에
*"a SILENT fallback would re-hide a stale vocabulary"* 라고 적혀 있는데,
garment 축에서 정확히 그 일이 벌어졌다.

### 3-3. 별칭 테이블 전체 제거

`GARMENT_EQUIV_GROUPS`(925행), `garment_equivalence_set()`(941행),
`--allow-garment-equivalence` 인자(1554행)를 **모두 삭제한다.**

**이유 — 이 테이블은 순기능이 없다.**

호출 지점이 두 곳뿐이고, 둘 다 문제다:

| 호출 지점 | 실제 효과 |
|---|---|
| `score_garment_vlm` 1082–1084행 | `allow_equivalence` 기본값이 **False**라 `{자기 자신}`만 반환. **판정에 아예 관여하지 않는다** |
| `assert_vocab_covers_targets` 960행 | `allow_equivalence=True`로 호출. `wool coat`을 `long coat`의 동의어로 인정해 **어휘 불일치 탐지를 무력화했다** |

즉 VLM 판정에는 기여하지 않으면서, 유일하게 작동하는 곳에서는 **오류를 숨기는
역할만** 했다. VLM이 정식 23종 중 하나를 고르고 그것이 target과 정확히 일치할 때만
PASS하는 것이 원래 설계이며, 별칭 계층은 그 위에 얹힌 불필요한 층이다.

**수정 후:**

```python
# score_garment_vlm
-    allow_equivalence = bool(getattr(args, "allow_garment_equivalence", False))
-    equiv = garment_equivalence_set(g, allow_equivalence=allow_equivalence)
-    verdict = pred in equiv
+    # The VLM picks one of the canonical 23; PASS requires an exact match.
+    verdict = pred == g

# assert_vocab_covers_targets
-        if not (garment_equivalence_set(g, allow_equivalence=True) & vocab):
+        if g not in vocab:
```

`equivalence_set` / `allow_equivalence` 필드를 로그 dict에서도 제거한다.

**부수 효과 — 이 삭제로 6개의 잠재적 오염이 함께 사라진다.** 감사 결과
`GARMENT_EQUIV_GROUPS`에는 `wool coat` 외에도 상위어가 5건 더 있었다:

| 그룹 | 상위어 | 문제 |
|---|---|---|
| `long coat` | `wool coat`, `wool overcoat`, `overcoat` | 피코트 포함 |
| `slacks` | `pants` | **jeans, leggings까지 포함 (23종 내 형제)** |
| `sweater` | `knitwear` | cardigan 포함 |
| `tank top` | `sleeveless top` | crop top 등 포함 |
| `fleece jacket` | `fleece` | 플리스 베스트·풀오버 포함 |
| `jeans` | `denim` | 데님 재킷·스커트 포함 |

특히 `pants` → `slacks`는 `wool coat`과 동일한 구조로, **같은 23종 안의 다른
항목을 삼킨다.**

### 3-4. `VLM_GARMENT_PROMPT` — 구별 기준을 재질에서 길이로

```diff
-    "- Outerwear, keep these distinct: a pea coat is a short double-breasted wool coat; "
-    "a wool coat reaches roughly the knee or below; a fleece jacket is a soft brushed "
+    "- Outerwear, keep these distinct. Judge coats by LENGTH and CLOSURE, never by "
+    "fabric (both coats below are usually wool): a pea coat is hip-length and "
+    "double-breasted with wide lapels; a long coat reaches the knee or below and is "
+    "usually single-breasted; a fleece jacket is a soft brushed "
```

재질로 구별하려는 한 두 범주는 겹칠 수밖에 없다. 길이와 여밈이 실제 구별 기준이다.

### 3-5. `scripts/build_faiss_index.py` — 인덱스 빌드 시 별칭 정리

79–80행:

```diff
-    "long_coat":   ["wool coat", "wool overcoat", "overcoat"],
-    "coat":        ["coat", "overcoat", "wool coat"],
+    "long_coat":   ["long coat"],
+    # generic "coat" bucket: keep only if a coarse class is genuinely needed.
+    # It must never be used to populate long_coat cells.
+    "coat":        ["coat"],
```

### 3-6. `fsiglip/` 내 중복 수집기 ★확인 필요

`fsiglip/` 안에 어휘 테이블을 가진 파일이 **3개** 있고, 셋 다 동일한 오염을 갖고 있다.

| 파일 | 오염 위치 | 상태 |
|---|---|---|
| `collector_sam3.py` | 83, 93, 935행 | **문서상 정본** (README 102·200행, `docs/SETUP.md` 99행, `docs/SETUP_FSIGLIP.md` 158행) |
| `collect_topk_sam3_fsiglip_patch_rank_vlm_garment_axis_patches_benchmark_crop.py` | 89, 99, 941행 | **`run_benchmark_crop_4gpu.sh` 174행이 실제로 호출** |
| `collect_img_sam3.py` | 72, 82, 1018행 | 헤더에 "collector_sam3.py 사본"이라 명시된 구버전 |

**문서가 가리키는 정본과 4GPU 실행 스크립트가 부르는 파일이 다르다.**

작업 방침:

1. 현재 이미지 라이브러리가 어느 경로로 수집되었는지 먼저 확정한다
   (수집 로그·manifest·`rows.jsonl`에서 스크립트명이나 인자를 찾는다).
2. **결과와 무관하게 §3-2 ~ §3-4를 세 파일 모두에 적용한다.** 하나만 고치면
   다음 재수집이 어느 경로를 타느냐에 따라 오염이 되살아난다.
3. 이후 어휘를 공용 모듈로 추출하거나 세 파일이 모두 `configs.config`를 import
   하게 해, 사본이 갈라질 수 없게 만든다.
4. `collect_img_sam3.py`는 삭제하거나 `_archive/`로 옮긴다. 남길 경우 헤더에
   `SUPERSEDED — do not run`을 명시한다.
5. `run_benchmark_crop_4gpu.sh`가 정본을 부르도록 고칠지, `benchmark_crop.py`를
   정본으로 승격할지는 저자가 결정한다. **둘 중 하나로 일원화해야 한다.**

`vit/`, `retrieval/` 경로는 은퇴했으므로 수정하지 않고 "retired — fsiglip 경로가
정본" 주석만 남긴다.

---

## 4. 재수집 절차

### 4-1. 대상

`long_coat` 75셀 + `pea_coat` 64셀 = **139셀.** 다른 21종은 검색어가 정식 명칭이므로
건드리지 않는다.

### 4-2. 순서

1. **§3의 코드 수정을 모두 적용한다.** 특히 §3-3(별칭 테이블 삭제)을 빠뜨리면
   `assert_vocab_covers_targets`가 계속 침묵해 같은 종류의 불일치를 놓친다.

2. **§3-2 적용 직후 수집기를 한 번 dry-run 한다.** 어휘가 통일되었다면
   `assert_vocab_covers_targets`가 조용히 통과해야 한다. 여기서
   `SystemExit`이 나면 아직 어딘가에 별칭이 남아 있다는 뜻이다.

3. **기존 139셀의 주석을 `pending`으로 되돌린다.** 삭제하지 말고 이전 판정을
   보존해 전후 비교가 가능하게 한다.
   ```json
   {"status": "pending", "superseded": {"status": "available", "images": [...]}}
   ```

4. **139셀에 대해서만 수집기를 재실행한다.**
   ```bash
   python -m fsiglip.collector_sam3 \
       --plans data_wacv_scenario_v5/options/option_plans.jsonl \
       --only-garments long_coat,pea_coat \
       --out <새 출력 경로>
   ```
   `--only-garments` 플래그가 없으면 추가한다. 전체 재수집은 불필요하며 다른 셀의
   확정 상태를 흔들 위험만 있다.

5. **사람이 139셀을 재검수한다.** 검수 UI 라벨이 `long coat` / `pea coat`인지
   확인할 것. 검수 지침:
   > 피코트 = 엉덩이 길이, 더블브레스트, 넓은 라펠
   > 롱코트 = 무릎 이상 길이, 대개 싱글브레스트
   > **재질이 아니라 길이로 판단할 것.** 둘 다 보통 울이다.

6. **`attribute_library.json`을 갱신하고 새 해시를 기록한다.**

### 4-3. 성공 기준

| 항목 | 현재 | 기대 |
|---|---:|---|
| `pea_coat` available | 22/64 (34%) | **상승해야 한다** |
| `pea_coat` `no_such_garment` | 30 | **감소해야 한다** |
| `long_coat` available | 64/75 (85%) | 하락 가능 — **정상이다** |

**`long_coat` 확보율이 떨어지는 것은 실패가 아니라 성공 신호일 수 있다.**
기존 85%에 피코트가 섞여 있었다면 정확한 라벨로 재수집했을 때 낮아지는 것이 옳다.
두 수치를 함께 보고할 것.

세 수치가 모두 그대로라면 오염 가설이 틀렸다는 뜻이므로, 재수집 전후의 실제
이미지를 비교해 원인을 다시 조사한다.

### 4-4. `slacks` 표본 점검 (선택)

§3-3에서 제거되는 상위어 중 `pants` → `slacks`만이 23종 내 형제 항목
(`jeans`, `leggings`)을 삼키는 구조다. 다만 `slacks`는 검색어가 정식 명칭이었고
확보율도 87%로 정상이므로 재수집 대상은 아니다. **available 셀 중 20장을 눈으로
확인해 청바지·레깅스가 섞이지 않았는지만 점검**하고, 문제가 없으면 넘어간다.

### 4-5. 재수집 후

139셀이 확정되면 v5 옵션 플랜의 실현 가능 문항 수가 달라진다. `--cell-library`로
**옵션 플랜을 재생성**해야 하며, 기존 플랜을 재활용해서는 안 된다.
밸런스·카운터밸런스 리포트도 새 manifest 기준으로 재산출한다.

---

## 5. 논문에 남길 기록

`configs/scenarios.py`나 `docs/` 개정 이력에 다음 취지로 기록한다.

> `long_coat` 셀의 검색어 및 VLM 판정 라벨로 `wool coat`을 사용했으나, 이는
> 동의어가 아니라 피코트를 포함하는 상위 범주다. 그 결과 `long_coat` 셀이 피코트
> 후보를 흡수하고 `pea_coat` 셀의 확보율이 전체 최하위권(34%)으로 떨어졌다.
> garment 어휘를 `configs/config.py`의 정식 23종으로 통일하고, 별칭 계층
> (`GARMENT_EQUIV_GROUPS`)을 제거했으며 — 이 테이블은 VLM 판정에는 기여하지 않으면서
> 어휘 불일치 탐지만 무력화하고 있었다 — 두 코트의 구별 기준을 재질에서 길이·여밈으로
> 변경한 뒤 해당 139셀을 재수집·재검수했다.

리뷰어 대응 관점에서 이 기록은 **약점이 아니라 강점**이다. 어휘 계층 오류를 스스로
발견하고 수정했다는 것은 데이터 품질 관리가 작동했다는 증거다. 숨기지 말고
supplementary에 전후 수치를 함께 실을 것.

---

## 6. 하지 말 것

- 139셀 외의 재수집 — 다른 21종은 검색어가 정식 명칭이며 영향이 없다
- `DEFAULT_GARMENTS`에 별칭을 "혹시 몰라서" 남겨두기 — 실측으로 0건 사용이
  확인되었고, 남겨두면 상위어가 다시 섞여 들어올 통로가 된다
- `GARMENT_EQUIV_GROUPS`를 "정리해서 유지" — 진짜 동의어만 남겨도 결국
  `assert_vocab_covers_targets`의 탐지력을 낮춘다. **전체 삭제가 맞다**
- 시각적으로 혼동되는 아우터(blazer/cardigan, windbreaker/puffer/leather)에 대한
  별칭 추가 — 의도적으로 구별하게 둔 것이다
- `vit/`, `retrieval/` 수정 — 은퇴한 파이프라인이다.
  **단 `fsiglip/` 내 3개 파일은 모두 수정 대상이다 (§3-6)**
- 재수집 결과가 나오기 전에 v5 최종 산출물 동결
