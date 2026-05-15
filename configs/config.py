"""
POD-Bench 중앙 설정.

attribute taxonomy, TPO 축, 파이프라인 파라미터를 한 곳에서 관리합니다.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PROFILES_DIR = DATA_DIR / "profiles"
QUERIES_DIR = DATA_DIR / "queries"
OPTIONS_DIR = DATA_DIR / "options"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
FINAL_DIR = DATA_DIR / "final"

for d in [PROFILES_DIR, QUERIES_DIR, OPTIONS_DIR, IMAGES_DIR, LABELS_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 도메인 구성
# ─────────────────────────────────────────────
DOMAINS = {
    "fashion": {"weight": 0.7, "n_users_phase1": 35},
    "interior": {"weight": 0.3, "n_users_phase1": 15},
}


# ─────────────────────────────────────────────
# Attribute Taxonomy (도메인별)
# ─────────────────────────────────────────────
# 이 taxonomy는 user profile의 structured ground-truth 생성과
# 자동 라벨러의 매칭에 모두 사용됩니다.

FASHION_ATTRIBUTES = {
    "color_palette": [
        "muted_neutral",      # 베이지/그레이/카키 등
        "deep_dark",          # 검정/네이비/짙은 브라운
        "high_saturation",    # 빨강/주황/노랑 등 채도 높음
        "pastel_soft",        # 파스텔 톤
        "monochrome",         # 흑백
    ],
    "silhouette": [
        "relaxed",            # 여유로운 핏
        "fitted",             # 슬림한 핏
        "structured",         # 각진/정형
        "oversized",          # 의도적으로 큰 핏
    ],
    "material": [
        "natural_fiber",      # 면/린넨/울
        "synthetic",          # 폴리에스터/나일론
        "leather_suede",
        "denim",
        "knit",
    ],
    "style_identity": [
        "minimalist",
        "classic",
        "streetwear",
        "smart_casual",
        "formal",
        "sporty",
        "bohemian",
    ],
    "detail_load": [
        "plain",              # 디테일 최소
        "subtle_detail",
        "heavy_decoration",   # 큰 로고, 화려한 장식
    ],
    "pattern": [
        "solid",
        "small_pattern",
        "large_pattern",
        "graphic_print",
    ],
    "formality": [
        "very_casual",
        "casual",
        "smart_casual",
        "semi_formal",
        "formal",
    ],
}

INTERIOR_ATTRIBUTES = {
    "color_palette": [
        "warm_wood",          # 따뜻한 우드톤
        "cool_minimal",       # 화이트/그레이/블랙
        "earthy_natural",     # 베이지/테라코타/올리브
        "bold_colored",       # 강렬한 색감
        "scandinavian_light", # 라이트 우드 + 화이트
    ],
    "style_identity": [
        "minimalist",
        "scandinavian",
        "industrial",
        "mid_century_modern",
        "traditional",
        "rustic_farmhouse",
        "bohemian",
        "modern_contemporary",
    ],
    "material": [
        "natural_wood",
        "metal_industrial",
        "fabric_upholstered",
        "leather",
        "glass_acrylic",
        "rattan_woven",
    ],
    "form": [
        "clean_geometric",    # 직선/정형
        "curved_organic",     # 곡선/유기적
        "ornate_detailed",    # 장식적
    ],
    "size_scale": [
        "compact",
        "standard",
        "oversized_statement",
    ],
}

ATTRIBUTES_BY_DOMAIN = {
    "fashion": FASHION_ATTRIBUTES,
    "interior": INTERIOR_ATTRIBUTES,
}


# ─────────────────────────────────────────────
# TPO 축
# ─────────────────────────────────────────────
FASHION_TPO = {
    "time": {
        "season": ["spring", "summer", "fall", "winter"],
        "time_of_day": ["morning", "daytime", "evening", "night"],
        "weather": ["sunny", "rainy", "snowy", "windy", "hot_humid", "cold"],
    },
    "place": {
        "indoor_outdoor": ["indoor", "outdoor", "mixed"],
        "venue": ["office", "cafe", "park", "beach", "mountain", "city_street",
                  "restaurant", "wedding_venue", "gym", "home"],
        "formality_required": ["very_casual", "casual", "smart_casual",
                                "semi_formal", "formal"],
    },
    "occasion": {
        "activity": ["work", "exercise", "leisure", "social", "travel",
                     "errand", "date", "ceremony"],
        "social_role": ["alone", "with_friends", "with_family",
                         "with_colleagues", "with_strangers"],
        "intensity": ["everyday", "special", "intensive"],
    },
}

INTERIOR_TPO = {
    "time": {
        "time_of_day": ["morning_workspace", "daytime_living",
                         "evening_relaxation", "night_sleep"],
        "season": ["all_season", "summer_cooling", "winter_warming"],
    },
    "place": {
        "room_type": ["living_room", "bedroom", "study_workspace",
                       "dining", "kitchen", "small_apartment", "large_home"],
        "lighting_condition": ["bright_natural", "low_light", "artificial_only"],
    },
    "occasion": {
        "primary_use": ["work_from_home", "entertaining_guests",
                         "family_daily", "solo_living", "shared_space"],
        "user_composition": ["single", "couple", "family_with_kids",
                              "shared_with_roommates", "senior_living"],
    },
}

TPO_BY_DOMAIN = {
    "fashion": FASHION_TPO,
    "interior": INTERIOR_TPO,
}


# ─────────────────────────────────────────────
# Query 유형별 비율
# ─────────────────────────────────────────────
QUERY_TYPE_RATIO = {
    "explicit_tpo": 0.30,    # "비 오는 날 외출용"
    "implicit_tpo": 0.25,    # "야외 결혼식"
    "visual_tpo": 0.30,      # 모호 텍스트 + TPO 이미지
    "neutral": 0.15,         # TPO 없음
}


# ─────────────────────────────────────────────
# 선지 라벨
# ─────────────────────────────────────────────
OPTION_LABELS = {
    "A": "tpo_and_preference",  # TPO+선호 모두 충족 (정답)
    "B": "tpo_only",            # TPO 충족, 선호 위반
    "C": "preference_only",     # 선호 충족, TPO 위반
    "D": "neither",             # 둘 다 위반
}


# ─────────────────────────────────────────────
# 파이프라인 파라미터
# ─────────────────────────────────────────────
PHASE1_CONFIG = {
    "n_users_total": 50,
    "n_queries_per_user": 20,
    "n_options_per_query": 4,
    "target_instances": 1000,  # 50 * 20
}


# ─────────────────────────────────────────────
# 모델 / API 설정
# ─────────────────────────────────────────────
# 유료: GPT-5-mini만 사용
PAID_MODELS = {
    "gpt5_mini": {
        "model_name": "gpt-5-mini",
        "use_for": ["profile_generation", "query_generation",
                    "option_planning", "judge_ensemble_member_1"],
    },
}

# 무료: 로컬 vLLM 서빙
FREE_MODELS = {
    "local_judge": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "api_base": "http://localhost:8000/v1",
        "use_for": ["judge_ensemble_member_2"],
    },
    "local_judge_alt": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "api_base": "http://localhost:8001/v1",
        "use_for": ["judge_ensemble_member_3"],
    },
    "clip_encoder": {
        "model_name": "openai/clip-vit-large-patch14",
        "use_for": ["image_embedding", "homogeneity_check"],
    },
    "vlm_evaluator": {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "api_base": "http://localhost:8000/v1",
        "use_for": ["benchmark_evaluation"],
    },
}


# ─────────────────────────────────────────────
# 라벨 신뢰도 임계값
# ─────────────────────────────────────────────
LABEL_QUALITY_THRESHOLDS = {
    "krippendorff_alpha_min": 0.7,
    "cohen_kappa_min": 0.6,
    "judge_agreement_min": 2,  # 3개 judge 중 최소 2개 합의 필요
}

# Vision-essentiality 임계값
VISION_ESSENTIALITY_THRESHOLDS = {
    "blind_llm_max_acc": 0.40,      # text-only LLM 정답률 상한
    "captioner_llm_max_acc": 0.45,  # captioner→LLM 정답률 상한
    "full_vlm_min_acc": 0.55,       # full VLM 정답률 하한
}


# ─────────────────────────────────────────────
# Image collection
# ─────────────────────────────────────────────
IMAGE_COLLECTION = {
    "primary_source": "amazon_reviews_2023",
    "secondary_source": "google_search",
    "google_api_daily_quota": 100,
    "min_resolution": (224, 224),
    "preferred_resolution": (512, 512),
    "max_clip_distance_within_options": 0.45,  # 4선지 간 최대 시각 거리
}
