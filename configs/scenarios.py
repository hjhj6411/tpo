"""
POD-Bench v2 — Canonical Extreme Scenario Catalog (HARDCODED, 15 archetypes)

Design: "Archetype x Specificity Gradient" (PCogAlignBench, SocialIQA, RVBench).
Every scenario encodes an INDISPUTABLE TPO contrast — the incompatible garments
(and, for dress-coded archetypes, colors/patterns) are wrong on grounds of
physical danger, functional impossibility, or social norm under the
declared cultural frame (CULTURAL_FRAME, below).

15 archetypes (was 8). Two families:

  PHYSICAL (garment is the only TPO-constrained axis; color/pattern are pure
  preference axes via the relaxed compatibility check):
    extreme_cold, extreme_heat, aquatic_water, athletic_indoor,
    athletic_outdoor, rugged_outdoor, severe_weather, casual_leisure

  DRESS-CODED (garment AND color AND/OR pattern carry TPO meaning):
    business_professional, ultra_formal, judicial_civic, mourning_somber,
    religious_modest, semi_formal_social, wedding_celebration

WHY 15 archetypes now all produce instances
--------------------------------------------
In the clean 2x2 the active axis is color/pattern (PREFERENCE) and garment is
the TPO axis. A physical scenario (e.g. blizzard) does not constrain color, so
ANY liked color is situation-appropriate — color is then a *pure* preference
probe while the garment alone carries TPO. This is the cleanest 2x2 instance and
is what makes extreme_cold / heat / athletic / etc. contribute (they previously
produced 0 because color/pattern were unconstrained). The relaxed
`check_axis_compatibility` treats an unconstrained active axis as "all values
TPO-compatible". Dress-coded archetypes are unchanged (their color/pattern
constraints still restrict A/B to compatible values).

Instance generation:
  scenario x active_axis(color|pattern) x compatible_user -> one benchmark instance
"""

# ═══════════════════════════════════════════════════════════
#  CULTURAL FRAME  (scope condition for normative dress codes)
# ═══════════════════════════════════════════════════════════
# All *normative* dress codes (business, ultra-formal, judicial/civic, mourning,
# religious, semi-formal, wedding) are defined under ONE explicit convention set:
# contemporary Western / Euro-American norms. Rationale: the image pool is drawn
# from the (predominantly US) Amazon Reviews 2023 corpus, so the normative frame
# is aligned to the visual distribution, removing an image-label culture
# mismatch. We do NOT claim universality — e.g. white/red wedding & mourning
# conventions differ in parts of East/South Asia (here white is intentionally
# NEUTRAL for mourning, since it is not Western mourning attire). This is a
# stated scope condition, not a universality claim. The same frame is stated to
# the model at eval time via EVAL_FRAME_CLAUSE, so the task probes occasion
# reasoning within a declared convention rather than guessing the annotator's
# culture. PHYSICAL archetypes (cold/heat/water/athletic/rugged/weather/casual)
# are culture-invariant; their color & pattern stay preference-only.
CULTURAL_FRAME = "contemporary_western"
EVAL_FRAME_CLAUSE = (
    "Assume contemporary Western / international dress-code conventions "
    "when judging what is appropriate for the occasion."
)
# Archetypes whose color/pattern norms are governed by CULTURAL_FRAME.
FRAME_SCOPED_ARCHETYPES = {
    "business_professional", "ultra_formal", "judicial_civic",
    "mourning_somber", "religious_modest", "semi_formal_social",
    "wedding_celebration",
}


# ═══════════════════════════════════════════════════════════
#  Helper to build scenario dicts compactly (same signature as v2.0)
# ═══════════════════════════════════════════════════════════

def _sc(sid, arch, name, tpo,
        gc_compat, gc_incompat,
        co_compat=None, co_incompat=None,
        pa_compat=None, pa_incompat=None,
        q_explicit=None, q_implicit=None,
        justification=""):
    sc = {
        "scenario_id": sid, "archetype": arch, "name": name, "tpo": tpo,
        "garment_category": {"compatible": gc_compat,
                             "incompatible": gc_incompat},
        "justification": justification,
    }
    sc["color"] = ({"compatible": co_compat, "incompatible": co_incompat}
                   if (co_compat or co_incompat) else None)
    sc["pattern"] = ({"compatible": pa_compat, "incompatible": pa_incompat}
                     if (pa_compat or pa_incompat) else None)
    sc["query_seeds"] = {"explicit": q_explicit or [], "implicit": q_implicit or []}
    return sc


# ═══════════════════════════════════════════════════════════
#  ARCHETYPE REGISTRY (15)
# ═══════════════════════════════════════════════════════════

SCENARIO_ARCHETYPES = {
    # physical (color/pattern unconstrained → pure-preference active axis)
    "extreme_cold":          "Extreme Cold / Winter",
    "extreme_heat":          "Extreme Heat / Summer",
    "aquatic_water":         "Water & Swimming",
    "athletic_indoor":       "Indoor Athletic / Gym",
    "athletic_outdoor":      "Outdoor Athletic / Field Sports",
    "rugged_outdoor":        "Rugged Outdoor / Hiking & Camping",
    "severe_weather":        "Severe Weather Events",
    "casual_leisure":        "Casual Leisure / Everyday Outing",
    # dress-coded (garment + color/pattern carry TPO)
    "business_professional": "Business / Professional",
    "ultra_formal":          "Ultra-Formal / Ceremonial",
    "judicial_civic":        "Judicial / Civic / Official",
    "mourning_somber":       "Mourning / Somber",
    "religious_modest":      "Religious / Sacred / Modest",
    "semi_formal_social":    "Semi-Formal Social",
    "wedding_celebration":   "Wedding / Celebration",
}

# shared garment palettes for physical archetypes (kept consistent so the
# TPO contrast is identical across scenarios within an archetype)
_COLD_C  = ["parka", "coat", "jacket", "sweater", "hoodie"]  # trench_coat/windbreaker removed: thin shells inadequate for sub-zero (physical)
_COLD_I  = ["shorts", "tank_top", "t_shirt", "dress", "skirt"]
_HEAT_C  = ["t_shirt", "tank_top", "shorts"]
_HEAT_I  = ["parka", "coat", "trench_coat", "sweater", "suit_jacket", "blazer"]
_WATER_C = ["t_shirt", "tank_top", "shorts"]
_WATER_I = ["coat", "blazer", "suit_jacket", "parka", "sweater", "trench_coat", "dress"]
_GYM_C   = ["t_shirt", "tank_top", "shorts", "hoodie", "sweater"]
_GYM_I   = ["suit_jacket", "blazer", "coat", "dress", "trench_coat", "parka"]
_FIELD_C = ["t_shirt", "tank_top", "shorts", "jacket", "windbreaker", "hoodie"]
_FIELD_I = ["coat", "blazer", "suit_jacket", "dress", "parka", "trench_coat"]
_RUGGED_C = ["jacket", "windbreaker", "hoodie", "jeans", "t_shirt", "sweater", "parka"]
_RUGGED_I = ["suit_jacket", "blazer", "dress", "skirt", "blouse"]
_STORM_C = ["windbreaker", "jacket", "coat", "parka"]
_STORM_I = ["dress", "tank_top", "shorts", "skirt"]
_CASUAL_C = ["t_shirt", "shorts", "hoodie", "jeans", "sweater", "jacket"]
_CASUAL_I = ["suit_jacket", "blazer", "trench_coat"]


# ═══════════════════════════════════════════════════════════
#  CANONICAL SCENARIOS
# ═══════════════════════════════════════════════════════════

CANONICAL_SCENARIOS = [

    # ════════ 1. EXTREME COLD (4) — physical ════════
    _sc("cold_blizzard_outdoor", "extreme_cold",
        "Blizzard & Sub-zero Outdoor All-day",
        {"time": {"season": "winter", "weather": "snowy"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "travel", "formality_required": "very_casual"}},
        gc_compat=_COLD_C, gc_incompat=_COLD_I,
        q_explicit=[
            "It's going to be a blizzard with sub-zero temperatures and I'll be outside all day. What should I wear?",
            "Heavy snow and freezing winds are forecast and I'll be outdoors for hours. What outfit makes sense?",
        ],
        q_implicit=[
            "I'm heading to a winter festival up in the mountains this weekend. Any outfit advice?",
            "Spending the whole day at an outdoor holiday market in January. What should I throw on?",
        ],
        justification="Sub-zero blizzard makes light/exposed clothing a hypothermia risk."),

    _sc("cold_polar_expedition", "extreme_cold",
        "Polar / Arctic Expedition Day",
        {"time": {"season": "winter", "weather": "snowy"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "travel", "formality_required": "very_casual"}},
        gc_compat=_COLD_C, gc_incompat=_COLD_I,
        q_explicit=[
            "I'm joining an arctic expedition and will be outdoors in extreme cold all day. What should I wear?",
            "Going out onto the polar ice in freezing wind for the whole day. What's the right outfit?",
        ],
        q_implicit=[
            "Booked a trip to a polar research base for some fieldwork. What do I pack to wear?",
            "Heading way up north for a glacier trek next week. Outfit thoughts?",
        ],
        justification="Polar exposure without heavy insulation causes frostbite/hypothermia."),

    _sc("cold_ski_slope", "extreme_cold",
        "All-day Skiing on the Slopes",
        {"time": {"season": "winter", "weather": "snowy"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "exercise", "formality_required": "very_casual"}},
        gc_compat=_COLD_C, gc_incompat=_COLD_I,
        q_explicit=[
            "I'll be skiing on the slopes all day in freezing alpine weather. What should I wear?",
            "Spending the day on a snowy mountain skiing in sub-zero temps. Outfit advice?",
        ],
        q_implicit=[
            "Got a lift pass for the resort tomorrow and I'll be on the mountain till dark. What should I wear?",
            "First time hitting the slopes this season. What outfit makes sense?",
        ],
        justification="Skiing in alpine cold rules out exposed-skin clothing."),

    _sc("cold_ice_fishing", "extreme_cold",
        "Frozen-Lake Ice Fishing",
        {"time": {"season": "winter", "weather": "cold"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "casual_outing", "formality_required": "very_casual"}},
        gc_compat=_COLD_C, gc_incompat=_COLD_I,
        q_explicit=[
            "Sitting out on a frozen lake ice fishing for hours in sub-zero cold. What should I wear?",
            "I'll be still on the ice in freezing wind all morning. What's the warmest sensible outfit?",
        ],
        q_implicit=[
            "Going ice fishing with my uncle this weekend. What should I wear?",
            "Planning a quiet day out on the frozen lake. Outfit advice?",
        ],
        justification="Hours motionless on ice in freezing air requires maximum insulation."),

    # ════════ 2. EXTREME HEAT (4) — physical ════════
    _sc("heat_beach_day", "extreme_heat",
        "Scorching Beach Day",
        {"time": {"season": "summer", "weather": "hot_humid"},
         "place": {"venue": "beach", "indoor_outdoor": "outdoor"},
         "occasion": {"activity": "casual_outing", "formality_required": "very_casual"}},
        gc_compat=_HEAT_C, gc_incompat=_HEAT_I,
        q_explicit=[
            "It's 38°C and I'm heading to the beach for the whole day. What should I wear?",
            "Brutal heat today and I'll be on the sand for hours. What outfit makes sense?",
        ],
        q_implicit=[
            "Planning a beach day this weekend at the peak of summer. Outfit?",
            "Meeting friends down by the shore tomorrow afternoon. What should I wear?",
        ],
        justification="Heavy outerwear at a scorching beach causes heatstroke risk."),

    _sc("heat_desert_trek", "extreme_heat",
        "Desert Trek (Peak Summer)",
        {"time": {"season": "summer", "weather": "hot_humid"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "travel", "formality_required": "very_casual"}},
        gc_compat=_HEAT_C, gc_incompat=_HEAT_I,
        q_explicit=[
            "Trekking across a desert in 42°C heat. What should I wear?",
            "Out in the open desert under blazing sun all day. What's the right outfit?",
        ],
        q_implicit=[
            "Spending the day exploring Dubai in August. Outfit advice?",
            "Doing a guided tour through the dunes next week. What should I wear?",
        ],
        justification="42°C desert heat in a parka/coat is a heatstroke emergency."),

    _sc("heat_tropical_resort", "extreme_heat",
        "Tropical Resort Pool Party",
        {"time": {"season": "summer", "weather": "hot_humid"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "social_gathering", "formality_required": "very_casual"}},
        gc_compat=_HEAT_C, gc_incompat=_HEAT_I,
        q_explicit=[
            "Pool party at a tropical resort in sweltering heat. What should I wear?",
            "It's humid and boiling and I'm headed to a poolside party. Outfit?",
        ],
        q_implicit=[
            "Friends are throwing a poolside get-together at the resort. What should I wear?",
            "Spending the afternoon lounging by the resort pool. Outfit advice?",
        ],
        justification="Parka/suit jacket at a tropical pool party is physically unbearable."),

    _sc("heat_summer_festival", "extreme_heat",
        "Midsummer Outdoor Music Festival",
        {"time": {"season": "summer", "weather": "hot_humid"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "social_gathering", "formality_required": "very_casual"}},
        gc_compat=_HEAT_C, gc_incompat=_HEAT_I,
        q_explicit=[
            "Going to an all-day outdoor music festival in 35°C heat. What should I wear?",
            "Standing in a festival crowd under the sun from noon to night. Outfit advice?",
        ],
        q_implicit=[
            "Got tickets to a summer festival this weekend. What should I wear?",
            "Spending Saturday at an open-air concert downtown. Outfit?",
        ],
        justification="All-day outdoor festival in extreme heat makes heavy outerwear a health hazard."),

    # ════════ 3. AQUATIC / WATER (3) — physical ════════
    _sc("water_swim_session", "aquatic_water",
        "Swimming Pool Lap Session",
        {"place": {"venue": "pool", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "exercise", "formality_required": "very_casual"}},
        gc_compat=_WATER_C, gc_incompat=_WATER_I,
        q_explicit=[
            "Heading to the pool for a lap-swimming session. What should I wear?",
            "Going to swim laps for an hour at the pool. What's the right outfit?",
        ],
        q_implicit=[
            "Starting regular swim training this week. What should I bring to wear?",
            "Joined the local pool for morning swims. Outfit advice?",
        ],
        justification="A parka/blazer/dress to swim laps is absurd and waterlogged-dangerous."),

    _sc("water_park_day", "aquatic_water",
        "Water Park Visit (Peak Summer)",
        {"time": {"season": "summer", "weather": "hot_humid"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "casual_outing", "formality_required": "very_casual"}},
        gc_compat=_WATER_C, gc_incompat=_WATER_I,
        q_explicit=[
            "Taking the family to a water park in scorching heat. What should I wear?",
            "Spending the day going down water slides in the sun. Outfit?",
        ],
        q_implicit=[
            "Day out at the water park this weekend. What should I wear?",
            "Kids want to hit the aqua park tomorrow. Outfit advice?",
        ],
        justification="Coat/parka/dress at a water park in summer heat is nonsensical and unsafe."),

    _sc("water_poolside_lounge", "aquatic_water",
        "Resort Poolside Lounging",
        {"time": {"season": "summer", "weather": "hot_humid"},
         "place": {"venue": "pool", "indoor_outdoor": "outdoor"},
         "occasion": {"activity": "casual_outing", "formality_required": "very_casual"}},
        gc_compat=_WATER_C, gc_incompat=_WATER_I,
        q_explicit=[
            "Lounging poolside all afternoon in the heat, going in and out of the water. What should I wear?",
            "Spending a hot day by the pool with frequent dips. Outfit?",
        ],
        q_implicit=[
            "Relaxing by the hotel pool tomorrow afternoon. What should I wear?",
            "Booked a cabana by the pool for the day. Outfit advice?",
        ],
        justification="Heavy structured garments by a pool in summer heat are impractical and get soaked."),

    # ════════ 4. ATHLETIC INDOOR (3) — physical ════════
    _sc("gym_weight_training", "athletic_indoor",
        "Gym Weight Training",
        {"place": {"venue": "gym", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "exercise", "formality_required": "very_casual"}},
        gc_compat=_GYM_C, gc_incompat=_GYM_I,
        q_explicit=[
            "Heading to the gym for a heavy weight-training session. What should I wear?",
            "Doing barbell and machine work at the gym today. What's the right outfit?",
        ],
        q_implicit=[
            "Got a lifting session after work today. What should I wear?",
            "Starting a strength program at the gym this week. Outfit advice?",
        ],
        justification="Weight training in a suit/coat/dress restricts movement and damages garments."),

    _sc("gym_yoga_class", "athletic_indoor",
        "Yoga / Pilates Class",
        {"place": {"venue": "gym", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "exercise", "formality_required": "very_casual"}},
        gc_compat=_GYM_C, gc_incompat=_GYM_I,
        q_explicit=[
            "I have a yoga class tonight with lots of stretching and floor poses. What should I wear?",
            "Going to a Pilates session focused on full-body mobility. Outfit?",
        ],
        q_implicit=[
            "Starting yoga this week. What's appropriate to wear?",
            "Signed up for a mat-based fitness class. What should I wear?",
        ],
        justification="Yoga requires full-body flexibility; blazers/coats/dresses physically prevent poses."),

    _sc("gym_indoor_basketball", "athletic_indoor",
        "Indoor Basketball / Futsal",
        {"place": {"venue": "gym", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "exercise", "formality_required": "very_casual"}},
        gc_compat=_GYM_C, gc_incompat=_GYM_I,
        q_explicit=[
            "Playing indoor basketball tonight with lots of running and jumping. What should I wear?",
            "Got a futsal match on an indoor court this evening. Outfit?",
        ],
        q_implicit=[
            "Joining a pick-up league at the indoor court. What's right to wear?",
            "Friends booked the gym court for a game tonight. Outfit advice?",
        ],
        justification="Court sport in a blazer/coat/dress restricts movement and risks injury."),

    # ════════ 5. ATHLETIC OUTDOOR (3) — physical ════════
    _sc("field_road_run", "athletic_outdoor",
        "Long-distance Road Run",
        {"place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "exercise", "formality_required": "very_casual"}},
        gc_compat=_FIELD_C, gc_incompat=_FIELD_I,
        q_explicit=[
            "Running a half-marathon this weekend. What should I wear?",
            "Going out for a long training run on the road today. Outfit?",
        ],
        q_implicit=[
            "Training for my first marathon. What's the right thing to wear?",
            "Building up my distance with weekend runs. Outfit advice?",
        ],
        justification="Running distance in a coat/blazer/dress is physically impossible and overheating."),

    _sc("field_tennis_match", "athletic_outdoor",
        "Outdoor Tennis Match",
        {"place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "exercise", "formality_required": "very_casual"}},
        gc_compat=_FIELD_C, gc_incompat=_FIELD_I,
        q_explicit=[
            "Tennis match on an outdoor court this afternoon. What should I wear?",
            "Playing a couple of sets in the sun today. Outfit?",
        ],
        q_implicit=[
            "Joined a weekend tennis club. What should I wear on court?",
            "Booked a court with a friend for tomorrow. Outfit advice?",
        ],
        justification="Playing tennis in a coat/parka/dress is physically impossible."),

    _sc("field_soccer_match", "athletic_outdoor",
        "Outdoor Soccer Match",
        {"place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "exercise", "formality_required": "very_casual"}},
        gc_compat=_FIELD_C, gc_incompat=_FIELD_I,
        q_explicit=[
            "Playing a full soccer match on grass this weekend. What should I wear?",
            "Out on the pitch running for 90 minutes today. Outfit?",
        ],
        q_implicit=[
            "Joined a Sunday-league soccer team. What should I wear to play?",
            "Friends organized a kickabout at the park field. Outfit advice?",
        ],
        justification="Running a soccer match in a coat/blazer/dress is impossible and tears garments."),

    # ════════ 6. RUGGED OUTDOOR (3) — physical ════════
    _sc("rugged_mountain_hike", "rugged_outdoor",
        "Day Hike on Rough Mountain Trail",
        {"place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "travel", "formality_required": "very_casual"}},
        gc_compat=_RUGGED_C, gc_incompat=_RUGGED_I,
        q_explicit=[
            "Hiking a rough mountain trail all day with steep, rocky sections. What should I wear?",
            "Doing a long day hike over uneven terrain. What's the right outfit?",
        ],
        q_implicit=[
            "Planning a mountain hike with friends this weekend. Outfit advice?",
            "Heading up the ridge trail on Saturday. What should I wear?",
        ],
        justification="A suit/blazer/dress/skirt on a rugged hike is impractical and easily damaged."),

    _sc("rugged_wilderness_camping", "rugged_outdoor",
        "Wilderness Camping Trip",
        {"place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "travel", "formality_required": "very_casual"}},
        gc_compat=_RUGGED_C, gc_incompat=_RUGGED_I,
        q_explicit=[
            "Camping in the backcountry for the weekend, setting up tents and gathering wood. What should I wear?",
            "Roughing it outdoors for two days of camping. Outfit?",
        ],
        q_implicit=[
            "Planning a camping trip out in the woods with friends. What should I wear?",
            "Heading off-grid to a campsite this weekend. Outfit advice?",
        ],
        justification="Camping chores in a blazer/suit/dress are impractical and ruin the garment."),

    _sc("rugged_trail_scramble", "rugged_outdoor",
        "Rocky Trail Scramble",
        {"place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "travel", "formality_required": "very_casual"}},
        gc_compat=_RUGGED_C, gc_incompat=_RUGGED_I,
        q_explicit=[
            "Scrambling over boulders and rough rock on a trail today. What should I wear?",
            "Doing a hands-on rocky scramble route. What's the right outfit?",
        ],
        q_implicit=[
            "Going bouldering along a rugged trail this weekend. Outfit advice?",
            "Tackling a steep, rocky route with a group. What should I wear?",
        ],
        justification="A rocky scramble in formal/delicate clothing is a safety hazard."),

    # ════════ 7. SEVERE WEATHER (4) — physical ════════
    _sc("weather_typhoon", "severe_weather",
        "Typhoon / Hurricane Errands",
        {"time": {"weather": "rainy", "wind": "extreme"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "errands"}},
        gc_compat=_STORM_C, gc_incompat=_STORM_I,
        q_explicit=[
            "A typhoon is hitting and I have to go out for urgent errands. What should I wear?",
            "Driving rain and violent wind outside and I must head out. Outfit?",
        ],
        q_implicit=[
            "Need to run essential errands during the big storm. What should I wear?",
            "The storm's bad but I have to get to the store. Outfit advice?",
        ],
        justification="Typhoon-force wind and rain makes exposed-skin/delicate clothing dangerous."),

    _sc("weather_hailstorm", "severe_weather",
        "Hailstorm Outdoor Exposure",
        {"time": {"weather": "hail"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "errands"}},
        gc_compat=_STORM_C, gc_incompat=_STORM_I,
        q_explicit=[
            "There's a hailstorm and I have to walk home through it. What should I wear?",
            "Hail is coming down and I need to be outside briefly. What's the right outfit?",
        ],
        q_implicit=[
            "Caught out in sudden hail last time — what should I wear when it happens again?",
            "Forecast says hail and I still need to head out. Outfit advice?",
        ],
        justification="Hailstones on bare skin cause injury; protective outerwear is essential."),

    _sc("weather_dust_storm", "severe_weather",
        "Sandstorm / Dust Storm Outdoor",
        {"time": {"weather": "windy", "air": "dust"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "errands"}},
        gc_compat=_STORM_C, gc_incompat=_STORM_I,
        q_explicit=[
            "A dust storm is rolling in and I need to cross town. What should I wear?",
            "Thick blowing sand outside and I have to be out in it. Outfit?",
        ],
        q_implicit=[
            "Living in a desert city during sandstorm season — what should I wear outdoors?",
            "The air's full of blowing dust and I need to head out. Outfit advice?",
        ],
        justification="Sand/dust on bare skin causes abrasion; full coverage is needed."),

    _sc("weather_freezing_rain", "severe_weather",
        "Freezing Rain / Ice Storm Commute",
        {"time": {"season": "winter", "weather": "rainy"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "travel"}},
        gc_compat=_STORM_C, gc_incompat=["dress", "tank_top", "shorts", "skirt", "t_shirt"],
        q_explicit=[
            "Freezing rain and icy roads, and I'm walking to work. What should I wear?",
            "An ice storm is making my commute treacherous and wet. Outfit?",
        ],
        q_implicit=[
            "The roads are iced over and I still have to get across town. What should I wear?",
            "Sleety, freezing commute ahead this morning. Outfit advice?",
        ],
        justification="Freezing rain plus sub-zero wind chill makes light clothing hypothermia-inducing."),

    # ════════ 8. CASUAL LEISURE (4) — physical ════════
    _sc("casual_park_picnic", "casual_leisure",
        "Weekend Park Picnic",
        {"time": {"season": "spring", "weather": "sunny"},
         "place": {"venue": "park", "indoor_outdoor": "outdoor"},
         "occasion": {"activity": "casual_outing", "formality_required": "very_casual"}},
        gc_compat=_CASUAL_C, gc_incompat=_CASUAL_I,
        q_explicit=[
            "Having a relaxed picnic at the park on a sunny afternoon. What should I wear?",
            "Sitting on the grass with friends for a casual park hangout. Outfit?",
        ],
        q_implicit=[
            "Meeting friends at the park for a lazy Sunday. What should I wear?",
            "Packing a basket for an afternoon in the park. Outfit advice?",
        ],
        justification="A suit jacket/blazer/trench coat at a casual park picnic is universally over-dressed."),

    _sc("casual_farmers_market", "casual_leisure",
        "Weekend Farmers' Market",
        {"time": {"season": "spring", "weather": "sunny"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "casual_outing", "formality_required": "very_casual"}},
        gc_compat=_CASUAL_C, gc_incompat=_CASUAL_I,
        q_explicit=[
            "Strolling a weekend farmers' market for a couple of hours. What should I wear?",
            "Wandering the outdoor market stalls on a casual Saturday. Outfit?",
        ],
        q_implicit=[
            "Heading to the local market for produce this morning. What should I wear?",
            "Casual morning browsing the market. Outfit advice?",
        ],
        justification="A suit/blazer at a casual farmers' market is over-dressed."),

    _sc("casual_amusement_park", "casual_leisure",
        "Amusement Park All-Day Visit",
        {"place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "casual_outing", "formality_required": "very_casual"}},
        gc_compat=_CASUAL_C, gc_incompat=["suit_jacket", "blazer", "trench_coat", "coat"],
        q_explicit=[
            "Spending the whole day at an amusement park going on rides. What should I wear?",
            "On my feet all day at a theme park with lots of rides. Outfit?",
        ],
        q_implicit=[
            "Taking the family to the theme park this weekend. What should I wear?",
            "Day out at the amusement park tomorrow. Outfit advice?",
        ],
        justification="Formal wear at an amusement park restricts rides and movement."),

    _sc("casual_zoo_day", "casual_leisure",
        "Family Day at the Zoo",
        {"place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "casual_outing", "formality_required": "very_casual"}},
        gc_compat=_CASUAL_C, gc_incompat=["suit_jacket", "blazer", "trench_coat"],
        q_explicit=[
            "Walking around the zoo all day with the family. What should I wear?",
            "A full day of walking outdoors at the zoo. Outfit?",
        ],
        q_implicit=[
            "Taking the kids to the zoo this weekend. What should I wear?",
            "Day trip to the zoo tomorrow. Outfit advice?",
        ],
        justification="Formal attire for a full day of walking at the zoo is impractical and over-dressed."),

    # ════════ 9. BUSINESS / PROFESSIONAL (4) — dress-coded ════════
    _sc("biz_corporate_board", "business_professional",
        "Corporate Board Meeting",
        {"time": {"time_of_day": "daytime"},
         "place": {"venue": "office", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "work_meeting", "formality_required": "formal"}},
        gc_compat=["suit_jacket", "blazer", "shirt", "blouse"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray", "white", "beige"],
        co_incompat=["orange", "yellow", "pink"],
        pa_compat=["solid", "striped", "checkered"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "I have a corporate board meeting today and the dress code is strictly formal. What should I wear?",
            "Presenting to the executive board in a formal boardroom this afternoon. What should I wear?",
        ],
        q_implicit=[
            "Big meeting with the company's senior leadership this afternoon. Outfit advice?",
            "I'm sitting in with the directors at headquarters today. What should I wear?",
        ],
        justification="Hoodie/shorts and loud colors/prints at a corporate board meeting are universally inappropriate."),

    _sc("biz_investor_pitch", "business_professional",
        "Investor Pitch Presentation",
        {"time": {"time_of_day": "daytime"},
         "place": {"venue": "office", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "work_meeting", "formality_required": "business_casual"}},
        gc_compat=["blazer", "shirt", "suit_jacket", "blouse"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray", "white", "beige"],
        co_incompat=["orange", "yellow", "pink"],
        pa_compat=["solid", "striped", "checkered"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Pitching to venture capital investors today; formal business attire is expected. What should I wear?",
            "Presenting our funding pitch to investors in a few hours. What should I wear?",
        ],
        q_implicit=[
            "Big meeting with the VCs tomorrow morning to raise our round. Outfit advice?",
            "Trying to win over serious investors at a meeting tomorrow. What should I wear?",
        ],
        justification="Investor pitches demand credibility; casual wear and loud prints undermine professional trust."),

    _sc("biz_job_interview", "business_professional",
        "Formal Job Interview",
        {"time": {"time_of_day": "daytime"},
         "place": {"venue": "office", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "work_meeting", "formality_required": "business_casual"}},
        gc_compat=["blazer", "shirt", "blouse", "suit_jacket"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray", "white", "beige"],
        co_incompat=["orange", "yellow", "pink"],
        pa_compat=["solid", "striped", "checkered"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "I have an in-person job interview at a corporate office tomorrow. What should I wear?",
            "Interviewing for a professional role at their headquarters tomorrow. What should I wear?",
        ],
        q_implicit=[
            "Final-round interview with the hiring panel next week. Outfit advice?",
            "Meeting the team that decides if I get the job tomorrow. What should I wear?",
        ],
        justification="Hoodie/shorts and loud prints to a formal interview signal disrespect for the opportunity."),

    _sc("biz_client_meeting", "business_professional",
        "Client Meeting / Business Dinner",
        {"time": {"time_of_day": "evening"},
         "place": {"venue": "restaurant", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "work_meeting", "formality_required": "business_casual"}},
        gc_compat=["blazer", "shirt", "suit_jacket", "blouse"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray", "white", "beige"],
        co_incompat=["orange", "yellow", "pink"],
        pa_compat=["solid", "striped", "checkered"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Business dinner with an important client tonight; professional attire expected. What should I wear?",
            "Meeting a key client over dinner to represent the firm. What should I wear?",
        ],
        q_implicit=[
            "Taking our biggest client out to dinner tonight. Outfit advice?",
            "Dinner with the client whose account I manage. What should I wear?",
        ],
        justification="Business meetings represent the company; casual wear and loud prints are a professional failure."),

    # ════════ 10. ULTRA-FORMAL / CEREMONIAL (4) — dress-coded ════════
    _sc("formal_black_tie_gala", "ultra_formal",
        "Black-Tie Gala",
        {"time": {"time_of_day": "evening"},
         "place": {"indoor_outdoor": "indoor"},
         "occasion": {"activity": "social_gathering", "formality_required": "formal"}},
        gc_compat=["suit_jacket", "blazer", "dress"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray", "white"],
        co_incompat=["orange", "yellow", "pink"],  # red removed: red eveningwear appropriate under contemporary_western
        pa_compat=["solid", "striped"],
        pa_incompat=["camouflage", "argyle", "floral"],
        q_explicit=[
            "Attending a black-tie gala tonight with a strict formal dress code. What should I wear?",
            "Invited to a formal evening ball — black tie. What should I wear?",
        ],
        q_implicit=[
            "Got an invitation to a glamorous evening fundraiser at a grand hotel. Outfit advice?",
            "Big formal evening event downtown tonight. What should I wear?",
        ],
        justification="Black-tie events have an explicit dress code that excludes casual and loud clothing."),

    _sc("formal_opera_premiere", "ultra_formal",
        "Opera / Ballet Premiere",
        {"time": {"time_of_day": "evening"},
         "place": {"venue": "concert_hall", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "formal"}},
        gc_compat=["suit_jacket", "blazer", "dress", "shirt"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray", "white"],
        co_incompat=["orange", "yellow", "pink"],  # red removed: red eveningwear appropriate under contemporary_western
        pa_compat=["solid", "striped"],
        pa_incompat=["camouflage", "argyle", "floral"],
        q_explicit=[
            "Attending an opera premiere tonight; they enforce a formal dress code. What should I wear?",
            "Going to opening night at the opera house with a dress code. What should I wear?",
        ],
        q_implicit=[
            "Have tickets to the gala opening night at the opera. Outfit advice?",
            "Premiere performance at the grand concert hall tonight. What should I wear?",
        ],
        justification="Opera/ballet premieres with dress codes prohibit casual and loud attire."),

    _sc("formal_national_award", "ultra_formal",
        "National Award Ceremony (Recipient)",
        {"time": {"time_of_day": "evening"},
         "place": {"indoor_outdoor": "indoor"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "formal"}},
        gc_compat=["suit_jacket", "blazer", "dress"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray", "white"],
        co_incompat=["orange", "yellow", "pink"],  # red removed: red eveningwear appropriate under contemporary_western
        pa_compat=["solid", "striped"],
        pa_incompat=["camouflage", "argyle", "floral"],
        q_explicit=[
            "I'm receiving a national award tonight at a formal ceremony on stage. What should I wear?",
            "Being honored at a formal national award ceremony this evening. What should I wear?",
        ],
        q_implicit=[
            "I'm being recognized on stage at a prestigious ceremony tonight. Outfit advice?",
            "Accepting a major honor at the national hall this evening. What should I wear?",
        ],
        justification="Receiving a national award in a hoodie/shorts or loud prints disrespects the institution."),

    _sc("formal_state_banquet", "ultra_formal",
        "Head-of-State Banquet",
        {"time": {"time_of_day": "evening"},
         "place": {"indoor_outdoor": "indoor"},
         "occasion": {"activity": "social_gathering", "formality_required": "formal"}},
        gc_compat=["suit_jacket", "blazer", "dress"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray", "white"],
        co_incompat=["orange", "yellow", "pink"],  # red removed: red eveningwear appropriate under contemporary_western
        pa_compat=["solid", "striped"],
        pa_incompat=["camouflage", "argyle", "floral"],
        q_explicit=[
            "Invited to a state banquet hosted by a head of state, strict formal protocol. What should I wear?",
            "Attending a formal banquet at the presidential residence. What should I wear?",
        ],
        q_implicit=[
            "Got a formal invitation to dine at the presidential palace. Outfit advice?",
            "Attending an official banquet with dignitaries this evening. What should I wear?",
        ],
        justification="State banquets enforce strict formal protocol; casual or loud attire is unacceptable."),

    # ════════ 11. JUDICIAL / CIVIC / OFFICIAL (3) — dress-coded ════════
    _sc("civic_court_appearance", "judicial_civic",
        "Court Appearance",
        {"time": {"time_of_day": "daytime"},
         "place": {"indoor_outdoor": "indoor"},
         "occasion": {"formality_required": "formal"}},
        gc_compat=["suit_jacket", "blazer", "shirt", "blouse"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray"],
        co_incompat=["orange", "yellow", "pink", "red"],
        pa_compat=["solid", "striped"],
        pa_incompat=["camouflage", "argyle", "floral"],
        q_explicit=[
            "I have to appear in court tomorrow before a judge. What should I wear?",
            "Appearing in a courtroom for a formal hearing tomorrow. What should I wear?",
        ],
        q_implicit=[
            "I've been called before a judge next week. Outfit advice?",
            "Have to show up at the courthouse for my hearing. What should I wear?",
        ],
        justification="Court appearances require conservative formal dress; casual/loud wear may constitute contempt."),

    _sc("civic_supreme_argument", "judicial_civic",
        "High-Court Oral Argument",
        {"time": {"time_of_day": "daytime"},
         "place": {"indoor_outdoor": "indoor"},
         "occasion": {"formality_required": "formal"}},
        gc_compat=["suit_jacket", "blazer", "shirt", "blouse"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray"],
        co_incompat=["orange", "yellow", "pink", "red"],
        pa_compat=["solid", "striped"],
        pa_incompat=["camouflage", "argyle", "floral"],
        q_explicit=[
            "Presenting an oral argument before the high court. What should I wear?",
            "Arguing a case at the highest court next month. What should I wear?",
        ],
        q_implicit=[
            "I'm the attorney appearing at the top court next month. Outfit advice?",
            "Standing before the senior bench to argue a case soon. What should I wear?",
        ],
        justification="High-court proceedings demand the most conservative formal attire."),

    _sc("civic_govt_hearing", "judicial_civic",
        "Government / Parliamentary Hearing",
        {"time": {"time_of_day": "daytime"},
         "place": {"indoor_outdoor": "indoor"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "formal"}},
        gc_compat=["suit_jacket", "blazer", "shirt", "blouse"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray"],
        co_incompat=["orange", "yellow", "pink", "red"],
        pa_compat=["solid", "striped"],
        pa_incompat=["camouflage", "argyle", "floral"],
        q_explicit=[
            "Testifying at a formal government hearing on the record. What should I wear?",
            "Appearing before a parliamentary committee to give testimony. What should I wear?",
        ],
        q_implicit=[
            "I've been summoned to speak before a legislative committee. Outfit advice?",
            "Giving official testimony at a public hearing next week. What should I wear?",
        ],
        justification="Official government hearings follow conservative formal norms; casual/loud attire is improper."),

    # ════════ 12. MOURNING / SOMBER (4) — dress-coded ════════
    _sc("mourn_funeral", "mourning_somber",
        "Funeral Service",
        {"place": {"indoor_outdoor": "mixed"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "formal"}},
        gc_compat=["suit_jacket", "blazer", "coat", "shirt", "blouse", "dress"],
        gc_incompat=["shorts", "tank_top", "hoodie", "t_shirt"],
        co_compat=["black", "navy", "gray"],
        co_incompat=["orange", "yellow", "pink", "red", "green"],
        pa_compat=["solid"],
        pa_incompat=["camouflage", "argyle", "floral", "polka_dot"],
        q_explicit=[
            "I'm attending a funeral and need to dress respectfully and somberly. What should I wear?",
            "Going to a funeral service this week; dark, formal dress expected. What should I wear?",
        ],
        q_implicit=[
            "I have to say goodbye to a relative at the service this weekend. Outfit advice?",
            "Attending the burial of a close family friend. What should I wear?",
        ],
        justification="Funerals universally demand dark, somber attire; bright/casual/loud clothing is disrespectful."),

    _sc("mourn_memorial", "mourning_somber",
        "Memorial / Remembrance Service",
        {"place": {"indoor_outdoor": "mixed"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "formal"}},
        gc_compat=["suit_jacket", "blazer", "coat", "shirt", "dress"],
        gc_incompat=["shorts", "tank_top", "hoodie", "t_shirt"],
        co_compat=["black", "navy", "gray"],
        co_incompat=["orange", "yellow", "pink", "red", "green"],
        pa_compat=["solid"],
        pa_incompat=["camouflage", "argyle", "floral", "polka_dot"],
        q_explicit=[
            "Attending a memorial remembrance service for a colleague. What should I wear?",
            "Going to a solemn remembrance ceremony this week. What should I wear?",
        ],
        q_implicit=[
            "There's a service to honor someone who passed, and I'm attending. Outfit advice?",
            "Going to a remembrance gathering for a late mentor. What should I wear?",
        ],
        justification="Memorial services share funeral solemnity; bright/festive attire is disrespectful."),

    _sc("mourn_vigil", "mourning_somber",
        "Candlelight Vigil",
        {"time": {"time_of_day": "night"},
         "place": {"indoor_outdoor": "outdoor"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "smart_casual"}},
        gc_compat=["coat", "jacket", "shirt", "sweater", "blazer", "blouse"],
        gc_incompat=["shorts", "tank_top", "hoodie", "t_shirt"],
        co_compat=["black", "navy", "gray"],
        co_incompat=["orange", "yellow", "pink", "red", "green"],
        pa_compat=["solid"],
        pa_incompat=["camouflage", "argyle", "floral", "polka_dot"],
        q_explicit=[
            "Attending a candlelight vigil tonight to mourn and pay respects. What should I wear?",
            "Going to a solemn evening vigil; subdued dress expected. What should I wear?",
        ],
        q_implicit=[
            "There's a community gathering tonight to grieve a tragedy. Outfit advice?",
            "Joining a quiet evening tribute with candles tonight. What should I wear?",
        ],
        justification="Candlelight vigils are solemn; bright/festive colors and loud prints are inappropriate."),

    _sc("mourn_condolence_visit", "mourning_somber",
        "Condolence Visit to Bereaved Family",
        {"place": {"indoor_outdoor": "indoor"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "smart_casual"}},
        gc_compat=["shirt", "blouse", "blazer", "sweater", "coat", "dress"],
        gc_incompat=["shorts", "tank_top", "hoodie", "t_shirt"],
        co_compat=["black", "navy", "gray"],
        co_incompat=["orange", "yellow", "pink", "red", "green"],
        pa_compat=["solid"],
        pa_incompat=["camouflage", "argyle", "floral", "polka_dot"],
        q_explicit=[
            "Visiting a grieving family to offer my condolences in person. What should I wear?",
            "Paying respects at a bereaved family's home; subdued dress. What should I wear?",
        ],
        q_implicit=[
            "Going to sit with a friend whose parent just passed. Outfit advice?",
            "Visiting a household in mourning to offer support. What should I wear?",
        ],
        justification="Visiting a grieving family in bright party colors or loud prints is universally insensitive."),

    # ════════ 13. RELIGIOUS / MODEST (3) — dress-coded ════════
    _sc("relig_conservative_worship", "religious_modest",
        "Conservative Religious Worship Service",
        {"place": {"indoor_outdoor": "indoor"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "business_casual"}},
        gc_compat=["shirt", "blouse", "blazer", "dress", "sweater", "coat"],
        gc_incompat=["tank_top", "shorts", "t_shirt", "hoodie"],
        # color: dropped under CULTURAL_FRAME — modesty is coverage (garment);
        # no crisp color rule, so color is a preference-only axis here
        pa_compat=["solid", "striped"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Attending a conservative religious service where modest dress is required. What should I wear?",
            "Going to worship with a traditional congregation; modesty expected. What should I wear?",
        ],
        q_implicit=[
            "Joining a friend's family for their weekly service at a traditional congregation. Outfit advice?",
            "Attending a religious service known for its strict modesty norms. What should I wear?",
        ],
        justification="Conservative worship has modesty norms; tank tops/shorts and loud prints are widely prohibited."),

    _sc("relig_temple_ceremony", "religious_modest",
        "Temple Ceremony / Sacred Site Visit",
        {"place": {"venue": "temple", "indoor_outdoor": "mixed"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "business_casual"}},
        gc_compat=["shirt", "blouse", "blazer", "dress", "sweater", "coat"],
        gc_incompat=["tank_top", "shorts", "t_shirt", "hoodie"],
        # color: dropped under CULTURAL_FRAME — modesty is coverage (garment);
        # no crisp color rule, so color is a preference-only axis here
        pa_compat=["solid", "striped"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Visiting a sacred temple where shoulders and knees must be covered. What should I wear?",
            "Attending a temple ceremony with strict modesty rules. What should I wear?",
        ],
        q_implicit=[
            "Touring some historic religious sites that enforce a covered dress code. Outfit advice?",
            "Invited to a ceremony at a temple this weekend. What should I wear?",
        ],
        justification="Sacred sites require covered, modest attire; exposed/loud clothing is prohibited."),

    _sc("relig_solemn_observance", "religious_modest",
        "Solemn Religious Observance",
        {"place": {"indoor_outdoor": "indoor"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "business_casual"}},
        gc_compat=["shirt", "blouse", "blazer", "dress", "sweater", "coat"],
        gc_incompat=["tank_top", "shorts", "t_shirt", "hoodie"],
        # color: dropped under CULTURAL_FRAME — modesty is coverage (garment);
        # no crisp color rule, so color is a preference-only axis here
        pa_compat=["solid", "striped"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Attending a solemn religious observance where modest, subdued dress is expected. What should I wear?",
            "Going to an important religious holy-day service with modesty norms. What should I wear?",
        ],
        q_implicit=[
            "Joining a major holy-day gathering at the congregation. Outfit advice?",
            "Attending a significant religious observance with my family. What should I wear?",
        ],
        justification="Solemn observances require modest, subdued attire; exposed/loud clothing is improper."),

    # ════════ 14. SEMI-FORMAL SOCIAL (4) — dress-coded ════════
    _sc("social_gallery_opening", "semi_formal_social",
        "Art Gallery Opening / Vernissage",
        {"time": {"time_of_day": "evening"},
         "place": {"venue": "art_gallery", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "social_gathering", "formality_required": "smart_casual"}},
        gc_compat=["blazer", "shirt", "blouse", "dress"],
        gc_incompat=["hoodie", "shorts", "tank_top"],
        pa_compat=["solid", "striped", "checkered", "floral"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Attending an art gallery opening tonight; smart-casual minimum. What should I wear?",
            "Going to a vernissage at a downtown gallery this evening. What should I wear?",
        ],
        q_implicit=[
            "Got invited to the launch of a new exhibition tonight. Outfit advice?",
            "Heading to an opening reception at the art gallery. What should I wear?",
        ],
        justification="Gallery openings are smart-casual; shorts/hoodie and loud prints signal disinterest."),

    _sc("social_alumni_dinner", "semi_formal_social",
        "University Alumni Formal Dinner",
        {"time": {"time_of_day": "evening"},
         "place": {"venue": "restaurant", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "social_gathering", "formality_required": "business_casual"}},
        gc_compat=["blazer", "shirt", "blouse", "dress"],
        gc_incompat=["hoodie", "shorts", "tank_top"],
        pa_compat=["solid", "striped", "checkered", "floral"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Alumni formal dinner tonight in a hotel ballroom; smart attire expected. What should I wear?",
            "Going to a formal alumni reunion dinner this evening. What should I wear?",
        ],
        q_implicit=[
            "My old university is hosting a reunion dinner and I'm going. Outfit advice?",
            "Catching up with classmates at a fancy reunion dinner. What should I wear?",
        ],
        justification="Alumni formal dinners have dress codes; hoodies/shorts and loud prints are out of place."),

    _sc("social_company_gala", "semi_formal_social",
        "Company Annual Gala / Holiday Party",
        {"time": {"time_of_day": "evening"},
         "place": {"indoor_outdoor": "indoor"},
         "occasion": {"activity": "social_gathering", "formality_required": "business_casual"}},
        gc_compat=["blazer", "shirt", "blouse", "dress", "suit_jacket"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["black", "navy", "gray", "white", "beige"],
        co_incompat=["orange", "yellow"],
        pa_compat=["solid", "striped", "checkered"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Company annual gala tonight; smart-casual to semi-formal expected. What should I wear?",
            "Going to the office year-end gala this evening. What should I wear?",
        ],
        q_implicit=[
            "Our company's big end-of-year party is tonight. Outfit advice?",
            "Heading to the firm's annual celebration dinner. What should I wear?",
        ],
        justification="Company galas have implicit dress codes; casual wear and loud prints harm professional standing."),

    _sc("social_first_date_upscale", "semi_formal_social",
        "First Date at an Upscale Restaurant",
        {"time": {"time_of_day": "evening"},
         "place": {"venue": "restaurant", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "first_date", "formality_required": "smart_casual"}},
        gc_compat=["shirt", "blouse", "blazer", "dress"],
        gc_incompat=["hoodie", "shorts", "tank_top"],
        pa_compat=["solid", "striped", "floral", "checkered"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "First date at an upscale restaurant tonight; I want to look put-together. What should I wear?",
            "Dinner date at a nice, dressy restaurant this evening. What should I wear?",
        ],
        q_implicit=[
            "Taking someone I like to a really nice restaurant tonight. Outfit advice?",
            "Big first date at a fancy spot downtown tonight. What should I wear?",
        ],
        justification="An upscale restaurant date in a hoodie/shorts or loud prints signals disrespect."),

    # ════════ 15. WEDDING / CELEBRATION (3) — dress-coded ════════
    _sc("wedding_reception", "wedding_celebration",
        "Wedding Reception (Guest)",
        {"place": {"venue": "wedding_venue", "indoor_outdoor": "indoor"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "smart_casual"}},
        gc_compat=["blazer", "suit_jacket", "dress", "shirt", "blouse"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["navy", "beige", "blue", "gray", "green", "purple", "black"],
        co_incompat=["orange", "yellow", "white"],  # white added: guest-wears-white taboo under contemporary_western
        pa_compat=["solid", "striped", "floral", "checkered"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Attending a wedding reception in a hotel ballroom; smart attire expected. What should I wear?",
            "Going to a friend's wedding reception this weekend. What should I wear?",
        ],
        q_implicit=[
            "My friend is getting married and I'm going to the reception next Saturday. Outfit advice?",
            "Invited to celebrate a couple at their reception dinner. What should I wear?",
        ],
        justification="Wearing shorts/hoodie or loud prints to a wedding reception insults the couple."),

    _sc("wedding_garden_ceremony", "wedding_celebration",
        "Garden Wedding Ceremony (Guest)",
        {"time": {"season": "spring", "weather": "sunny"},
         "place": {"venue": "wedding_venue", "indoor_outdoor": "outdoor"},
         "occasion": {"activity": "ceremony_attendance", "formality_required": "smart_casual"}},
        gc_compat=["blazer", "suit_jacket", "dress", "shirt", "blouse"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["navy", "beige", "blue", "gray", "green", "purple", "black"],
        co_incompat=["orange", "yellow", "white"],  # white added: guest-wears-white taboo under contemporary_western
        pa_compat=["solid", "striped", "floral", "checkered"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Attending an outdoor garden wedding ceremony as a guest. What should I wear?",
            "Going to a daytime garden wedding; smart attire expected. What should I wear?",
        ],
        q_implicit=[
            "A couple I know is marrying in a garden this spring and I'm invited. Outfit advice?",
            "Heading to a lovely outdoor wedding at a botanical venue. What should I wear?",
        ],
        justification="A garden wedding still expects smart guest attire; shorts/hoodie and loud prints are out of place."),

    _sc("wedding_anniversary_party", "wedding_celebration",
        "Milestone Anniversary / Engagement Celebration",
        {"time": {"time_of_day": "evening"},
         "place": {"indoor_outdoor": "indoor"},
         "occasion": {"activity": "social_gathering", "formality_required": "smart_casual"}},
        gc_compat=["blazer", "suit_jacket", "dress", "shirt", "blouse"],
        gc_incompat=["hoodie", "shorts", "tank_top", "t_shirt"],
        co_compat=["navy", "beige", "blue", "gray", "green", "purple", "black"],
        co_incompat=["orange", "yellow", "white"],  # white added: guest-wears-white taboo under contemporary_western
        pa_compat=["solid", "striped", "floral", "checkered"],
        pa_incompat=["camouflage", "argyle"],
        q_explicit=[
            "Attending a milestone anniversary celebration dinner; smart attire expected. What should I wear?",
            "Going to an engagement celebration party this evening. What should I wear?",
        ],
        q_implicit=[
            "My parents' big anniversary party is this weekend and I'm attending. Outfit advice?",
            "Celebrating a couple's engagement at a nice venue tonight. What should I wear?",
        ],
        justification="Milestone celebrations expect smart attire; shorts/hoodie and loud prints are out of place."),
]


# ═══════════════════════════════════════════════════════════
#  HELPERS  (unchanged API)
# ═══════════════════════════════════════════════════════════

def get_constrained_axes(scenario):
    """Which axes does this scenario *explicitly* constrain (non-empty incompatible)?

    NOTE: under the relaxed compatibility check, color/pattern can be ACTIVE axes
    even when this returns only ['garment_category'] for a physical scenario.
    This helper is kept for reporting / count_axis_slots only.
    """
    axes = []
    for ax in ("garment_category", "color", "pattern"):
        c = scenario.get(ax)
        if c is not None and c.get("incompatible"):
            axes.append(ax)
    return axes


def get_scenario_by_id(sid):
    for sc in CANONICAL_SCENARIOS:
        if sc["scenario_id"] == sid:
            return sc
    return None


def scenarios_by_archetype():
    out = {}
    for sc in CANONICAL_SCENARIOS:
        out.setdefault(sc["archetype"], []).append(sc)
    return out


def count_axis_slots():
    total = 0
    by_axis = {"garment_category": 0, "color": 0, "pattern": 0}
    for sc in CANONICAL_SCENARIOS:
        for ax in get_constrained_axes(sc):
            by_axis[ax] += 1
            total += 1
    return total, by_axis


if __name__ == "__main__":
    from collections import Counter
    print(f"Scenarios: {len(CANONICAL_SCENARIOS)}")
    print(f"Archetypes: {len(SCENARIO_ARCHETYPES)}")
    by_arch = Counter(s["archetype"] for s in CANONICAL_SCENARIOS)
    for arch in SCENARIO_ARCHETYPES:
        print(f"  {arch:24s}: {by_arch.get(arch, 0)} scenarios")
    total, by_axis = count_axis_slots()
    print(f"Explicitly-constrained axis slots/user: {total} {by_axis}")