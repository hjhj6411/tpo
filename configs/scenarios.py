"""
POD-Bench v2 — Canonical Extreme Scenario Catalog
v4_preserve_53_scenarios

This file strictly preserves the previous 53 scenario IDs, names, TPO fields,
query seeds, and justifications. Only attribute labels inside the scenario
constraints are cleaned to match the revised garment/pattern vocabulary:
- shirt/blouse -> formal_shirt
- suit_jacket -> blazer
- jacket/coat/skirt -> context-appropriate canonical garments
- camouflage/argyle/plaid removed from pattern labels

v5 (WACV revision) — constraint-set fixes; scenario IDs/seeds unchanged:
- mourning: pattern compatible ['solid'] -> ['solid', 'striped'] (a single
  compatible value can never yield both a liked A-value and a non-preferred
  B-value; a subtle pinstripe is acceptable somber attire).
- mourn_vigil: 'hoodie' removed from incompatible garments (hoodies at a
  candlelight vigil are not universally inappropriate; formality is
  smart_casual).
- casual_leisure: incompatible garments extended with 'formal_shirt'
  (all 4) and 'slacks' (amusement park / zoo) so profiles that dislike
  blazer+trench_coat still have a preference-neutral TPO-violating garment.
- extreme_heat / aquatic_water: 'puffer_jacket' and 'leather_jacket' added to
  incompatible garments (same physical rationale as fleece/trench_coat).
- water_swim_session renamed to 'Pool Visit & Poolside Lounging' to match its
  query seeds (coverups/lounging, not lap swimming — swimwear is out of vocab).

v6 (WACV revision) — 7 scenarios added on top of the preserved 53 (total 60):
- New physical archetype `practical_work` (3): work_moving_day,
  work_home_renovation, work_garden_volunteering — durable/washable clothing
  required; formal or delicate garments are clearly TPO-violating.
- civic_citizenship_oath (judicial_civic), relig_baptism_guest
  (religious_modest), celebration_graduation (wedding_celebration) — bring
  every dress-coded archetype to comparable size and cover common ceremony
  types absent from v5.
- gym_climbing_bouldering (athletic_indoor) — full-mobility indoor sport,
  same constraint set as the other gym scenarios.

v7 (WACV revision) — wedding_celebration constraints tightened:
- color: compatible ['black', 'navy', 'gray']; incompatible ['white'].
- pattern: compatible ['solid', 'striped']; incompatible ['leopard', 'floral'].

v8 — hemisphere-ambiguity fix in query seeds:
- cold_blizzard_outdoor implicit seed 'outdoor holiday market in January'
  -> 'in the middle of winter' (January implies summer in the Southern
  Hemisphere; season words travel with the asker's hemisphere, month names
  do not). Audited all 240 seeds: the only other month reference,
  'Dubai in August' (heat_desert_trek), is location-anchored and kept.

v9 — explicit/implicit sharpening (full 240-seed audit):
- Implicit seeds must NOT state the constraint. 8 leaky implicit seeds
  rewritten to name only the event: relig_temple_ceremony ('enforce a
  covered dress code'), relig_conservative_worship ('strict modesty
  norms'), heat_summer_festival x2 ('peak heat' / 'midsummer heat'),
  formal_black_tie_gala ('formal evening event'), formal_opera_premiere
  ('formal opening-night crowd'), biz_investor_pitch ('Formal pitch
  meeting'), casual_farmers_market ('Casual morning').
- Implicit seeds must still LICENSE the inference. Season cues added to 2
  under-determined seeds: heat_beach_day ('by the shore' -> '+ at the
  height of summer'), heat_desert_trek ('through the dunes' ->
  'midsummer guided tour').
- Explicit seeds in dress-coded scenarios must state the dress
  expectation, since the occasion word alone (wedding, court, interview)
  also appears in implicit seeds. 19 weak explicit seeds strengthened
  with expectation phrases ('... attire expected' variants) across
  biz_*, civic_*, mourn_memorial, relig_baptism_guest, social_*,
  wedding_*, celebration_graduation.
- severe_weather implicit seeds necessarily mention the weather event
  (the weather IS the T/P context; without it the query is unanswerable).
  Kept as-is by design; treat this archetype's implicit split as
  weak-implicit in explicit-vs-implicit analyses.
"""

CULTURAL_FRAME = 'contemporary_western'
EVAL_FRAME_CLAUSE = 'Assume contemporary Western / international dress-code conventions when judging what is appropriate for the occasion.'
FRAME_SCOPED_ARCHETYPES = {'business_professional',
 'judicial_civic',
 'mourning_somber',
 'religious_modest',
 'semi_formal_social',
 'ultra_formal',
 'wedding_celebration',
 'club_code'}

SCENARIO_ARCHETYPES = {'extreme_cold': 'Extreme Cold / Winter',
 'extreme_heat': 'Extreme Heat / Summer',
 'aquatic_water': 'Water & Swimming',
 'athletic_indoor': 'Indoor Athletic / Gym',
 'athletic_outdoor': 'Outdoor Athletic / Field Sports',
 'rugged_outdoor': 'Rugged Outdoor / Hiking & Camping',
 'severe_weather': 'Severe Weather Events',
 'casual_leisure': 'Casual Leisure / Everyday Outing',
 'practical_work': 'Manual / Practical Work',
 'business_professional': 'Business / Professional',
 'ultra_formal': 'Ultra-Formal / Ceremonial',
 'judicial_civic': 'Judicial / Civic / Official',
 'mourning_somber': 'Mourning / Somber',
 'religious_modest': 'Religious / Sacred / Modest',
 'semi_formal_social': 'Semi-Formal Social',
 'wedding_celebration': 'Wedding / Celebration',
 'safety_visibility': 'Night Visibility / Road Safety',
 'field_stealth': 'Field Stealth / Wildlife & Camouflage',
 'stage_media': 'Stage & Media Production',
 'club_code': 'Club / Institutional Athletic Code'}

CANONICAL_SCENARIOS = [{'scenario_id': 'cold_blizzard_outdoor',
  'archetype': 'extreme_cold',
  'name': 'Blizzard & Sub-zero Outdoor All-day',
  'tpo': {'time': {'season': 'winter', 'weather': 'snowy'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'travel', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['fleece_jacket', 'puffer_jacket', 'sweater', 'hoodie'],
                       'incompatible': ['shorts', 'tank_top', 't_shirt', 'dress', 'mini_skirt']},
  'justification': 'Sub-zero blizzard makes light/exposed clothing a hypothermia risk.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ["It's going to be a blizzard with sub-zero temperatures and I'll be "
                               'outside all day. What should I wear?',
                               "Heavy snow and freezing winds are forecast and I'll be outdoors "
                               'for hours. What outfit makes sense?'],
                  'implicit': ["I'm heading to a winter festival up in the mountains this weekend. "
                               'Any outfit advice?',
                               'Spending the whole day at an outdoor holiday market in the middle '
                               'of winter. What should I throw on?']}},
 {'scenario_id': 'cold_polar_expedition',
  'archetype': 'extreme_cold',
  'name': 'Polar / Arctic Expedition Day',
  'tpo': {'time': {'season': 'winter', 'weather': 'snowy'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'travel', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['fleece_jacket', 'puffer_jacket', 'sweater', 'hoodie'],
                       'incompatible': ['shorts', 'tank_top', 't_shirt', 'dress', 'mini_skirt']},
  'justification': 'Polar exposure without heavy insulation causes frostbite/hypothermia.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ["I'm joining an arctic expedition and will be outdoors in extreme "
                               'cold all day. What should I wear?',
                               'Going out onto the polar ice in freezing wind for the whole day. '
                               "What's the right outfit?"],
                  'implicit': ['Booked a trip to a polar research base for some fieldwork. What do '
                               'I pack to wear?',
                               'Heading way up north for a glacier trek next week. Outfit '
                               'thoughts?']}},
 {'scenario_id': 'cold_ski_slope',
  'archetype': 'extreme_cold',
  'name': 'All-day Skiing on the Slopes',
  'tpo': {'time': {'season': 'winter', 'weather': 'snowy'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'exercise', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['fleece_jacket', 'puffer_jacket', 'sweater', 'hoodie'],
                       'incompatible': ['shorts', 'tank_top', 't_shirt', 'dress', 'mini_skirt']},
  'justification': 'Skiing in alpine cold rules out exposed-skin clothing.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ["I'll be skiing on the slopes all day in freezing alpine weather. "
                               'What should I wear?',
                               'Spending the day on a snowy mountain skiing in sub-zero temps. '
                               'Outfit advice?'],
                  'implicit': ["Got a lift pass for the resort tomorrow and I'll be on the "
                               'mountain till dark. What should I wear?',
                               'First time hitting the slopes this season. What outfit makes '
                               'sense?']}},
 {'scenario_id': 'cold_ice_fishing',
  'archetype': 'extreme_cold',
  'name': 'Frozen-Lake Ice Fishing',
  'tpo': {'time': {'season': 'winter', 'weather': 'cold'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'casual_outing', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['fleece_jacket', 'puffer_jacket', 'sweater', 'hoodie'],
                       'incompatible': ['shorts', 'tank_top', 't_shirt', 'dress', 'mini_skirt']},
  'justification': 'Hours motionless on ice in freezing air requires maximum insulation.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Sitting out on a frozen lake ice fishing for hours in sub-zero '
                               'cold. What should I wear?',
                               "I'll be still on the ice in freezing wind all morning. What's the "
                               'warmest sensible outfit?'],
                  'implicit': ['Going ice fishing with my uncle this weekend. What should I wear?',
                               'Planning a quiet day out on the frozen lake. Outfit advice?']}},
 {'scenario_id': 'heat_beach_day',
  'archetype': 'extreme_heat',
  'name': 'Scorching Beach Day',
  'tpo': {'time': {'season': 'summer', 'weather': 'hot_humid'},
          'place': {'venue': 'beach', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'casual_outing', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'tank_top', 'shorts'],
                       'incompatible': ['fleece_jacket', 'puffer_jacket', 'leather_jacket', 'trench_coat',
                                        'sweater', 'blazer']},
  'justification': 'Heavy outerwear at a scorching beach causes heatstroke risk.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ["It's 38°C and I'm heading to the beach for the whole day. What "
                               'should I wear?',
                               "Brutal heat today and I'll be on the sand for hours. What outfit "
                               'makes sense?'],
                  'implicit': ['Planning a beach day this weekend at the peak of summer. Outfit?',
                               'Meeting friends down by the shore tomorrow afternoon at the '
                               'height of summer. What should I wear?']}},
 {'scenario_id': 'heat_desert_trek',
  'archetype': 'extreme_heat',
  'name': 'Desert Trek (Peak Summer)',
  'tpo': {'time': {'season': 'summer', 'weather': 'hot_humid'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'travel', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'tank_top', 'shorts'],
                       'incompatible': ['fleece_jacket', 'puffer_jacket', 'leather_jacket', 'trench_coat',
                                        'sweater', 'blazer']},
  'justification': '42°C desert heat in a fleece/coat is a heatstroke emergency.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Trekking across a desert in 42°C heat. What should I wear?',
                               "Out in the open desert under blazing sun all day. What's the right "
                               'outfit?'],
                  'implicit': ['Spending the day exploring Dubai in August. Outfit advice?',
                               'Doing a midsummer guided tour through the dunes next week. What '
                               'should I wear?']}},
 {'scenario_id': 'heat_tropical_resort',
  'archetype': 'extreme_heat',
  'name': 'Tropical Resort Pool Party',
  'tpo': {'time': {'season': 'summer', 'weather': 'hot_humid'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'social_gathering', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'tank_top', 'shorts'],
                       'incompatible': ['fleece_jacket', 'puffer_jacket', 'leather_jacket', 'trench_coat',
                                        'sweater', 'blazer']},
  'justification': 'Fleece/suit jacket at a tropical pool party is physically unbearable.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Pool party at a tropical resort in sweltering heat. What should I '
                               'wear?',
                               "It's humid and boiling and I'm headed to a poolside party. "
                               'Outfit?'],
                  'implicit': ['Friends are throwing a poolside get-together at the resort. What '
                               'should I wear?',
                               'Spending the afternoon lounging by the resort pool. Outfit '
                               'advice?']}},
 {'scenario_id': 'heat_summer_festival',
  'archetype': 'extreme_heat',
  'name': 'Midsummer Outdoor Music Festival',
  'tpo': {'time': {'season': 'summer', 'weather': 'hot_humid'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'social_gathering', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'tank_top', 'shorts'],
                       'incompatible': ['fleece_jacket', 'puffer_jacket', 'leather_jacket', 'trench_coat',
                                        'sweater', 'blazer']},
  'justification': 'All-day outdoor festival in extreme heat makes heavy outerwear a health '
                   'hazard.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Going to an all-day outdoor music festival in 35°C heat. What '
                               'should I wear?',
                               'Standing in a festival crowd under the sun from noon to night. '
                               'Outfit advice?'],
                  'implicit': ['Got tickets to an all-day outdoor summer festival this weekend. '
                               'What should I wear?',
                               'Spending Saturday at an open-air concert downtown in midsummer. '
                               'Outfit?']}},
 {'scenario_id': 'water_swim_session',
  'archetype': 'aquatic_water',
  'name': 'Pool Visit & Poolside Lounging',
  'tpo': {'place': {'venue': 'pool', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'exercise', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'tank_top', 'shorts'],
                       'incompatible': ['trench_coat', 'blazer', 'fleece_jacket', 'puffer_jacket',
                                        'leather_jacket', 'sweater', 'dress']},
  'justification': 'A fleece/blazer/dress at the pool is absurd and waterlogged-dangerous.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Heading to the pool for light water activities and lounging. What '
                               'should I wear?',
                               'Going to the pool for casual water exercise and time by the deck. '
                               "What's the right outfit?"],
                  'implicit': ['Starting regular pool visits this week. What should I bring to '
                               'wear over swimwear?',
                               'Joined the local pool for morning water workouts. Outfit '
                               'advice?']}},
 {'scenario_id': 'water_park_day',
  'archetype': 'aquatic_water',
  'name': 'Water Park Visit (Peak Summer)',
  'tpo': {'time': {'season': 'summer', 'weather': 'hot_humid'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'casual_outing', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'tank_top', 'shorts'],
                       'incompatible': ['trench_coat', 'blazer', 'fleece_jacket', 'puffer_jacket',
                                        'leather_jacket', 'sweater', 'dress']},
  'justification': 'Coat/fleece/dress at a water park in summer heat is nonsensical and unsafe.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Taking the family to a water park in scorching heat. What should I '
                               'wear?',
                               'Spending the day going down water slides in the sun. Outfit?'],
                  'implicit': ['Day out at the water park this weekend. What should I wear?',
                               'Kids want to hit the aqua park tomorrow. Outfit advice?']}},
 {'scenario_id': 'water_poolside_lounge',
  'archetype': 'aquatic_water',
  'name': 'Resort Poolside Lounging',
  'tpo': {'time': {'season': 'summer', 'weather': 'hot_humid'},
          'place': {'venue': 'pool', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'casual_outing', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'tank_top', 'shorts'],
                       'incompatible': ['trench_coat', 'blazer', 'fleece_jacket', 'puffer_jacket',
                                        'leather_jacket', 'sweater', 'dress']},
  'justification': 'Heavy structured garments by a pool in summer heat are impractical and get '
                   'soaked.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Lounging poolside all afternoon in the heat, going in and out of '
                               'the water. What should I wear?',
                               'Spending a hot day by the pool with frequent dips. Outfit?'],
                  'implicit': ['Relaxing by the hotel pool tomorrow afternoon. What should I wear?',
                               'Booked a cabana by the pool for the day. Outfit advice?']}},
 {'scenario_id': 'gym_weight_training',
  'archetype': 'athletic_indoor',
  'name': 'Gym Weight Training',
  'tpo': {'place': {'venue': 'gym', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'exercise', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt',
                                      'tank_top',
                                      'shorts',
                                      'hoodie',
                                      'sweater',
                                      'leggings'],
                       'incompatible': ['blazer', 'trench_coat', 'dress', 'fleece_jacket', 'formal_shirt']},
  'justification': 'Weight training in a suit/coat/dress restricts movement and damages garments.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Heading to the gym for a heavy weight-training session. What '
                               'should I wear?',
                               "Doing barbell and machine work at the gym today. What's the right "
                               'outfit?'],
                  'implicit': ['Got a lifting session after work today. What should I wear?',
                               'Starting a strength program at the gym this week. Outfit '
                               'advice?']}},
 {'scenario_id': 'gym_yoga_class',
  'archetype': 'athletic_indoor',
  'name': 'Yoga / Pilates Class',
  'tpo': {'place': {'venue': 'gym', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'exercise', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt',
                                      'tank_top',
                                      'shorts',
                                      'hoodie',
                                      'sweater',
                                      'leggings'],
                       'incompatible': ['blazer', 'trench_coat', 'dress', 'fleece_jacket', 'formal_shirt']},
  'justification': 'Yoga requires full-body flexibility; blazers/coats/dresses physically prevent '
                   'poses.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['I have a yoga class tonight with lots of stretching and floor '
                               'poses. What should I wear?',
                               'Going to a Pilates session focused on full-body mobility. Outfit?'],
                  'implicit': ["Starting yoga this week. What's appropriate to wear?",
                               'Signed up for a mat-based fitness class. What should I wear?']}},
 {'scenario_id': 'gym_indoor_basketball',
  'archetype': 'athletic_indoor',
  'name': 'Indoor Basketball / Futsal',
  'tpo': {'place': {'venue': 'gym', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'exercise', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt',
                                      'tank_top',
                                      'shorts',
                                      'hoodie',
                                      'sweater',
                                      'leggings'],
                       'incompatible': ['blazer', 'trench_coat', 'dress', 'fleece_jacket', 'formal_shirt']},
  'justification': 'Court sport in a blazer/coat/dress restricts movement and risks injury.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Playing indoor basketball tonight with lots of running and '
                               'jumping. What should I wear?',
                               'Got a futsal match on an indoor court this evening. Outfit?'],
                  'implicit': ["Joining a pick-up league at the indoor court. What's right to "
                               'wear?',
                               'Friends booked the gym court for a game tonight. Outfit advice?']}},
 {'scenario_id': 'gym_climbing_bouldering',
  'archetype': 'athletic_indoor',
  'name': 'Indoor Climbing / Bouldering Gym',
  'tpo': {'place': {'venue': 'gym', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'exercise', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt',
                                      'tank_top',
                                      'shorts',
                                      'hoodie',
                                      'sweater',
                                      'leggings'],
                       'incompatible': ['blazer', 'trench_coat', 'dress', 'fleece_jacket', 'formal_shirt']},
  'justification': 'Climbing demands full range of motion and abrades fabric; blazers/coats/'
                   'dresses prevent movement and get damaged on the wall.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Trying an indoor bouldering session tonight with lots of climbing '
                               'and stretching. What should I wear?',
                               'Spending a couple of hours on the wall at the climbing gym. What '
                               'should I wear?'],
                  'implicit': ['A friend invited me to their climbing gym this week. Outfit '
                               'advice?',
                               'Booked my first session at the bouldering place downtown. What '
                               'should I wear?']}},
 {'scenario_id': 'field_road_run',
  'archetype': 'athletic_outdoor',
  'name': 'Long-distance Road Run',
  'tpo': {'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'exercise', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'tank_top', 'shorts', 'windbreaker', 'hoodie'],
                       'incompatible': ['trench_coat', 'blazer', 'dress', 'fleece_jacket', 'formal_shirt']},
  'justification': 'Running distance in a coat/blazer/dress is physically impossible and '
                   'overheating.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Running a half-marathon this weekend. What should I wear?',
                               'Going out for a long training run on the road today. Outfit?'],
                  'implicit': ["Training for my first marathon. What's the right thing to wear?",
                               'Building up my distance with weekend runs. Outfit advice?']}},
 {'scenario_id': 'field_tennis_match',
  'archetype': 'athletic_outdoor',
  'name': 'Outdoor Tennis Match',
  'tpo': {'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'exercise', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'tank_top', 'shorts', 'windbreaker', 'hoodie'],
                       'incompatible': ['trench_coat', 'blazer', 'dress', 'fleece_jacket', 'formal_shirt']},
  'justification': 'Playing tennis in a coat/fleece/dress is physically impossible.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Tennis match on an outdoor court this afternoon. What should I '
                               'wear?',
                               'Playing a couple of sets in the sun today. Outfit?'],
                  'implicit': ['Joined a weekend tennis club. What should I wear on court?',
                               'Booked a court with a friend for tomorrow. Outfit advice?']}},
 {'scenario_id': 'field_soccer_match',
  'archetype': 'athletic_outdoor',
  'name': 'Outdoor Soccer Match',
  'tpo': {'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'exercise', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'tank_top', 'shorts', 'windbreaker', 'hoodie'],
                       'incompatible': ['trench_coat', 'blazer', 'dress', 'fleece_jacket', 'formal_shirt']},
  'justification': 'Running a soccer match in a coat/blazer/dress is impossible and tears '
                   'garments.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Playing a full soccer match on grass this weekend. What should I '
                               'wear?',
                               'Out on the pitch running for 90 minutes today. Outfit?'],
                  'implicit': ['Joined a Sunday-league soccer team. What should I wear to play?',
                               'Friends organized a kickabout at the park field. Outfit advice?']}},
 {'scenario_id': 'rugged_mountain_hike',
  'archetype': 'rugged_outdoor',
  'name': 'Day Hike on Rough Mountain Trail',
  'tpo': {'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'travel', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['windbreaker',
                                      'hoodie',
                                      'jeans',
                                      't_shirt',
                                      'sweater',
                                      'fleece_jacket'],
                       'incompatible': ['blazer', 'dress', 'long_skirt', 'formal_shirt', 'slacks']},
  'justification': 'A suit/blazer/dress/skirt on a rugged hike is impractical and easily damaged.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Hiking a rough mountain trail all day with steep, rocky sections. '
                               'What should I wear?',
                               "Doing a long day hike over uneven terrain. What's the right "
                               'outfit?'],
                  'implicit': ['Planning a mountain hike with friends this weekend. Outfit advice?',
                               'Heading up the ridge trail on Saturday. What should I wear?']}},
 {'scenario_id': 'rugged_wilderness_camping',
  'archetype': 'rugged_outdoor',
  'name': 'Wilderness Camping Trip',
  'tpo': {'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'travel', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['windbreaker',
                                      'hoodie',
                                      'jeans',
                                      't_shirt',
                                      'sweater',
                                      'fleece_jacket'],
                       'incompatible': ['blazer', 'dress', 'long_skirt', 'formal_shirt', 'slacks']},
  'justification': 'Camping chores in a blazer/suit/dress are impractical and ruin the garment.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Camping in the backcountry for the weekend, setting up tents and '
                               'gathering wood. What should I wear?',
                               'Roughing it outdoors for two days of camping. Outfit?'],
                  'implicit': ['Planning a camping trip out in the woods with friends. What should '
                               'I wear?',
                               'Heading off-grid to a campsite this weekend. Outfit advice?']}},
 {'scenario_id': 'rugged_trail_scramble',
  'archetype': 'rugged_outdoor',
  'name': 'Rocky Trail Scramble',
  'tpo': {'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'travel', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['windbreaker',
                                      'hoodie',
                                      'jeans',
                                      't_shirt',
                                      'sweater',
                                      'fleece_jacket'],
                       'incompatible': ['blazer', 'dress', 'long_skirt', 'formal_shirt', 'slacks']},
  'justification': 'A rocky scramble in formal/delicate clothing is a safety hazard.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Scrambling over boulders and rough rock on a trail today. What '
                               'should I wear?',
                               "Doing a hands-on rocky scramble route. What's the right outfit?"],
                  'implicit': ['Going bouldering along a rugged trail this weekend. Outfit advice?',
                               'Tackling a steep, rocky route with a group. What should I wear?']}},
 {'scenario_id': 'weather_typhoon',
  'archetype': 'severe_weather',
  'name': 'Typhoon / Hurricane Errands',
  'tpo': {'time': {'weather': 'rainy', 'wind': 'extreme'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'errands'}},
  'garment_category': {'compatible': ['windbreaker', 'puffer_jacket', 'fleece_jacket'],
                       'incompatible': ['dress', 'tank_top', 'shorts', 'mini_skirt']},
  'justification': 'Typhoon-force wind and rain makes exposed-skin/delicate clothing dangerous.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['A typhoon is hitting and I have to go out for urgent errands. What '
                               'should I wear?',
                               'Driving rain and violent wind outside and I must head out. '
                               'Outfit?'],
                  'implicit': ['Need to run essential errands during the big storm. What should I '
                               'wear?',
                               "The storm's bad but I have to get to the store. Outfit advice?"]}},
 {'scenario_id': 'weather_hailstorm',
  'archetype': 'severe_weather',
  'name': 'Hailstorm Outdoor Exposure',
  'tpo': {'time': {'weather': 'hail'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'errands'}},
  'garment_category': {'compatible': ['windbreaker', 'puffer_jacket', 'fleece_jacket'],
                       'incompatible': ['dress', 'tank_top', 'shorts', 'mini_skirt']},
  'justification': 'Hailstones on bare skin cause injury; protective outerwear is essential.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ["There's a hailstorm and I have to walk home through it. What "
                               'should I wear?',
                               "Hail is coming down and I need to be outside briefly. What's the "
                               'right outfit?'],
                  'implicit': ['Caught out in sudden hail last time — what should I wear when it '
                               'happens again?',
                               'Forecast says hail and I still need to head out. Outfit advice?']}},
 {'scenario_id': 'weather_dust_storm',
  'archetype': 'severe_weather',
  'name': 'Sandstorm / Dust Storm Outdoor',
  'tpo': {'time': {'weather': 'windy', 'air': 'dust'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'errands'}},
  'garment_category': {'compatible': ['windbreaker', 'puffer_jacket', 'fleece_jacket'],
                       'incompatible': ['dress', 'tank_top', 'shorts', 'mini_skirt']},
  'justification': 'Sand/dust on bare skin causes abrasion; full coverage is needed.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['A dust storm is rolling in and I need to cross town. What should I '
                               'wear?',
                               'Thick blowing sand outside and I have to be out in it. Outfit?'],
                  'implicit': ['Living in a desert city during sandstorm season — what should I '
                               'wear outdoors?',
                               "The air's full of blowing dust and I need to head out. Outfit "
                               'advice?']}},
 {'scenario_id': 'weather_freezing_rain',
  'archetype': 'severe_weather',
  'name': 'Freezing Rain / Ice Storm Commute',
  'tpo': {'time': {'season': 'winter', 'weather': 'rainy'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'travel'}},
  'garment_category': {'compatible': ['windbreaker', 'puffer_jacket', 'fleece_jacket'],
                       'incompatible': ['dress', 'tank_top', 'shorts', 'mini_skirt', 't_shirt']},
  'justification': 'Freezing rain plus sub-zero wind chill makes light clothing '
                   'hypothermia-inducing.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ["Freezing rain and icy roads, and I'm walking to work. What should "
                               'I wear?',
                               'An ice storm is making my commute treacherous and wet. Outfit?'],
                  'implicit': ['The roads are iced over and I still have to get across town. What '
                               'should I wear?',
                               'Sleety, freezing commute ahead this morning. Outfit advice?']}},
 {'scenario_id': 'casual_park_picnic',
  'archetype': 'casual_leisure',
  'name': 'Weekend Park Picnic',
  'tpo': {'time': {'season': 'spring', 'weather': 'sunny'},
          'place': {'venue': 'park', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'casual_outing', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt',
                                      'shorts',
                                      'hoodie',
                                      'jeans',
                                      'sweater',
                                      'windbreaker',
                                      'cardigan',
                                      'sweatshirt'],
                       'incompatible': ['blazer', 'trench_coat', 'formal_shirt']},
  'justification': 'A suit jacket/blazer/trench coat or a pressed dress shirt at a casual park '
                   'picnic is universally over-dressed.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Having a relaxed picnic at the park on a sunny afternoon. What '
                               'should I wear?',
                               'Sitting on the grass with friends for a casual park hangout. '
                               'Outfit?'],
                  'implicit': ['Meeting friends at the park for a lazy Sunday. What should I wear?',
                               'Packing a basket for an afternoon in the park. Outfit advice?']}},
 {'scenario_id': 'casual_farmers_market',
  'archetype': 'casual_leisure',
  'name': "Weekend Farmers' Market",
  'tpo': {'time': {'season': 'spring', 'weather': 'sunny'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'casual_outing', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt',
                                      'shorts',
                                      'hoodie',
                                      'jeans',
                                      'sweater',
                                      'windbreaker',
                                      'cardigan',
                                      'sweatshirt'],
                       'incompatible': ['blazer', 'trench_coat', 'formal_shirt']},
  'justification': "A suit/blazer or a pressed dress shirt at a casual farmers' market is "
                   'over-dressed.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ["Strolling a weekend farmers' market for a couple of hours. What "
                               'should I wear?',
                               'Wandering the outdoor market stalls on a casual Saturday. Outfit?'],
                  'implicit': ['Heading to the outdoor local market for produce this morning. What '
                               'should I wear?',
                               "Browsing the farmers' market stalls tomorrow morning. Outfit "
                               'advice?']}},
 {'scenario_id': 'casual_amusement_park',
  'archetype': 'casual_leisure',
  'name': 'Amusement Park All-Day Visit',
  'tpo': {'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'casual_outing', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt',
                                      'shorts',
                                      'hoodie',
                                      'jeans',
                                      'sweater',
                                      'windbreaker',
                                      'cardigan',
                                      'sweatshirt'],
                       'incompatible': ['blazer', 'trench_coat', 'formal_shirt', 'slacks']},
  'justification': 'Formal wear (blazer, dress shirt, pressed slacks) at an amusement park '
                   'restricts rides and movement.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Spending the whole day at an amusement park going on rides. What '
                               'should I wear?',
                               'On my feet all day at a theme park with lots of rides. Outfit?'],
                  'implicit': ['Taking the family to the theme park this weekend. What should I '
                               'wear?',
                               'Day out at the amusement park tomorrow. Outfit advice?']}},
 {'scenario_id': 'casual_zoo_day',
  'archetype': 'casual_leisure',
  'name': 'Family Day at the Zoo',
  'tpo': {'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'casual_outing', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt',
                                      'shorts',
                                      'hoodie',
                                      'jeans',
                                      'sweater',
                                      'windbreaker',
                                      'cardigan',
                                      'sweatshirt'],
                       'incompatible': ['blazer', 'trench_coat', 'formal_shirt', 'slacks']},
  'justification': 'Formal attire (blazer, dress shirt, pressed slacks) for a full day of walking '
                   'at the zoo is impractical and over-dressed.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Walking around the zoo all day with the family. What should I '
                               'wear?',
                               'A full day of walking outdoors at the zoo. Outfit?'],
                  'implicit': ['Taking the kids to the zoo this weekend. What should I wear?',
                               'Day trip to the zoo tomorrow. Outfit advice?']}},
 {'scenario_id': 'work_moving_day',
  'archetype': 'practical_work',
  'name': 'Helping a Friend Move House',
  'tpo': {'place': {'indoor_outdoor': 'mixed'},
          'occasion': {'activity': 'errands', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'sweatshirt', 'hoodie', 'jeans', 'shorts'],
                       'incompatible': ['blazer', 'trench_coat', 'formal_shirt', 'slacks',
                                        'dress', 'long_skirt', 'mini_skirt']},
  'justification': 'Hauling boxes and furniture up stairs all day ruins delicate or formal '
                   'garments and restricts movement.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ["I'm helping a friend move apartments all day — hauling boxes and "
                               'furniture. What should I wear?',
                               'Moving day: carrying heavy boxes up and down stairs for hours. '
                               'What outfit makes sense?'],
                  'implicit': ['Finally moving into my new place this weekend. What should I '
                               'wear?',
                               'Promised to help my sister move on Saturday. Outfit advice?']}},
 {'scenario_id': 'work_home_renovation',
  'archetype': 'practical_work',
  'name': 'Home Renovation / Painting Day',
  'tpo': {'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'errands', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'sweatshirt', 'hoodie', 'jeans', 'shorts'],
                       'incompatible': ['blazer', 'trench_coat', 'formal_shirt', 'slacks',
                                        'dress', 'long_skirt', 'mini_skirt']},
  'justification': 'Painting and sanding cover clothing in paint and dust; formal or delicate '
                   'garments are ruined.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ["I'm painting my living room and sanding walls all weekend. What "
                               'should I wear?',
                               'Doing messy DIY renovation work around the house today. What '
                               'should I wear?'],
                  'implicit': ['Redecorating my apartment this weekend. What should I wear?',
                               'Got a big DIY project around the house tomorrow. Outfit '
                               'advice?']}},
 {'scenario_id': 'work_garden_volunteering',
  'archetype': 'practical_work',
  'name': 'Community Garden Volunteering',
  'tpo': {'time': {'season': 'spring', 'weather': 'sunny'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'casual_outing', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'sweatshirt', 'hoodie', 'jeans', 'shorts'],
                       'incompatible': ['blazer', 'trench_coat', 'formal_shirt', 'slacks',
                                        'dress', 'long_skirt', 'mini_skirt']},
  'justification': 'Digging and planting in soil demand durable, washable clothing; formal or '
                   'delicate items are impractical and get ruined.',
  'color': None,
  'pattern': None,
  'query_seeds': {'explicit': ['Volunteering at the community garden — digging beds and planting '
                               'all morning. What should I wear?',
                               'Spending the day doing yard work and planting in the dirt. What '
                               'should I wear?'],
                  'implicit': ['Signed up for the neighborhood garden day this weekend. What '
                               'should I wear?',
                               'Helping out at the local community garden tomorrow. Outfit '
                               'advice?']}},
 {'scenario_id': 'biz_corporate_board',
  'archetype': 'business_professional',
  'name': 'Corporate Board Meeting',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'office', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'work_meeting', 'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Hoodie/shorts and loud colors/prints at a corporate board meeting are '
                   'universally inappropriate.',
  'color': {'compatible': ['black', 'navy', 'gray', 'white', 'beige'],
            'incompatible': ['orange', 'yellow']},
  'pattern': {'compatible': ['solid', 'striped', 'checkered'],
              'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ['I have a corporate board meeting today and the dress code is '
                               'strictly formal. What should I wear?',
                               'Presenting to the executive board in a formal boardroom this '
                               'afternoon. What should I wear?'],
                  'implicit': ["Big meeting with the company's senior leadership this afternoon. "
                               'Outfit advice?',
                               "I'm sitting in with the directors at headquarters today. What "
                               'should I wear?']}},
 {'scenario_id': 'biz_investor_pitch',
  'archetype': 'business_professional',
  'name': 'Investor Pitch Presentation',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'office', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'work_meeting', 'formality_required': 'business_casual'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Investor pitches demand credibility; casual wear and loud prints undermine '
                   'professional trust.',
  'color': {'compatible': ['black', 'navy', 'gray', 'white', 'beige'],
            'incompatible': ['orange', 'yellow']},
  'pattern': {'compatible': ['solid', 'striped', 'checkered'],
              'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ['Pitching to venture capital investors today; formal business '
                               'attire is expected. What should I wear?',
                               'Presenting our funding pitch to investors in a few hours; '
                               'professional attire expected. What should I wear?'],
                  'implicit': ['Pitch meeting with VCs tomorrow morning to raise our round. '
                               'Outfit advice?',
                               'Trying to win over serious investors in a polished office pitch '
                               'tomorrow. What should I wear?']}},
 {'scenario_id': 'biz_job_interview',
  'archetype': 'business_professional',
  'name': 'Formal Job Interview',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'office', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'work_meeting', 'formality_required': 'business_casual'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Hoodie/shorts and loud prints to a formal interview signal disrespect for the '
                   'opportunity.',
  'color': {'compatible': ['black', 'navy', 'gray', 'white', 'beige'],
            'incompatible': ['orange', 'yellow']},
  'pattern': {'compatible': ['solid', 'striped', 'checkered'],
              'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ['I have an in-person job interview at a corporate office tomorrow '
                               'and formal attire is expected. What should I wear?',
                               'Interviewing for a professional role at their headquarters '
                               'tomorrow; business attire expected. What should I wear?'],
                  'implicit': ['Final-round interview with the hiring panel next week. Outfit '
                               'advice?',
                               'Meeting the team that decides if I get the job tomorrow. What '
                               'should I wear?']}},
 {'scenario_id': 'biz_client_meeting',
  'archetype': 'business_professional',
  'name': 'Client Meeting / Business Dinner',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'venue': 'restaurant', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'work_meeting', 'formality_required': 'business_casual'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Business meetings represent the company; casual wear and loud prints are a '
                   'professional failure.',
  'color': {'compatible': ['black', 'navy', 'gray', 'white', 'beige'],
            'incompatible': ['orange', 'yellow']},
  'pattern': {'compatible': ['solid', 'striped', 'checkered'],
              'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ['Business dinner with an important client tonight; professional '
                               'attire expected. What should I wear?',
                               'Meeting a key client over dinner to represent the firm; smart '
                               'business attire expected. What should I wear?'],
                  'implicit': ['Taking our biggest client out to dinner tonight. Outfit advice?',
                               'Dinner with the client whose account I manage. What should I '
                               'wear?']}},
 {'scenario_id': 'formal_black_tie_gala',
  'archetype': 'ultra_formal',
  'name': 'Black-Tie Gala',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'social_gathering', 'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'dress', 'formal_shirt'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Black-tie events have an explicit dress code that excludes casual and loud '
                   'clothing.',
  'color': {'compatible': ['black', 'navy', 'gray', 'white'],
            'incompatible': ['orange', 'yellow', 'pink'],
            'violation_garment_scope': ['blazer', 'formal_shirt', 'slacks']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ['Attending a black-tie gala tonight with a strict formal dress '
                               'code. What should I wear?',
                               'Invited to a formal evening ball — black tie. What should I wear?'],
                  'implicit': ['Got an invitation to a glamorous evening fundraiser at a grand '
                               'hotel. Outfit advice?',
                               'Invited to an evening charity ball downtown tonight. What should '
                               'I wear?']}},
 {'scenario_id': 'formal_opera_premiere',
  'archetype': 'ultra_formal',
  'name': 'Opera / Ballet Premiere',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'venue': 'concert_hall', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'dress', 'formal_shirt'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Opera/ballet premieres with dress codes prohibit casual and loud attire.',
  'color': {'compatible': ['black', 'navy', 'gray', 'white'],
            'incompatible': ['orange', 'yellow', 'pink'],
            'violation_garment_scope': ['blazer', 'formal_shirt', 'slacks']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ['Attending an opera premiere tonight; they enforce a formal dress '
                               'code. What should I wear?',
                               'Going to opening night at the opera house with a dress code. What '
                               'should I wear?'],
                  'implicit': ['Have tickets to the gala opening night at the opera. Outfit '
                               'advice?',
                               'Premiere performance at the grand concert hall on opening night. '
                               'What should I wear?']}},
 {'scenario_id': 'formal_national_award',
  'archetype': 'ultra_formal',
  'name': 'National Award Ceremony (Recipient)',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'dress', 'formal_shirt'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Receiving a national award in a hoodie/shorts or loud prints disrespects the '
                   'institution.',
  'color': {'compatible': ['black', 'navy', 'gray', 'white'],
            'incompatible': ['orange', 'yellow', 'pink'],
            'violation_garment_scope': ['blazer', 'formal_shirt', 'slacks']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ["I'm receiving a national award tonight at a formal ceremony on "
                               'stage. What should I wear?',
                               'Being honored at a formal national award ceremony this evening. '
                               'What should I wear?'],
                  'implicit': ["I'm being recognized on stage at a prestigious ceremony tonight. "
                               'Outfit advice?',
                               'Accepting a major honor at the national hall this evening. What '
                               'should I wear?']}},
 {'scenario_id': 'formal_state_banquet',
  'archetype': 'ultra_formal',
  'name': 'Head-of-State Banquet',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'social_gathering', 'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'dress', 'formal_shirt'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'State banquets enforce strict formal protocol; casual or loud attire is '
                   'unacceptable.',
  'color': {'compatible': ['black', 'navy', 'gray', 'white'],
            'incompatible': ['orange', 'yellow', 'pink'],
            'violation_garment_scope': ['blazer', 'formal_shirt', 'slacks']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ['Invited to a state banquet hosted by a head of state, strict '
                               'formal protocol. What should I wear?',
                               'Attending a formal banquet at the presidential residence. What '
                               'should I wear?'],
                  'implicit': ['Got an invitation to dine at the presidential palace. Outfit '
                               'advice?',
                               'Attending an official banquet with dignitaries this evening. What '
                               'should I wear?']}},
 {'scenario_id': 'civic_court_appearance',
  'archetype': 'judicial_civic',
  'name': 'Court Appearance',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Court appearances require conservative formal dress; casual/loud wear may '
                   'constitute contempt.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['orange', 'yellow', 'pink', 'red']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ['I have to appear in court tomorrow before a judge; conservative '
                               'formal dress is expected. What should I wear?',
                               'Appearing in a courtroom for a formal hearing tomorrow. What '
                               'should I wear?'],
                  'implicit': ["I've been called before a judge next week. Outfit advice?",
                               'Have to show up at the courthouse for my hearing. What should I '
                               'wear?']}},
 {'scenario_id': 'civic_supreme_argument',
  'archetype': 'judicial_civic',
  'name': 'High-Court Oral Argument',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'High-court proceedings demand the most conservative formal attire.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['orange', 'yellow', 'pink', 'red']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ['Presenting an oral argument before the high court, where strict '
                               'formal dress is required. What should I wear?',
                               'Arguing a case at the highest court next month; the dress code is '
                               'strictly conservative. What should I wear?'],
                  'implicit': ["I'm the attorney appearing at the top court next month. Outfit "
                               'advice?',
                               'Standing before the senior bench to argue a case soon. What should '
                               'I wear?']}},
 {'scenario_id': 'civic_govt_hearing',
  'archetype': 'judicial_civic',
  'name': 'Government / Parliamentary Hearing',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Official government hearings follow conservative formal norms; casual/loud '
                   'attire is improper.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['orange', 'yellow', 'pink', 'red']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ['Testifying at a formal government hearing on the record. What '
                               'should I wear?',
                               'Appearing before a parliamentary committee to give testimony; '
                               'formal dress expected. What should I wear?'],
                  'implicit': ["I've been summoned to speak before a legislative committee. Outfit "
                               'advice?',
                               'Giving official testimony at a public hearing next week. What '
                               'should I wear?']}},
 {'scenario_id': 'civic_citizenship_oath',
  'archetype': 'judicial_civic',
  'name': 'Citizenship / Naturalization Oath Ceremony',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'A naturalization oath is an official state ceremony; casual or loud attire '
                   'disrespects the occasion.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['orange', 'yellow', 'pink', 'red']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ["I'm taking the oath at my citizenship ceremony next week; formal "
                               'dress is expected. What should I wear?',
                               'Attending my naturalization ceremony at the federal courthouse; '
                               'formal attire expected. What should I wear?'],
                  'implicit': ['I finally become a citizen at an official ceremony this month. '
                               'Outfit advice?',
                               'Big official ceremony at the courthouse where I get sworn in. '
                               'What should I wear?']}},
 {'scenario_id': 'mourn_funeral',
  'archetype': 'mourning_somber',
  'name': 'Funeral Service',
  'tpo': {'place': {'indoor_outdoor': 'mixed'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'trench_coat', 'formal_shirt', 'dress', 'sweater'],
                       'incompatible': ['shorts', 'tank_top', 'hoodie', 't_shirt']},
  'justification': 'Funerals universally demand dark, somber attire; bright/casual/loud clothing '
                   'is disrespectful.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['orange', 'yellow', 'pink', 'red']},
  'pattern': {'compatible': ['solid', 'striped'], 'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ["I'm attending a funeral and need to dress respectfully and "
                               'somberly. What should I wear?',
                               'Going to a funeral service this week; dark, formal dress expected. '
                               'What should I wear?'],
                  'implicit': ['I have to say goodbye to a relative at the service this weekend. '
                               'Outfit advice?',
                               'Attending the burial of a close family friend. What should I '
                               'wear?']}},
 {'scenario_id': 'mourn_memorial',
  'archetype': 'mourning_somber',
  'name': 'Memorial / Remembrance Service',
  'tpo': {'place': {'indoor_outdoor': 'mixed'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'formal'}},
  'garment_category': {'compatible': ['blazer', 'trench_coat', 'formal_shirt', 'dress', 'sweater'],
                       'incompatible': ['shorts', 'tank_top', 'hoodie', 't_shirt']},
  'justification': 'Memorial services share funeral solemnity; bright/festive attire is '
                   'disrespectful.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['orange', 'yellow', 'pink', 'red']},
  'pattern': {'compatible': ['solid', 'striped'], 'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ['Attending a memorial remembrance service for a colleague; dark, '
                               'respectful dress expected. What should I wear?',
                               'Going to a solemn remembrance ceremony this week. What should I '
                               'wear?'],
                  'implicit': ["There's a service to honor someone who passed, and I'm attending. "
                               'Outfit advice?',
                               'Going to a remembrance gathering for a late mentor. What should I '
                               'wear?']}},
 {'scenario_id': 'mourn_vigil',
  'archetype': 'mourning_somber',
  'name': 'Candlelight Vigil',
  'tpo': {'time': {'time_of_day': 'night'},
          'place': {'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'smart_casual'}},
  'garment_category': {'compatible': ['trench_coat', 'blazer', 'formal_shirt', 'sweater'],
                       'incompatible': ['shorts', 'tank_top', 't_shirt']},
  'justification': 'Candlelight vigils are solemn; bright/festive colors and loud prints are '
                   'inappropriate.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['orange', 'yellow', 'pink', 'red']},
  'pattern': {'compatible': ['solid', 'striped'], 'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ['Attending a candlelight vigil tonight to mourn and pay respects. '
                               'What should I wear?',
                               'Going to a solemn evening vigil; subdued dress expected. What '
                               'should I wear?'],
                  'implicit': ["There's a community gathering tonight to grieve a tragedy. Outfit "
                               'advice?',
                               'Joining a quiet evening tribute with candles tonight. What should '
                               'I wear?']}},
 {'scenario_id': 'mourn_condolence_visit',
  'archetype': 'mourning_somber',
  'name': 'Condolence Visit to Bereaved Family',
  'tpo': {'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'smart_casual'}},
  'garment_category': {'compatible': ['formal_shirt', 'blazer', 'sweater', 'trench_coat', 'dress'],
                       'incompatible': ['shorts', 'tank_top', 'hoodie', 't_shirt']},
  'justification': 'Visiting a grieving family in bright party colors or loud prints is '
                   'universally insensitive.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['orange', 'yellow', 'pink', 'red']},
  'pattern': {'compatible': ['solid', 'striped'], 'incompatible': ['leopard', 'polka_dot', 'floral']},
  'query_seeds': {'explicit': ['Visiting a grieving family to offer my condolences in person. What '
                               'should I wear?',
                               "Paying respects at a bereaved family's home; subdued dress. What "
                               'should I wear?'],
                  'implicit': ['Going to sit with a friend whose parent just passed. Outfit '
                               'advice?',
                               'Visiting a household in mourning to offer support. What should I '
                               'wear?']}},
 {'scenario_id': 'relig_conservative_worship',
  'archetype': 'religious_modest',
  'name': 'Conservative Religious Worship Service',
  'tpo': {'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'business_casual'}},
  'garment_category': {'compatible': ['formal_shirt',
                                      'blazer',
                                      'dress',
                                      'sweater',
                                      'trench_coat',
                                      'long_skirt',
                                      'cardigan'],
                       'incompatible': ['tank_top', 'shorts', 't_shirt', 'hoodie']},
  'justification': 'Conservative worship has modesty norms; tank tops/shorts and loud prints are '
                   'widely prohibited.',
  'color': None,
  'pattern': {'compatible': ['solid', 'striped'], 'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ['Attending a conservative religious service where modest dress is '
                               'required. What should I wear?',
                               'Going to worship with a traditional congregation; modesty '
                               'expected. What should I wear?'],
                  'implicit': ["Joining a friend's family for their weekly service at a "
                               'traditional congregation. Outfit advice?',
                               'Attending a holy-day service at a deeply traditional congregation '
                               'with relatives. What should I wear?']}},
 {'scenario_id': 'relig_temple_ceremony',
  'archetype': 'religious_modest',
  'name': 'Temple Ceremony / Sacred Site Visit',
  'tpo': {'place': {'venue': 'temple', 'indoor_outdoor': 'mixed'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'business_casual'}},
  'garment_category': {'compatible': ['formal_shirt',
                                      'blazer',
                                      'dress',
                                      'sweater',
                                      'trench_coat',
                                      'long_skirt',
                                      'cardigan'],
                       'incompatible': ['tank_top', 'shorts', 't_shirt', 'hoodie']},
  'justification': 'Sacred sites require covered, modest attire; exposed/loud clothing is '
                   'prohibited.',
  'color': None,
  'pattern': {'compatible': ['solid', 'striped'], 'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ['Visiting a sacred temple where shoulders and knees must be '
                               'covered. What should I wear?',
                               'Attending a temple ceremony with strict modesty rules. What should '
                               'I wear?'],
                  'implicit': ['Touring several historic temples and shrines on my trip next '
                               'week. Outfit advice?',
                               'Invited to a ceremony at a temple this weekend. What should I '
                               'wear?']}},
 {'scenario_id': 'relig_solemn_observance',
  'archetype': 'religious_modest',
  'name': 'Solemn Religious Observance',
  'tpo': {'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'business_casual'}},
  'garment_category': {'compatible': ['formal_shirt',
                                      'blazer',
                                      'dress',
                                      'sweater',
                                      'trench_coat',
                                      'long_skirt',
                                      'cardigan'],
                       'incompatible': ['tank_top', 'shorts', 't_shirt', 'hoodie']},
  'justification': 'Solemn observances require modest, subdued attire; exposed/loud clothing is '
                   'improper.',
  'color': None,
  'pattern': {'compatible': ['solid', 'striped'], 'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ['Attending a solemn religious observance where modest, subdued '
                               'dress is expected. What should I wear?',
                               'Going to an important religious holy-day service with modesty '
                               'norms. What should I wear?'],
                  'implicit': ['Joining a major holy-day gathering at the congregation. Outfit '
                               'advice?',
                               'Attending a significant religious observance with my family. What '
                               'should I wear?']}},
 {'scenario_id': 'relig_baptism_guest',
  'archetype': 'religious_modest',
  'name': 'Baptism / Christening (Guest)',
  'tpo': {'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'business_casual'}},
  'garment_category': {'compatible': ['formal_shirt',
                                      'blazer',
                                      'dress',
                                      'sweater',
                                      'trench_coat',
                                      'long_skirt',
                                      'cardigan'],
                       'incompatible': ['tank_top', 'shorts', 't_shirt', 'hoodie']},
  'justification': 'A christening in a house of worship expects modest, respectful dress; '
                   'exposed or loud clothing is inappropriate.',
  'color': None,
  'pattern': {'compatible': ['solid', 'striped'], 'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ["Invited to a baby's baptism at a traditional church where modest "
                               'dress is expected. What should I wear?',
                               'Attending a christening ceremony where modest dress is expected. '
                               'What should I wear?'],
                  'implicit': ["My friend's baby is being baptized this Sunday and I'm invited. "
                               'Outfit advice?',
                               'Going to a family christening at their congregation. What should '
                               'I wear?']}},
 {'scenario_id': 'social_gallery_opening',
  'archetype': 'semi_formal_social',
  'name': 'Art Gallery Opening / Vernissage',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'venue': 'art_gallery', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'social_gathering', 'formality_required': 'smart_casual'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'dress', 'long_skirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top']},
  'justification': 'Gallery openings are smart-casual; shorts/hoodie and loud prints signal '
                   'disinterest.',
  'color': None,
  'pattern': {'compatible': ['solid', 'striped', 'checkered', 'floral'],
              'incompatible': ['leopard']},
  'query_seeds': {'explicit': ['Attending an art gallery opening tonight; smart-casual minimum. '
                               'What should I wear?',
                               'Going to a vernissage at a downtown gallery this evening; smart '
                               'casual expected. What should I wear?'],
                  'implicit': ['Got invited to an evening gallery exhibition opening reception. '
                               'Outfit advice?',
                               'Heading to an opening reception at the art gallery. What should I '
                               'wear?']}},
 {'scenario_id': 'social_alumni_dinner',
  'archetype': 'semi_formal_social',
  'name': 'University Alumni Formal Dinner',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'venue': 'restaurant', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'social_gathering', 'formality_required': 'business_casual'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'dress', 'long_skirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top']},
  'justification': 'Alumni formal dinners have dress codes; hoodies/shorts and loud prints are out '
                   'of place.',
  'color': None,
  'pattern': {'compatible': ['solid', 'striped', 'checkered', 'floral'],
              'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ['Alumni formal dinner tonight in a hotel ballroom; smart attire '
                               'expected. What should I wear?',
                               'Going to a formal alumni reunion dinner this evening. What should '
                               'I wear?'],
                  'implicit': ["My old university is hosting a reunion dinner and I'm going. "
                               'Outfit advice?',
                               'Catching up with classmates at a fancy reunion dinner. What should '
                               'I wear?']}},
 {'scenario_id': 'social_company_gala',
  'archetype': 'semi_formal_social',
  'name': 'Company Annual Gala / Holiday Party',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'social_gathering', 'formality_required': 'business_casual'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'dress', 'long_skirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Company galas have implicit dress codes; casual wear and loud prints harm '
                   'professional standing.',
  'color': {'compatible': ['black', 'navy', 'gray', 'white', 'beige'],
            'incompatible': ['orange', 'yellow']},
  'pattern': {'compatible': ['solid', 'striped', 'checkered'],
              'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ['Company annual gala tonight; smart-casual to semi-formal expected. '
                               'What should I wear?',
                               'Going to the office year-end gala this evening; semi-formal '
                               'expected. What should I wear?'],
                  'implicit': ["Our company's big end-of-year party is tonight. Outfit advice?",
                               "Heading to the firm's annual celebration dinner. What should I "
                               'wear?']}},
 {'scenario_id': 'social_first_date_upscale',
  'archetype': 'semi_formal_social',
  'name': 'First Date at an Upscale Restaurant',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'venue': 'restaurant', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'first_date', 'formality_required': 'smart_casual'}},
  'garment_category': {'compatible': ['formal_shirt', 'blazer', 'dress', 'long_skirt', 'slacks'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top']},
  'justification': 'An upscale restaurant date in a hoodie/shorts or loud prints signals '
                   'disrespect.',
  'color': None,
  'pattern': {'compatible': ['solid', 'striped', 'floral', 'checkered'],
              'incompatible': ['leopard']},
  'query_seeds': {'explicit': ['First date at an upscale restaurant tonight; I want to look '
                               'put-together. What should I wear?',
                               'Dinner date at a nice, dressy restaurant this evening. What should '
                               'I wear?'],
                  'implicit': ['Taking someone I like to a really nice restaurant tonight. Outfit '
                               'advice?',
                               'Big first date at a fancy spot downtown tonight. What should I '
                               'wear?']}},
 {'scenario_id': 'wedding_reception',
  'archetype': 'wedding_celebration',
  'name': 'Wedding Reception (Guest)',
  'tpo': {'place': {'venue': 'wedding_venue', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'smart_casual'}},
  'garment_category': {'compatible': ['blazer', 'dress', 'formal_shirt', 'long_skirt'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Wearing shorts/hoodie or loud prints to a wedding reception insults the '
                   'couple.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['white']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'floral']},
  'query_seeds': {'explicit': ['Attending a wedding reception in a hotel ballroom; smart attire '
                               'expected. What should I wear?',
                               "Going to a friend's wedding reception this weekend; proper guest "
                               'attire expected. What should I wear?'],
                  'implicit': ["My friend is getting married and I'm going to the reception next "
                               'Saturday. Outfit advice?',
                               'Invited to celebrate a couple at their reception dinner. What '
                               'should I wear?']}},
 {'scenario_id': 'wedding_garden_ceremony',
  'archetype': 'wedding_celebration',
  'name': 'Garden Wedding Ceremony (Guest)',
  'tpo': {'time': {'season': 'spring', 'weather': 'sunny'},
          'place': {'venue': 'wedding_venue', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'smart_casual'}},
  'garment_category': {'compatible': ['blazer', 'dress', 'formal_shirt', 'long_skirt'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'A garden wedding still expects smart guest attire; shorts/hoodie and loud '
                   'prints are out of place.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['white']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'floral']},
  'query_seeds': {'explicit': ['Attending an outdoor garden wedding ceremony as a guest; smart '
                               'daytime attire expected. What should I wear?',
                               'Going to a daytime garden wedding; smart attire expected. What '
                               'should I wear?'],
                  'implicit': ["A couple I know is marrying in a garden this spring and I'm "
                               'invited. Outfit advice?',
                               'Heading to a lovely outdoor wedding at a botanical venue. What '
                               'should I wear?']}},
 {'scenario_id': 'wedding_anniversary_party',
  'archetype': 'wedding_celebration',
  'name': 'Milestone Anniversary / Engagement Celebration',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'social_gathering', 'formality_required': 'smart_casual'}},
  'garment_category': {'compatible': ['blazer', 'dress', 'formal_shirt', 'long_skirt'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'Milestone celebrations expect smart attire; shorts/hoodie and loud prints are '
                   'out of place.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['white']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'floral']},
  'query_seeds': {'explicit': ['Attending a milestone anniversary celebration dinner; smart attire '
                               'expected. What should I wear?',
                               'Going to an engagement celebration party this evening; dressy '
                               'attire expected. What should I wear?'],
                  'implicit': ["My parents' big anniversary dinner at a nice venue is this "
                               'weekend. Outfit advice?',
                               "Celebrating a couple's engagement at a nice venue tonight. What "
                               'should I wear?']}},
 {'scenario_id': 'celebration_graduation',
  'archetype': 'wedding_celebration',
  'name': 'University Graduation Ceremony (Guest)',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'indoor_outdoor': 'mixed'},
          'occasion': {'activity': 'ceremony_attendance', 'formality_required': 'smart_casual'}},
  'garment_category': {'compatible': ['blazer', 'dress', 'formal_shirt', 'long_skirt'],
                       'incompatible': ['hoodie', 'shorts', 'tank_top', 't_shirt']},
  'justification': 'A graduation ceremony expects smart guest attire; hoodies/shorts and loud '
                   'animal prints are out of place.',
  'color': {'compatible': ['black', 'navy', 'gray'],
            'incompatible': ['white']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'floral']},
  'query_seeds': {'explicit': ["Attending my sister's university graduation ceremony; smart "
                               'attire expected. What should I wear?',
                               'Going to a commencement ceremony at the big auditorium; neat, '
                               'respectful attire expected. What should I wear?'],
                  'implicit': ["My best friend graduates next week and I'll be in the audience. "
                               'Outfit advice?',
                               'Watching the commencement at the university hall this Friday. '
                               'What should I wear?']}},

 # ── v2 archetypes: functional / non-formal coded scenarios ──────────────
 # These fill the constraint-source x formality grid (see archetype index
 # v2): functional-casual (safety_visibility, field_stealth, stage_media),
 # functional-formal (stage_tv_interview), normative-casual (club_code).
 # Niche-knowledge scenarios (field_wildlife_hide, stage_greenscreen_shoot)
 # carry the rule inside their implicit seeds (weak-implicit by design).
 {'scenario_id': 'visib_night_road_run',
  'archetype': 'safety_visibility',
  'name': 'Night Run Along an Unlit Road',
  'tpo': {'time': {'time_of_day': 'night'},
          'place': {'venue': 'roadside', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'running', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['windbreaker', 't_shirt', 'sweatshirt', 'leggings', 'shorts'],
                       'incompatible': ['blazer', 'trench_coat', 'slacks', 'dress', 'long_skirt']},
  'justification': 'Runners beside traffic at night must be visible to drivers; dark clothing '
                   'is a documented safety hazard, and restrictive formal wear impedes running.',
  'color': {'compatible': ['yellow', 'orange', 'white'],
            'incompatible': ['black', 'navy', 'gray', 'brown']},
  'pattern': None,
  'query_seeds': {'explicit': ['Going for a run along an unlit country road tonight; I need to '
                               'stay visible to drivers. What should I wear?',
                               'Night run on the road shoulder tonight — high-visibility gear '
                               'recommended. What should I wear?'],
                  'implicit': ['Heading out for a run along the highway after sunset. What '
                               'should I wear?',
                               'Evening jog on the roadside once it gets dark. Outfit advice?']}},
 {'scenario_id': 'visib_night_bike_commute',
  'archetype': 'safety_visibility',
  'name': 'Cycling Home After Dark',
  'tpo': {'time': {'time_of_day': 'night'},
          'place': {'venue': 'road', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'cycling', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['windbreaker', 't_shirt', 'sweatshirt', 'leggings'],
                       'incompatible': ['trench_coat', 'long_skirt', 'dress', 'blazer']},
  'justification': 'Cyclists in traffic after dark need high-visibility clothing; dark colors '
                   'make riders invisible to drivers, and loose formal garments catch in the bike.',
  'color': {'compatible': ['yellow', 'orange', 'white'],
            'incompatible': ['black', 'navy', 'gray', 'brown']},
  'pattern': None,
  'query_seeds': {'explicit': ['Cycling home on city roads after dark; I want drivers to see me. '
                               'What should I wear?',
                               'Night bike commute along the main road; high-visibility clothing '
                               'advised. What should I wear?'],
                  'implicit': ['Riding my bike home late tonight. What should I wear?',
                               'Cycling back from work after sunset today. Outfit advice?']}},
 {'scenario_id': 'visib_dawn_roadside_cleanup',
  'archetype': 'safety_visibility',
  'name': 'Dawn Roadside Cleanup Volunteering',
  'tpo': {'time': {'time_of_day': 'early_morning'},
          'place': {'venue': 'roadside', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'volunteering', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'sweatshirt', 'windbreaker', 'jeans', 'leggings'],
                       'incompatible': ['dress', 'blazer', 'slacks', 'trench_coat']},
  'justification': 'Working on a road shoulder in low dawn light requires bright, visible '
                   'clothing for safety and practical garments for physical work.',
  'color': {'compatible': ['yellow', 'orange', 'white'],
            'incompatible': ['black', 'navy', 'gray', 'brown']},
  'pattern': None,
  'query_seeds': {'explicit': ['Volunteering for a roadside litter cleanup before sunrise; we '
                               'were told to stay visible to traffic. What should I wear?',
                               'Early-morning highway cleanup event; bright, practical clothes '
                               'recommended. What should I wear?'],
                  'implicit': ['Joining a roadside cleanup crew at dawn tomorrow. What should I '
                               'wear?',
                               'Helping pick up litter along the highway early tomorrow. Outfit '
                               'advice?']}},
 {'scenario_id': 'field_safari_tour',
  'archetype': 'field_stealth',
  'name': 'African Safari Game Drive',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'savanna', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'wildlife_viewing', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['formal_shirt', 't_shirt', 'windbreaker', 'jeans', 'shorts'],
                       'incompatible': ['dress', 'mini_skirt', 'blazer', 'leather_jacket']},
  'justification': 'Safari operators require muted earth tones: bright colors and loud prints '
                   'startle wildlife, and animal/camouflage prints are prohibited or restricted '
                   'in several countries.',
  'color': {'compatible': ['brown', 'beige', 'green'],
            'incompatible': ['red', 'orange', 'yellow', 'pink', 'white']},
  'pattern': {'compatible': ['solid', 'striped', 'checkered'],
              'incompatible': ['leopard', 'floral', 'polka_dot']},
  'query_seeds': {'explicit': ['Booked a safari game drive; the guide asked for muted earth '
                               'tones only. What should I wear?',
                               'Safari tour tomorrow — the operator says no bright colors or '
                               'animal prints. What should I wear?'],
                  'implicit': ['Going on an African safari game drive next week. Outfit advice?',
                               'First safari trip — riding in an open jeep to see the animals. '
                               'What should I wear?']}},
 {'scenario_id': 'field_paintball_match',
  'archetype': 'field_stealth',
  'name': 'Forest Paintball Match',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'forest', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'paintball', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['windbreaker', 'sweatshirt', 'hoodie', 'jeans', 'slacks'],
                       'incompatible': ['tank_top', 'shorts', 'mini_skirt', 'dress']},
  'justification': 'Woodland paintball rewards dark natural colors that blend into cover; '
                   'bright colors make you an easy target, and exposed skin bruises from hits.',
  'color': {'compatible': ['green', 'brown', 'black'],
            'incompatible': ['white', 'yellow', 'pink', 'orange', 'red']},
  'pattern': {'compatible': ['solid', 'striped', 'checkered'],
              'incompatible': ['polka_dot', 'floral']},
  'query_seeds': {'explicit': ['Forest paintball match this weekend; I want to blend in and '
                               'keep my skin covered. What should I wear?',
                               'Playing woodland paintball — dark colors and full coverage '
                               'recommended. What should I wear?'],
                  'implicit': ['Playing paintball out in the woods with friends on Saturday. '
                               'What should I wear?',
                               'First paintball game at the forest field this weekend. Outfit '
                               'advice?']}},
 {'scenario_id': 'field_wildlife_hide',
  'archetype': 'field_stealth',
  'name': 'Wildlife Photography from a Hide',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'nature_reserve', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'wildlife_photography', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['sweatshirt', 'hoodie', 'windbreaker', 'fleece_jacket', 'jeans'],
                       'incompatible': ['dress', 'blazer', 'mini_skirt', 'trench_coat']},
  'justification': 'Rangers ask hide visitors to wear muted, quiet clothing: high-visibility '
                   'colors and busy prints scare animals away from the observation area.',
  'color': {'compatible': ['green', 'brown', 'beige', 'gray'],
            'incompatible': ['white', 'red', 'orange', 'yellow', 'pink']},
  'pattern': {'compatible': ['solid', 'striped', 'checkered'],
              'incompatible': ['floral', 'polka_dot']},
  'query_seeds': {'explicit': ['Wildlife photography session from a hide tomorrow; I was told '
                               'to wear muted, low-key clothing. What should I wear?',
                               'Spending the day in a bird hide — the ranger said no bright '
                               'colors or loud prints. What should I wear?'],
                  'implicit': ['The photography guide asked us to dress low-key for the hide '
                               'session tomorrow. Outfit advice?',
                               "Day in the wildlife hide — the ranger's notes say to blend in. "
                               'What should I wear?']}},
 {'scenario_id': 'stage_backstage_crew',
  'archetype': 'stage_media',
  'name': 'Concert Backstage Crew Shift',
  'tpo': {'time': {'time_of_day': 'evening'},
          'place': {'venue': 'concert_hall', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'stage_crew_work', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'sweatshirt', 'hoodie', 'jeans', 'slacks'],
                       'incompatible': ['blazer', 'formal_shirt', 'dress', 'trench_coat', 'mini_skirt']},
  'justification': 'Stage crews wear all black ("show blacks") to stay invisible to the '
                   'audience during scene changes, in practical clothes for hauling equipment.',
  'color': {'compatible': ['black'],
            'incompatible': ['white', 'yellow', 'pink', 'red', 'orange']},
  'pattern': {'compatible': ['solid'],
              'incompatible': ['leopard', 'floral', 'polka_dot']},
  'query_seeds': {'explicit': ['Working backstage crew at a concert tonight; we are required '
                               'to wear all black. What should I wear?',
                               'Stagehand shift at the theater — show blacks required. What '
                               'should I wear?'],
                  'implicit': ['Helping run backstage at a live show tonight. What should I '
                               'wear?',
                               'Crew call at the concert hall this evening. Outfit advice?']}},
 {'scenario_id': 'stage_greenscreen_shoot',
  'archetype': 'stage_media',
  'name': 'Chroma-Key Studio Shoot (Green & Blue Stages)',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'studio', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'video_shoot', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'sweatshirt', 'sweater', 'jeans', 'leggings'],
                       'incompatible': ['trench_coat', 'windbreaker']},
  'justification': 'The studio keys on both green and blue stages, so clothing in either chroma '
                   'color is removed in compositing; fine repeating patterns cause moire flicker '
                   'on camera, and shiny, flappy outerwear disrupts the keying.',
  'color': {'compatible': ['red', 'pink', 'black', 'gray'],
            'incompatible': ['green', 'blue']},
  'pattern': {'compatible': ['solid'],
              'incompatible': ['striped', 'checkered', 'polka_dot']},
  'query_seeds': {'explicit': ['Filming in front of a green screen tomorrow; the studio said '
                               'no green, no blue, and no fine patterns. What should I wear?',
                               'Green-screen shoot for a video — told to avoid the chroma '
                               'colors and busy prints. What should I wear?'],
                  'implicit': ["The studio sent wardrobe notes for tomorrow's chroma-key "
                               'shoot: nothing that interferes with the key. Outfit advice?',
                               'Shooting on the green-screen stage tomorrow; wardrobe rules '
                               'apply. What should I wear?']}},
 {'scenario_id': 'stage_tv_interview',
  'archetype': 'stage_media',
  'name': 'Live TV News Interview',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'tv_studio', 'indoor_outdoor': 'indoor'},
          'occasion': {'activity': 'media_appearance', 'formality_required': 'business_casual'}},
  'garment_category': {'compatible': ['blazer', 'formal_shirt', 'dress', 'slacks'],
                       'incompatible': ['hoodie', 'tank_top', 't_shirt', 'shorts']},
  'justification': 'On-air guests dress professionally; studios advise against green (chroma '
                   'risk), pure white (camera bloom), and fine patterns (moire) on camera.',
  'color': {'compatible': ['navy', 'gray', 'blue', 'beige', 'purple'],
            'incompatible': ['green', 'white']},
  'pattern': {'compatible': ['solid'],
              'incompatible': ['striped', 'checkered', 'polka_dot']},
  'query_seeds': {'explicit': ['I am being interviewed on live TV news; smart, camera-friendly '
                               'clothes expected. What should I wear?',
                               'On-air TV interview tomorrow — professional attire that works '
                               'on camera. What should I wear?'],
                  'implicit': ['A news channel is interviewing me on air tomorrow. Outfit '
                               'advice?',
                               'Going on live television for an interview this week. What '
                               'should I wear?']}},
 {'scenario_id': 'club_tennis_whites',
  'archetype': 'club_code',
  'name': 'Club Tennis Tournament (All-White Rule)',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'tennis_club', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'tennis_match', 'formality_required': 'very_casual'}},
  'garment_category': {'compatible': ['t_shirt', 'shorts', 'mini_skirt'],
                       'incompatible': ['jeans', 'leather_jacket', 'trench_coat', 'slacks',
                                        'blazer', 'hoodie']},
  'justification': 'Traditional tennis clubs enforce a Wimbledon-style all-white rule for '
                   'players; colored or patterned clothing violates the club code.',
  'color': {'compatible': ['white'],
            'incompatible': ['black', 'navy', 'gray', 'brown', 'red', 'blue', 'green', 'pink',
                             'orange', 'yellow', 'purple', 'beige']},
  'pattern': {'compatible': ['solid'],
              'incompatible': ['striped', 'checkered', 'floral', 'polka_dot', 'leopard']},
  'query_seeds': {'explicit': ["Playing in my club's tennis tournament; the all-white dress "
                               'rule applies. What should I wear?',
                               'Club championship tennis match — whites-only rule like '
                               'Wimbledon. What should I wear?'],
                  'implicit': ['Playing a match at the traditional tennis club on Saturday. '
                               'What should I wear?',
                               "Entered the members' tournament at our old tennis club. Outfit "
                               'advice?']}},
 {'scenario_id': 'club_yacht_regatta',
  'archetype': 'club_code',
  'name': 'Yacht Club Regatta Day',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'marina', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'regatta', 'formality_required': 'smart_casual'}},
  'garment_category': {'compatible': ['formal_shirt', 'windbreaker', 'slacks', 'shorts', 't_shirt'],
                       'incompatible': ['hoodie', 'leather_jacket', 'sweatshirt', 'mini_skirt']},
  'justification': 'Yacht club regattas follow a traditional nautical dress code: crisp marine '
                   'colors and tidy sailing wear; loud colors and streetwear are out of place.',
  'color': {'compatible': ['navy', 'white', 'beige'],
            'incompatible': ['orange', 'yellow']},
  'pattern': {'compatible': ['solid', 'striped'],
              'incompatible': ['leopard', 'floral', 'polka_dot']},
  'query_seeds': {'explicit': ['Attending our yacht club regatta day; classic nautical dress '
                               'expected. What should I wear?',
                               'Regatta at the sailing club — traditional marine style '
                               'expected. What should I wear?'],
                  'implicit': ['Spending Saturday at the yacht club regatta. Outfit advice?',
                               'Invited to watch the sailing club races this weekend. What '
                               'should I wear?']}},
 {'scenario_id': 'club_golf_round',
  'archetype': 'club_code',
  'name': 'Country Club Golf Round',
  'tpo': {'time': {'time_of_day': 'daytime'},
          'place': {'venue': 'golf_course', 'indoor_outdoor': 'outdoor'},
          'occasion': {'activity': 'golf', 'formality_required': 'smart_casual'}},
  'garment_category': {'compatible': ['formal_shirt', 'sweater', 'slacks', 'long_skirt', 'shorts'],
                       'incompatible': ['jeans', 'tank_top', 'hoodie', 't_shirt']},
  'justification': 'Traditional country clubs require collared shirts and tailored bottoms on '
                   'the course; denim, athletic streetwear, and loud prints violate the code.',
  'color': None,
  'pattern': {'compatible': ['solid', 'striped', 'checkered'],
              'incompatible': ['leopard', 'polka_dot']},
  'query_seeds': {'explicit': ['Playing a round at a traditional country club; collared shirts '
                               'required and no denim. What should I wear?',
                               'Golf at a members-only club — classic golf dress code '
                               'enforced. What should I wear?'],
                  'implicit': ['Tee time at the old country club tomorrow morning. What should '
                               'I wear?',
                               'First round of golf at a private club with a friend. Outfit '
                               'advice?']}}]


def get_constrained_axes(scenario):
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
