# Comprehension relation quality audit (conv-30)

| source | n | temporal_ordering | both endpoints existing | pred>=5 words | ordering predicate | near-dup of fact | mean conf |
|---|---:|---:|---:|---:|---:|---:|---:|
| COMP-P0 (original C3) | 26 | 17 | 7 (27%) | 10 | 21 | 0 | 0.883 |
| C6 (full-conv original) | 18 | 7 | 1 (6%) | 0 | 10 | 0 | 0.842 |
| COMP-P1 — Evidence-grounded composition | 14 | 0 | 1 (7%) | 0 | 1 | 0 | 0.811 |
| COMP-P2 — Cross-session bridge discovery | 19 | 0 | 1 (5%) | 0 | 4 | 0 | 0.794 |
| COMP-P3 — Event and state tracking | 27 | 0 | 1 (4%) | 0 | 0 | 0 | 0.834 |
| COMP-P4 — Retrieval-oriented comprehension | 16 | 2 | 1 (6%) | 0 | 2 | 0 | 0.861 |
| COMP-P5 — Generate then verify | 7 | 1 | 0 (0%) | 0 | 1 | 0 | 0.879 |

## COMP-P0 (original C3)

- types: {'temporal_ordering': 17, 'causal': 2, 'state_change': 5, 'identity_unification': 1, 'enabling_link': 1}
- derivation: {'Composed': 20, 'Inferred': 6}
- predicate words mean/max: 5.19/12; object >6 words: 1; any existing endpoint: 24
- sessions per relation: 2.15

| subject | predicate | object | type | both-exist | pred-w | near-dup | sessions |
|---|---|---|---|---:|---:|---:|---|
| Jon | lost_job_before | started dream business | temporal_ordering | - | 3 | - | [6, 11] |
| Gina | had_interview_before | accepted_for fashion internship | temporal_ordering | - | 3 | - | [11, 12] |
| Jon | searched_for_location_before | official opening night | temporal_ordering | Y | 4 | - | [3, 15] |
| Jon | worked_on_business_plan_before | worked_on_opening dance studio | temporal_ordering | - | 5 | - | [12, 15] |
| Jon | looked_for_investors_before | official opening night | temporal_ordering | Y | 4 | - | [12, 15] |
| Jon | attended_networking_events_before | went_full-time_with biz idea | temporal_ordering | - | 4 | - | [16, 17] |
| Jon | posted_dance_videos_before | started_learning marketing and analytics tools | temporal_ordering | - | 4 | - | [14, 17] |
| Gina | lost_job_before | opened online clothing store | temporal_ordering | - | 3 | - | [6, 14] |
| Jon | lost_job_enabled | started dream business | causal | - | 3 | - | [9, 11] |
| Gina | lost_job_enabled | started own online clothing store | causal | - | 3 | - | [10, 14] |
| Jon | used_whiteboard_before | color-coded achievements | temporal_ordering | - | 3 | - | [12, 13] |
| Jon | offered_one-on-one_mentoring_before | mentored aspiring dancers | temporal_ordering | - | 4 | - | [13, 17] |
| Jon | visited_Paris_before | took_trip_to Rome | temporal_ordering | - | 3 | - | [2, 15] |
| Gina | launched_ad_campaign_before | emailed wholesalers | temporal_ordering | - | 4 | - | [2, 3] |
| Gina | opened_online_clothes_store_before | acquired new unique pieces | temporal_ordering | - | 5 | - | [6, 16] |
| Jon | attended_fair_before | worked_on_opening dance studio | temporal_ordering | - | 3 | - | [10, 15] |
| Jon | got_possible_leads_before | official opening night | temporal_ordering | Y | 4 | - | [10, 15] |
| Jon's dance studio | expanded_from_startup_to_social_media_and_competition_before_financial_trouble | tenuous grounds and investor search | state_change | - | 11 | - | [1, 8, 18] |
| Gina's clothing store | developed_from_online_store_to_customer_order_website | new website for customers to make orders | state_change | - | 8 | - | [5, 8, 18] |
| nearby festival performance | identity_unified_with | Jon's performance that was awesome | identity_unification | - | 3 | - | [1, 5] |
| Jon's dance studio | later_needed_investors_after_hosting_competition | investors | temporal_ordering | Y | 6 | - | [8, 18] |
| Gina | later_helped_Jon_with_social_media_after_Jon_expanded_social_media_presence | Jon's dance studio social media | enabling_link | - | 12 | - | [8, 18] |
| Jon | continued_business_after_shutting_down_bank_account | dance studio | state_change | Y | 7 | - | [8, 18] |
| Gina's planned influencer work | preceded_Jon's_request_for_marketing_help | marketing strategies | temporal_ordering | - | 6 | - | [8, 18] |
| Jon | moved_from_banker_to_dance_studio_owner_before_temp_job | temp job | state_change | Y | 10 | - | [1, 5, 18] |
| Gina | moved_from_Door_Dash_job_loss_to_clothing_store_business | clothing store | state_change | Y | 10 | - | [1, 8, 18] |

## C6 (full-conv original)

- types: {'temporal_ordering': 7, 'identity_unification': 2, 'causal_link': 5, 'state_change': 4}
- derivation: {'Composed': 9, 'Inferred': 9}
- predicate words mean/max: 2.06/4; object >6 words: 4; any existing endpoint: 5
- sessions per relation: 2.06

| subject | predicate | object | type | both-exist | pred-w | near-dup | sessions |
|---|---|---|---|---:|---:|---:|---|
| Jon's dance studio | opened_after | Jon searched for ideal spot | temporal_ordering | - | 2 | - | [2, 15] |
| Gina's online clothing store | opened_after | Gina launched ad campaign | temporal_ordering | - | 2 | - | [2, 6] |
| Jon's festival performance | same_event_as | nearby festival performance by Jon's dancers | identity_unification | - | 3 | - | [1, 5] |
| Gina's fashion internship | resulted_from | Gina's design internship interview | causal_link | - | 2 | - | [11, 12] |
| Jon's dance studio grand opening | occurred_after | Jon was working on opening a dance studio | temporal_ordering | - | 2 | - | [15, 17] |
| Jon's networking events | enabled | Jon met investors | causal_link | - | 1 | - | [16, 18] |
| Jon's business plan | revised_after | Jon wrapped up business plan | temporal_ordering | - | 2 | - | [12, 18] |
| Jon's search for investors | continued_after | Jon got temp job | temporal_ordering | - | 2 | - | [12, 18] |
| Gina's online clothing store | expanded_after | Gina opened online clothes store | state_change | - | 2 | - | [6, 18] |
| Jon's dance studio social media presence | expanded_after | Gina advised social media channels and influencers | causal_link | - | 2 | - | [8, 14] |
| Jon's dance studio | became_financially_tenuous_after | official opening night | state_change | - | 4 | - | [15, 18] |
| Jon's lost job as banker | enabled | Jon started dream business | causal_link | - | 1 | - | [1, 11] |
| Gina's lost job at Door Dash | enabled | Gina started own online clothing store | causal_link | - | 1 | - | [1, 10, 14] |
| Jon's dance studio | progressed_from | searching for location to running own studio | state_change | - | 2 | - | [3, 5] |
| Gina's clothing store | progressed_from | ad campaign to online store opening | state_change | - | 2 | - | [2, 6] |
| Jon's dance competition hosting | occurred_after | Jon was getting ready for a dance competition | temporal_ordering | - | 2 | - | [4, 8] |
| Jon's whiteboard use | continued_after | Jon read The Lean Startup | temporal_ordering | - | 2 | - | [12, 13] |
| Jon's dance studio | was_promoted_at | fair | identity_unification | Y | 3 | - | [10, 12] |

## COMP-P1 — Evidence-grounded composition

- types: {'causal': 2, 'enabling_link': 4, 'event_dependency': 1, 'identity_unification': 3, 'goal_outcome': 1, 'state_change': 3}
- derivation: {'Composed': 7, 'Inferred': 7}
- predicate words mean/max: 1.14/2; object >6 words: 0; any existing endpoint: 3
- sessions per relation: 2.0

| subject | predicate | object | type | both-exist | pred-w | near-dup | sessions |
|---|---|---|---|---:|---:|---:|---|
| Jon's job loss | pushed | Jon's dance studio business | causal | - | 1 | - | [9, 11] |
| Jon's location search | enabled | Jon's studio opening | enabling_link | - | 1 | - | [2, 15] |
| Jon's business plan | preceded | Jon's investor search | event_dependency | - | 1 | - | [12, 16] |
| Gina's design internship interview | led_to | Gina's fashion internship | identity_unification | - | 2 | - | [11, 12] |
| Gina's online clothes store | became | Gina's own online clothing store | identity_unification | - | 1 | - | [6, 10] |
| Jon's fair attendance | produced | Jon's possible leads | goal_outcome | - | 1 | - | [10, 14] |
| Jon's grand opening | enabled | Jon's full-time biz idea | state_change | - | 1 | - | [15, 17] |
| Jon's whiteboard | supported | Jon's business plan | enabling_link | - | 1 | - | [12, 13] |
| nearby festival performance | became | Jon's awesome performance | identity_unification | - | 1 | - | [1, 5] |
| Jon's dance studio plan | became | running own studio | state_change | - | 1 | - | [1, 5] |
| Jon's dance studio | became | tenuous grounds | state_change | Y | 1 | - | [8, 18] |
| Gina's online store | enabled | new website | enabling_link | - | 1 | - | [8, 18] |
| Jon's social media expansion | enabled | Gina's social media help | enabling_link | - | 1 | - | [8, 18] |
| Jon's banker job loss | led_to | dance studio startup | causal | - | 2 | - | [1, 5] |

## COMP-P2 — Cross-session bridge discovery

- types: {'plan_outcome': 5, 'event_consequence': 1, 'state_change': 6, 'causal': 4, 'event_sequence': 3}
- derivation: {'Composed': 12, 'Inferred': 7}
- predicate words mean/max: 1.58/3; object >6 words: 0; any existing endpoint: 8
- sessions per relation: 2.05

| subject | predicate | object | type | both-exist | pred-w | near-dup | sessions |
|---|---|---|---|---:|---:|---:|---|
| Jon's dance studio search | led_to | official opening night | plan_outcome | - | 2 | - | [3, 15] |
| Jon's business plan | preceded | official opening night | plan_outcome | - | 1 | - | [12, 15] |
| Jon's fair promotion | generated | possible leads | event_consequence | - | 1 | - | [10, 16] |
| Gina's design internship interview | resulted_in | fashion internship acceptance | plan_outcome | - | 2 | - | [11, 12] |
| Gina's online clothes store | expanded_into | new unique pieces | state_change | - | 2 | - | [6, 16] |
| Jon's lost job | pushed | dream business | causal | - | 1 | - | [9, 11] |
| Jon's lost job | enabled | full-time biz idea | causal | - | 1 | - | [16, 17] |
| Gina's lost Door Dash job | led_to | online clothing store | causal | - | 2 | - | [6, 14] |
| Jon's whiteboard | became | color-coded achievement system | state_change | - | 1 | - | [12, 13] |
| Jon's grand opening | became | welcoming dancers of all ages | state_change | - | 1 | - | [15, 19] |
| Jon's one-on-one mentoring | continued_as | mentored aspiring dancers | event_sequence | - | 2 | - | [13, 17] |
| Jon's dance routine video | preceded | posted dance videos | event_sequence | - | 1 | - | [11, 14] |
| Gina's fashion internship acceptance | preceded | fashion editors reach-out | state_change | - | 1 | - | [12, 17] |
| nearby festival performance | was_completed_as | awesome performance | plan_outcome | - | 3 | - | [1, 5] |
| Jon's dance studio | changed_from | tenuous grounds | state_change | Y | 2 | - | [1, 8, 18] |
| Gina's online store | developed_into | clothing store | state_change | - | 2 | - | [8, 18] |
| Jon's banker job loss | led_to | dance studio startup | causal | - | 2 | - | [1, 5] |
| Gina's Door Dash job loss | preceded | Gina's online store | event_sequence | - | 1 | - | [1, 8] |
| Gina's planned influencer work | aligned_with | social media collaboration | plan_outcome | - | 2 | - | [8, 18] |

## COMP-P3 — Event and state tracking

- types: {'state_transition': 7, 'status_change': 10, 'plan_outcome': 4, 'relationship_change': 2, 'causal': 2, 'goal_change': 2}
- derivation: {'Composed': 25, 'Explicit': 2}
- predicate words mean/max: 1.74/2; object >6 words: 0; any existing endpoint: 8
- sessions per relation: 2.11

| subject | predicate | object | type | both-exist | pred-w | near-dup | sessions |
|---|---|---|---|---:|---:|---:|---|
| Jon's dance studio | changed_from | location search | state_transition | - | 2 | - | [3, 9, 15] |
| Jon's business plan | changed_from | wrapping up | status_change | - | 2 | - | [12, 16, 19] |
| Jon's biz idea | became | full-time venture | status_change | - | 1 | - | [16, 17] |
| Jon's grand opening | changed_from | planned to upcoming | status_change | - | 2 | - | [15, 19] |
| Gina's clothing store | expanded_to | physical space | state_transition | - | 2 | - | [2, 3] |
| Gina's online clothes store | changed_from | dream to open | state_transition | - | 2 | - | [6, 7, 16] |
| Gina's fashion internship | changed_from | interview to accepted | status_change | - | 2 | - | [11, 12] |
| Gina's fashion recognition | changed_from | store growth to editor notice | status_change | - | 2 | - | [14, 17] |
| Jon's promotion | expanded_from | fair to social media | plan_outcome | - | 2 | - | [10, 14] |
| Jon's business prep | expanded_from | plans to investors | plan_outcome | - | 2 | - | [12, 16] |
| Jon's mentoring | expanded_from | young dancers to aspiring dancers | state_transition | - | 2 | - | [13, 17] |
| Jon's support system | expanded_to | business mentor | relationship_change | - | 2 | - | [13, 14] |
| Jon's business skills | expanded_to | marketing and analytics tools | state_transition | - | 2 | - | [12, 17] |
| Jon's confidence | recovered_from | low confidence | state_transition | - | 2 | - | [10, 16] |
| Jon's job loss | led_to | dream business start | causal | - | 2 | - | [9, 11] |
| Gina's job loss | led_to | online clothing store | causal | - | 2 | - | [10, 14] |
| Jon | changed_from | banker to business starter | state_transition | - | 2 | - | [1, 5] |
| Jon's dance studio | became | running own studio | status_change | Y | 1 | - | [1, 5] |
| nearby festival performance | became | completed awesome performance | plan_outcome | - | 1 | - | [1, 5] |
| Gina's online store | expanded_to | new website for orders | status_change | - | 2 | - | [5, 18] |
| Jon's dance studio | expanded_to | social media and workshops | status_change | - | 2 | - | [1, 8] |
| Jon's dance studio | became | on tenuous grounds | status_change | - | 1 | - | [8, 18] |
| Jon | added | temp job and investor search | plan_outcome | - | 1 | - | [1, 18] |
| Jon and Gina | became | marketing collaborators | relationship_change | - | 1 | - | [8, 18] |
| Jon's marketing plan | expanded_to | online platform and social media | goal_change | - | 2 | - | [8, 18] |
| Gina's business focus | changed_from | customer base to website orders | goal_change | - | 2 | - | [8, 18] |
| Jon's bank account | became | shut down for biz | status_change | - | 1 | - | [1, 8] |

## COMP-P4 — Retrieval-oriented comprehension

- types: {'causal': 4, 'temporal_ordering': 2, 'enabling_link': 3, 'goal_outcome': 2, 'identity_unification': 1, 'state_change': 4}
- derivation: {'Composed': 14, 'Inferred': 2}
- predicate words mean/max: 2.19/4; object >6 words: 0; any existing endpoint: 10
- sessions per relation: 2.44

| subject | predicate | object | type | both-exist | pred-w | near-dup | sessions |
|---|---|---|---|---:|---:|---:|---|
| Jon's job loss | pushed_start_of | Jon's dance studio | causal | - | 3 | - | [9, 11, 17] |
| Jon's dance studio | opened_after | Jon's location search | temporal_ordering | - | 2 | - | [3, 15, 19] |
| Gina's support | sustained_motivation_for | Jon's dance studio | enabling_link | - | 3 | - | [7, 16, 19] |
| Jon's fair attendance | generated_leads_for | Jon's dance studio | goal_outcome | - | 3 | - | [10, 14] |
| Gina's marketing advice | led_to | Jon's dance videos | causal | - | 2 | - | [14, 17] |
| Gina's fashion internship | followed | Gina's design internship interview | temporal_ordering | - | 1 | - | [11, 12] |
| Gina's online clothing store | blended | dance and fashion | identity_unification | - | 1 | - | [6, 9, 14] |
| Jon's whiteboard | supported_tracking_of | Jon's business milestones | enabling_link | - | 3 | - | [12, 13, 16] |
| Jon's dance studio | became_space_for | dancer self-expression | state_change | - | 3 | - | [11, 13, 19] |
| Jon's dance studio | expanded_into | one-on-one mentoring | state_change | - | 2 | - | [13, 17, 19] |
| Jon's banker job loss | enabled_start_of | Jon's dance studio | causal | - | 3 | - | [1, 5] |
| nearby festival performance | reaffirmed | Jon's dancing passion | causal | - | 1 | - | [1, 5] |
| Jon's dance studio | shifted_from_progress_to | tenuous grounds | state_change | Y | 4 | - | [8, 18] |
| Gina's clothing store | expanded_into | new website | state_change | - | 2 | - | [8, 18] |
| Gina's social media help | extended | dance studio's social media presence | enabling_link | - | 1 | - | [8, 18] |
| Gina's fashion bloggers plan | anticipated | Instagram and TikTok strategy | goal_outcome | - | 1 | - | [8, 18] |

## COMP-P5 — Generate then verify

- types: {'causal': 3, 'enabling_link': 1, 'goal_outcome': 2, 'temporal_ordering': 1}
- derivation: {'Composed': 6, 'Inferred': 1}
- predicate words mean/max: 2.14/3; object >6 words: 0; any existing endpoint: 5
- sessions per relation: 2.14

| subject | predicate | object | type | both-exist | pred-w | near-dup | sessions |
|---|---|---|---|---:|---:|---:|---|
| Jon losing job | pushed_start_of | dance studio business | causal | - | 3 | - | [9, 11, 17] |
| whiteboard | enabled_tracking_of | ideas and milestones | enabling_link | - | 3 | - | [12, 13] |
| written goals | kept_motivated | Jon | causal | - | 2 | - | [13, 16] |
| Jon's fair attendance | generated | possible leads | goal_outcome | - | 1 | - | [10, 14] |
| Gina's design internship interview | preceded | fashion internship acceptance | temporal_ordering | - | 1 | - | [11, 12] |
| Jon's one-on-one mentoring | supports_goal_of | dancers' full potential | goal_outcome | - | 3 | - | [13, 17] |
| Jon's banker job loss | enabled_starting | dance studio | causal | - | 2 | - | [1, 5] |
