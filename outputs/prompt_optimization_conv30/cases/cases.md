# Case studies

## q16 (cat2)

**Q**: When did Jon go to a fair to get more exposure for his dance studio?  
**Gold**: 24 April, 2023  
**Gold sessions**: [10] turns ['D10:1']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s05, s09, s11, s02, s18 | [9] | 0.0 | False | 5 |
| C2 | s18, s05, s01, s02, s17 | [8] | 0.0 | True | 0 |
| cmpo_compp4 | s10, s14, s18, s05, s01 | [1] | 1.0 | False | 1 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: 'No answer' (EM 0, F1 0.00)  
**P0 on this graph (E2)**: '24 April, 2023'  
**P7 on this graph (E3)**: '24 April, 2023'

Relations touching gold sessions:
- `Jon's fair attendance --generated_leads_for--> Jon's dance studio` [goal_outcome] sessions=[10, 14] conf=0.86

## q17 (cat1)

**Q**: Why did Gina decide to start her own clothing store?  
**Gold**: She always loved fashion trends and finding unique pieces and she lost her job so decided it was time to start her own business.  
**Gold sessions**: [1, 6] turns ['D6:8', 'D1:3']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s08, s05, s06, s03, s18 | [12, 3] | 0.0 | True | 0 |
| C2 | s10, s02, s03, s08, s05 | [11, 6] | 0.0 | False | 3 |
| cmpo_compp4 | s10, s02, s03, s08, s05 | [11, 6] | 0.0 | False | 3 |

**P0 on G_raw (E0)**: 'After losing my job, I wanted to take control of my own destiny' (EM 0, F1 0.16)  
**P7 on G_raw (E1)**: "After losing my job, I wanted to take control of my own destiny, I've been one 'cause I lost my job" (EM 0, F1 0.18)  
**P0 on this graph (E2)**: 'After losing her job, she wanted to take control of her own destiny'  
**P7 on this graph (E3)**: "After losing my job, I wanted to take control of my own destiny, I'm passionate about fashion trends and finding unique pieces, I wanted to blend my love for dance and fashion"

Relations touching gold sessions:
- `Gina's online clothing store --blended--> dance and fashion` [identity_unification] sessions=[6, 9, 14] conf=0.84
- `Jon's banker job loss --enabled_start_of--> Jon's dance studio` [causal] sessions=[1, 5] conf=0.86
- `nearby festival performance --reaffirmed--> Jon's dancing passion` [causal] sessions=[1, 5] conf=0.91

## q24 (cat1)

**Q**: Which events has Jon participated in to promote his business venture?  
**Gold**: fair, networking events, dance competition  
**Gold sessions**: [8, 10, 16] turns ['D10:1', 'D16:6', 'D8:4']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s16, s18, s08, s04, s14 | [3, 7, 1] | 1.0 | False | 3 |
| C2 | s16, s18, s08, s04, s14 | [3, 7, 1] | 1.0 | False | 3 |
| cmpo_compp4 | s16, s18, s08, s14, s04 | [3, 7, 1] | 1.0 | False | 3 |

**P0 on G_raw (E0)**: 'a fair, networking events, a dance competition' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: 'networking events, a fair, the last networking event' (EM 0, F1 0.55)  
**P0 on this graph (E2)**: 'a fair, networking events, a dance competition'  
**P7 on this graph (E3)**: 'networking events, a fair, the last networking event'

Relations touching gold sessions:
- `Gina's support --sustained_motivation_for--> Jon's dance studio` [enabling_link] sessions=[7, 16, 19] conf=0.88
- `Jon's fair attendance --generated_leads_for--> Jon's dance studio` [goal_outcome] sessions=[10, 14] conf=0.86
- `Jon's whiteboard --supported_tracking_of--> Jon's business milestones` [enabling_link] sessions=[12, 13, 16] conf=0.91
- `Jon's dance studio --shifted_from_progress_to--> tenuous grounds` [state_change] sessions=[8, 18] conf=0.89
- `Gina's clothing store --expanded_into--> new website` [state_change] sessions=[8, 18] conf=0.84
- `Gina's social media help --extended--> dance studio's social media presence` [enabling_link] sessions=[8, 18] conf=0.82
- `Gina's fashion bloggers plan --anticipated--> Instagram and TikTok strategy` [goal_outcome] sessions=[8, 18] conf=0.63

## q29 (cat1)

**Q**: Which cities has Jon visited?  
**Gold**: Paris, Rome  
**Gold sessions**: [2, 15] turns ['D2:4', 'D15:1']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s02, s18, s15, s17, s04 | [1, 3] | 1.0 | False | 2 |
| C2 | s02, s18, s15, s17, s04 | [1, 3] | 1.0 | False | 2 |
| cmpo_compp4 | s02, s18, s15, s17, s04 | [1, 3] | 1.0 | False | 2 |

**P0 on G_raw (E0)**: 'Paris, Rome' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: 'Paris, Rome' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'Paris, Rome'  
**P7 on this graph (E3)**: 'Paris, Rome'

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9

## q31 (cat1)

**Q**: How long did it take for Jon to open his studio?  
**Gold**: six months  
**Gold sessions**: [1, 15] turns ['D1:2', 'D15:13']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s05, s10, s11, s15, s09 | [11, 4] | 0.0 | False | 5 |
| C2 | s05, s10, s11, s15, s09 | [11, 4] | 0.0 | False | 5 |
| cmpo_compp4 | s05, s10, s11, s15, s09 | [9, 4] | 0.0 | False | 5 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '4 months and 18 days' (EM 0, F1 0.29)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: '2 months and 10 days'

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9
- `Jon's banker job loss --enabled_start_of--> Jon's dance studio` [causal] sessions=[1, 5] conf=0.86
- `nearby festival performance --reaffirmed--> Jon's dancing passion` [causal] sessions=[1, 5] conf=0.91

## q33 (cat2)

**Q**: When did Jon visit networking events for his store?  
**Gold**: 20 June, 2023  
**Gold sessions**: [16] turns ['D16:6']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s16, s18, s08, s10, s01 | [1] | 1.0 | False | 4 |
| C2 | s18, s16, s08, s10, s01 | [2] | 1.0 | False | 1 |
| cmpo_compp4 | s18, s16, s08, s10, s01 | [2] | 1.0 | False | 1 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '21 June, 2023' (EM 0, F1 0.67)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: '20 June, 2023'

Relations touching gold sessions:
- `Gina's support --sustained_motivation_for--> Jon's dance studio` [enabling_link] sessions=[7, 16, 19] conf=0.88
- `Jon's whiteboard --supported_tracking_of--> Jon's business milestones` [enabling_link] sessions=[12, 13, 16] conf=0.91

## q54 (cat4)

**Q**: How is Gina's store doing?  
**Gold**: The store is doing great.  
**Gold sessions**: [4] turns ['D4:2']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s04, s08, s05, s03, s06 | [1] | 1.0 | False | 1 |
| C2 | s08, s05, s06, s03, s02 | [9] | 0.0 | True | 0 |
| cmpo_compp4 | s08, s05, s06, s03, s02 | [9] | 0.0 | True | 0 |

**P0 on G_raw (E0)**: 'doing great' (EM 0, F1 0.67)  
**P7 on G_raw (E1)**: "The store's doing great!" (EM 0, F1 0.57)  
**P0 on this graph (E2)**: "so far, so good - customers love the new offers and promotions, which means I'm seeing more sales"  
**P7 on this graph (E3)**: "So far, so good - customers love the new offers and promotions, which means I'm seeing more sales, People seem to really like my designs, Growing my customer base is the main focus right now"

## q4 (cat4)

**Q**: Why did Jon decide to start his dance studio?  
**Gold**: He lost his job and decided to start his own business to share his passion.  
**Gold sessions**: [1] turns ['D1:2', 'D1:4']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s13, s09, s11, s05, s02 | [7] | 0.0 | False | 5 |
| C2 | s13, s11, s05, s01, s09 | [4] | 1.0 | False | 4 |
| cmpo_compp4 | s13, s11, s05, s01, s09 | [4] | 1.0 | False | 4 |

**P0 on G_raw (E0)**: 'Losing my job gave me the push to finally start my dream business: my own dance studio' (EM 0, F1 0.32)  
**P7 on G_raw (E1)**: "Losing my job gave me the push to finally start my dream business: my own dance studio, Losing my job was tough but it gave me the push I needed to do what I love, I want to create a place for people to dance and express themselves - it's been a dream of mine, I'm turning my loves of dance into a business, I'm aiming to turn my dancing passion into a business" (EM 0, F1 0.20)  
**P0 on this graph (E2)**: "he's passionate about dancing and it'd be great to share it with others"  
**P7 on this graph (E3)**: "'cause I'm passionate about dancing and it'd be great to share it with others, I've been into dancing since I was a kid and it's been my passion and escape, so I can teach others the joy that dancing brings me, Losing my job gave me the push to finally start my dream business: my own dance studio!, Losing my job was tough but it gave me the push I needed to do what I love, I want to create a place for people to dance and express themselves - it's been a dream of mine, I want it to be a place of support and encouragement for all our dancers"

Relations touching gold sessions:
- `Jon's banker job loss --enabled_start_of--> Jon's dance studio` [causal] sessions=[1, 5] conf=0.86
- `nearby festival performance --reaffirmed--> Jon's dancing passion` [causal] sessions=[1, 5] conf=0.91

## q7 (cat2)

**Q**: When did Gina launch an ad campaign for her store?  
**Gold**: 29 January, 2023  
**Gold sessions**: [2] turns ['D2:1']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s02, s08, s05, s16, s18 | [1] | 1.0 | False | 3 |
| C2 | s02, s08, s05, s18, s03 | [1] | 1.0 | False | 3 |
| cmpo_compp4 | s02, s08, s05, s18, s03 | [1] | 1.0 | False | 3 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '29 January, 2023' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: '29 January, 2023'

## q13 (cat2)

**Q**: When did Gina open her online clothing store?  
**Gold**: 16 March, 2023  
**Gold sessions**: [6] turns ['D6:6']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s06, s10, s14, s07, s08 | [1] | 1.0 | False | 5 |
| C2 | s10, s06, s07, s14, s08 | [2] | 1.0 | False | 5 |
| cmpo_compp4 | s10, s06, s14, s07, s08 | [2] | 1.0 | False | 5 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '16 March, 2023' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: '16 March, 2023'

Relations touching gold sessions:
- `Gina's online clothing store --blended--> dance and fashion` [identity_unification] sessions=[6, 9, 14] conf=0.84

## q19 (cat2)

**Q**: When did Gina interview for a design internship?  
**Gold**: 10 May, 2023  
**Gold sessions**: [11] turns ['D11:14']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s11, s12, s16, s05, s03 | [1] | 1.0 | False | 1 |
| C2 | s11, s12, s16, s05, s03 | [1] | 1.0 | False | 2 |
| cmpo_compp4 | s11, s12, s16, s05, s03 | [1] | 1.0 | False | 3 |

**P0 on G_raw (E0)**: 'yesterday' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '10 May, 2023' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'yesterday'  
**P7 on this graph (E3)**: '10 May, 2023'

Relations touching gold sessions:
- `Jon's job loss --pushed_start_of--> Jon's dance studio` [causal] sessions=[9, 11, 17] conf=0.96
- `Gina's fashion internship --followed--> Gina's design internship interview` [temporal_ordering] sessions=[11, 12] conf=0.82
- `Jon's dance studio --became_space_for--> dancer self-expression` [state_change] sessions=[11, 13, 19] conf=0.89

## q30 (cat2)

**Q**: When Jon is planning to open his dance studio?  
**Gold**: 20 June, 2023  
**Gold sessions**: [15] turns ['D15:5']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s11, s15, s03, s01, s02 | [2] | 1.0 | False | 4 |
| C2 | s11, s15, s01, s03, s02 | [2] | 1.0 | False | 4 |
| cmpo_compp4 | s11, s15, s01, s02, s05 | [2] | 1.0 | False | 4 |

**P0 on G_raw (E0)**: 'tomorrow' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '20 June, 2023' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: '20 June, 2023'  
**P7 on this graph (E3)**: '20 June, 2023'

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9

## q38 (cat2)

**Q**: When did Gina go to a dance class with a group of friends?  
**Gold**: 21 July 2023  
**Gold sessions**: [19] turns ['D19:6']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s19, s01, s05, s17, s02 | [1] | 1.0 | False | 1 |
| C2 | s19, s01, s05, s17, s02 | [1] | 1.0 | False | 2 |
| cmpo_compp4 | s19, s01, s05, s17, s02 | [1] | 1.0 | False | 2 |

**P0 on G_raw (E0)**: 'Last Friday' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '21 July, 2023' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'Last Friday'  
**P7 on this graph (E3)**: '21 July, 2023'

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9
- `Gina's support --sustained_motivation_for--> Jon's dance studio` [enabling_link] sessions=[7, 16, 19] conf=0.88
- `Jon's dance studio --became_space_for--> dancer self-expression` [state_change] sessions=[11, 13, 19] conf=0.89
- `Jon's dance studio --expanded_into--> one-on-one mentoring` [state_change] sessions=[13, 17, 19] conf=0.87

## q39 (cat4)

**Q**: What is Gina's favorite style of dance?  
**Gold**: Contemporary  
**Gold sessions**: [1] turns ['D1:9']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s01, s05, s02, s17, s04 | [1] | 1.0 | False | 1 |
| C2 | s01, s05, s02, s17, s08 | [1] | 1.0 | False | 1 |
| cmpo_compp4 | s01, s05, s02, s17, s08 | [1] | 1.0 | False | 1 |

**P0 on G_raw (E0)**: 'contemporary dance' (EM 0, F1 0.67)  
**P7 on G_raw (E1)**: 'contemporary' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'contemporary'  
**P7 on this graph (E3)**: 'contemporary'

Relations touching gold sessions:
- `Jon's banker job loss --enabled_start_of--> Jon's dance studio` [causal] sessions=[1, 5] conf=0.86
- `nearby festival performance --reaffirmed--> Jon's dancing passion` [causal] sessions=[1, 5] conf=0.91

## q71 (cat4)

**Q**: How does Gina describe the studio that Jon has opened?  
**Gold**: amazing  
**Gold sessions**: [15] turns ['D15:6']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s11, s19, s15, s03, s05 | [3] | 1.0 | False | 4 |
| C2 | s11, s19, s15, s03, s05 | [3] | 1.0 | False | 4 |
| cmpo_compp4 | s11, s19, s15, s03, s05 | [3] | 1.0 | False | 4 |

**P0 on G_raw (E0)**: 'amazing' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: 'The studio looks amazing' (EM 0, F1 0.50)  
**P0 on this graph (E2)**: 'amazing'  
**P7 on this graph (E3)**: 'The studio looks amazing'

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9

## q72 (cat4)

**Q**: How does Jon feel about the opening night of his dance studio?  
**Gold**: excited  
**Gold sessions**: [15] turns ['D15:7']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s05, s11, s15, s09, s03 | [3] | 1.0 | False | 4 |
| C2 | s05, s11, s15, s09, s01 | [3] | 1.0 | False | 4 |
| cmpo_compp4 | s05, s11, s15, s09, s01 | [3] | 1.0 | False | 4 |

**P0 on G_raw (E0)**: 'excited' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: "I'm excited!, feeling good and ready to give it my best, Tomorrow's gonna be an awesome night, Can't wait for it" (EM 0, F1 0.10)  
**P0 on this graph (E2)**: 'excited'  
**P7 on this graph (E3)**: "I'm excited!, feeling good and ready to give it my best, Tomorrow's gonna be an awesome night, Woohoo! Tomorrow's opening will be so much fun. Can't wait for it - and for you to be there!"

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9

## q78 (cat4)

**Q**: According to Gina, what makes Jon a perfect mentor and guide?  
**Gold**: His positivity and determination  
**Gold sessions**: [17] turns ['D17:7']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s07, s17, s14, s13, s05 | [2] | 1.0 | False | 4 |
| C2 | s17, s14, s05, s13, s04 | [1] | 1.0 | False | 2 |
| cmpo_compp4 | s17, s14, s05, s13, s04 | [1] | 1.0 | False | 2 |

**P0 on G_raw (E0)**: 'His positivity and determination' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: 'Loving what you do and bringing joy to others, Your positivity and determination' (EM 0, F1 0.35)  
**P0 on this graph (E2)**: 'Your positivity and determination'  
**P7 on this graph (E3)**: 'Loving what you do and bringing joy to others is so rewarding, Your positivity and determination will make your dance studio a hit'

Relations touching gold sessions:
- `Jon's job loss --pushed_start_of--> Jon's dance studio` [causal] sessions=[9, 11, 17] conf=0.96
- `Gina's marketing advice --led_to--> Jon's dance videos` [causal] sessions=[14, 17] conf=0.9
- `Jon's dance studio --expanded_into--> one-on-one mentoring` [state_change] sessions=[13, 17, 19] conf=0.87

## q85 (cat5)

**Q**: What is Gina's attitude towards participating in the dance festival?  
**Gold**: None  
**Gold sessions**: [1] turns ['D1:28']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s11, s01, s05, s08, s04 | [2] | 1.0 | False | 4 |
| C2 | s01, s11, s04, s05, s08 | [1] | 1.0 | False | 4 |
| cmpo_compp4 | s01, s11, s04, s05, s08 | [1] | 1.0 | False | 4 |

**P0 on G_raw (E0)**: 'No answer' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: "Wow, they look great! Can't wait to see them rock the festival. Gonna be awesome!" (EM 0, F1 0.00)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: "Can't wait to see them rock the festival. Gonna be awesome!"

Relations touching gold sessions:
- `Jon's banker job loss --enabled_start_of--> Jon's dance studio` [causal] sessions=[1, 5] conf=0.86
- `nearby festival performance --reaffirmed--> Jon's dancing passion` [causal] sessions=[1, 5] conf=0.91

## q63 (cat4)

**Q**: What kind of professional experience did Gina get accepted for on May 23, 2023?  
**Gold**: fashion internship  
**Gold sessions**: [12] turns ['D12:1']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s12, s16, s02, s03, s17 | [1] | 1.0 | False | 3 |
| C2 | s12, s02, s16, s03, s17 | [1] | 1.0 | False | 3 |
| cmpo_compp4 | s12, s11, s02, s16, s17 | [1] | 1.0 | False | 3 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: 'No answer' (EM 0, F1 0.00)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: 'No answer'

Relations touching gold sessions:
- `Gina's fashion internship --followed--> Gina's design internship interview` [temporal_ordering] sessions=[11, 12] conf=0.82
- `Jon's whiteboard --supported_tracking_of--> Jon's business milestones` [enabling_link] sessions=[12, 13, 16] conf=0.91

## q67 (cat4)

**Q**: How does Jon use the clipboard with a notepad attached to it?  
**Gold**: To set goals, track achievements, and find areas for improvement  
**Gold sessions**: [13] turns ['D13:11']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s13, s12, s17, s04, s19 | [1] | 1.0 | True | 0 |
| C2 | s13, s12, s17, s04, s19 | [1] | 1.0 | True | 0 |
| cmpo_compp4 | s13, s12, s17, s04, s19 | [1] | 1.0 | True | 0 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: 'No answer' (EM 0, F1 0.00)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: 'No answer'

Relations touching gold sessions:
- `Jon's whiteboard --supported_tracking_of--> Jon's business milestones` [enabling_link] sessions=[12, 13, 16] conf=0.91
- `Jon's dance studio --became_space_for--> dancer self-expression` [state_change] sessions=[11, 13, 19] conf=0.89
- `Jon's dance studio --expanded_into--> one-on-one mentoring` [state_change] sessions=[13, 17, 19] conf=0.87

