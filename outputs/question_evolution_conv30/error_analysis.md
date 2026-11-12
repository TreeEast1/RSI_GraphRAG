# Error Analysis (evaluation split)

## E0 -> E1

Fixed: **0**; worsened: **1**; still wrong: 24.

### Worsened

- **q38** (temporal) When did Gina go to a dance class with a group of friends?
  - gold: `21 July 2023`
  - E1: `Last Friday` (skills: static_ans_1_temporal_anchor, static_ans_2_temporal_granularity)

## E0 -> E2

Fixed: **1**; worsened: **2**; still wrong: 23.

### Fixed

- **q71** (single-hop) How does Gina describe the studio that Jon has opened?
  - gold: `amazing`
  - E2: `amazing` (skills: ans_use_explicit_caused_state_wording, ans_return_concise_explicit_purpose_for_why_)

### Worsened

- **q18** (multi-hop) Do Jon and Gina start businesses out of what they love?
  - gold: `Yes`
  - E2: `Yes, "I'm passionate about dancing and it'd be great to share it with others", "I'm passionate about fashion trends and finding unique pieces. Plus, I wanted to` (skills: none)
- **q38** (temporal) When did Gina go to a dance class with a group of friends?
  - gold: `21 July 2023`
  - E2: `Last Friday` (skills: ans_preserve_temporal_granularity_for_relati)

## E0 -> E6

Fixed: **1**; worsened: **2**; still wrong: 23.

### Fixed

- **q71** (single-hop) How does Gina describe the studio that Jon has opened?
  - gold: `amazing`
  - E6: `amazing` (skills: ans_use_explicit_caused_state_wording, ans_return_concise_explicit_purpose_for_why_)

### Worsened

- **q18** (multi-hop) Do Jon and Gina start businesses out of what they love?
  - gold: `Yes`
  - E6: `Yes, "I'm starting a dance studio 'cause I'm passionate about dancing and it'd be great to share it with others", "I'm passionate about dance and fashion so com` (skills: none)
- **q38** (temporal) When did Gina go to a dance class with a group of friends?
  - gold: `21 July 2023`
  - E6: `Last Friday` (skills: ans_preserve_temporal_granularity_for_relati)

## E2 -> E6

Fixed: **0**; worsened: **0**; still wrong: 25.

## Skill hit analysis (E2)

- Questions where at least one skill was retrieved: **28**
- Helpful (E0 wrong -> E2 right): **1**
- Harmful (E0 right -> E2 wrong): **1**
- Neutral: **26**
- Wrong without any skill hit: **5**

| skill | uses | helpful | harmful | neutral |
|---|---:|---:|---:|---:|
| ans_use_explicit_caused_state_wording | 18 | 1 | 0 | 17 |
| ans_return_concise_explicit_purpose_for_why_ | 18 | 1 | 0 | 17 |
| ans_preserve_temporal_granularity_for_relati | 10 | 0 | 1 | 9 |

## Skill hit analysis (E6, full system)

Hit/helpful/harmful/neutral/no-hit-wrong: 28 / 1 / 1 / 26 / 5

## Evidence present but the Reader still refused

- E2: **2** questions
  - q9 (multi-hop): Which city have both Jean and John visited? -> `No answer`
  - q67 (single-hop): How does Jon use the clipboard with a notepad attached to it? -> `No answer`
- E6: **2** questions
  - q9 (multi-hop): Which city have both Jean and John visited? -> `No answer`
  - q67 (single-hop): How does Jon use the clipboard with a notepad attached to it? -> `No answer`

## Interpretation

- A skill hit does not mean the skill caused the change; compare the same question under E0.  The per-skill table separates helpful from harmful hits.
- Skills that are retrieved but never change an answer are neutral, not evidence of generalisation.
- Repeated-run instability bounds how much of any single-question flip can be attributed to the skill layer.
