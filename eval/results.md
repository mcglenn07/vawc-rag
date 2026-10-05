# Evaluation results

Generated 2026-10-05 01:32 UTC by `eval/run_eval.py`. Percentages show raw counts. With about 45 scored questions, one question is about 2 points, so differences under about 5 points are treated as ties.

Test set: 52 questions, 52 marked verified by the owner.

## Retrieval

| Run | Embedding | Hit@3 | Hit@5 | Hit@20 | MRR | nDCG@5 | Precision@5 | Both sources sent |
|---|---|---|---|---|---|---|---|---|
| `20261003-081538-sw400-ov0-hp` | text-embedding-3-large | 91% (41/45) | 93% (42/45) | 96% (43/45) | 0.86 | 0.88 | 0.24 | 92% (12/13) |
| `20261003-102647-sw400-ov0-hp` | text-embedding-3-small | 89% (40/45) | 93% (42/45) | 96% (43/45) | 0.84 | 0.86 | 0.24 | 92% (12/13) |
| `20261003-102656-sw250-ov0-hp` | text-embedding-3-large | 91% (41/45) | 96% (43/45) | 96% (43/45) | 0.86 | 0.88 | 0.24 | 92% (12/13) |
| `20261003-102704-sw250-ov1-hp` | text-embedding-3-large | 91% (41/45) | 93% (42/45) | 96% (43/45) | 0.86 | 0.88 | 0.24 | 92% (12/13) |
| `20261003-102714-sw400-ov0-nohp` | text-embedding-3-large | 82% (37/45) | 87% (39/45) | 93% (42/45) | 0.80 | 0.80 | 0.22 | 85% (11/13) |

Hit@K: all expected sections in the top K (pairs not counted). MRR and nDCG use the ranking without crosswalk pairs. Both sources sent: for `both` questions, law and IRR section both in what the model receives (top 5 plus crosswalk pairs).

## Answers

| Run | Judge | Accuracy | Completeness | Relevance | Refused not_in_docs | False refusals | Clean citations |
|---|---|---|---|---|---|---|---|
| `20261003-081538-sw400-ov0-hp` | gpt-4.1 judging gpt-4.1-mini | 4.84 | 4.80 | 4.93 | 100% (7/7) | 2% (1/45) | 100% (52/52) |

Scores are 1 to 5 from the judge model against the reference answers. Clean citations: answers where every cited section was among the retrieved sections.

## Missed at top 5: `20261003-081538-sw400-ov0-hp` (3)

| Type | Question | Expected | Retrieved top 5 |
|---|---|---|---|
| irr_only | What kind of assistance can the Department of Health provide to women and children who are victims of violence? | IRR, Rule IX, Section 61 | IRR, Rule VI, Section 40; RA 9262, Section 40; RA 9262, Section 35; IRR, Rule VI, Section 39; RA 9262, Section 31 |
| both | How long do I have to file a criminal case for violence against women and their children? | RA 9262, Section 24; IRR, Rule III, Section 9 | RA 9262, Section 25; IRR, Rule III, Section 6; RA 9262, Section 6; RA 9262, Section 7; IRR, Rule III, Section 7 |
| follow_up | Can my ex-boyfriend be covered by this? | RA 9262, Section 3 | RA 9262, Section 14; IRR, Rule IV, Section 13; IRR, Rule IV, Section 11; RA 9262, Section 8; RA 9262, Section 20 |

## Missed at top 5: `20261003-102647-sw400-ov0-hp` (3)

| Type | Question | Expected | Retrieved top 5 |
|---|---|---|---|
| both | How long do I have to file a criminal case for violence against women and their children? | RA 9262, Section 24; IRR, Rule III, Section 9 | RA 9262, Section 6; RA 9262, Section 5; IRR, Rule III, Section 6; IRR, Rule III, Section 7; RA 9262, Section 25 |
| follow_up | Where do they file it? | RA 9262, Section 10 | RA 9262, Section 9; IRR, Rule IV, Section 21; IRR, Rule IV, Section 20; RA 9262, Section 11; IRR, Rule IV, Section 12 |
| follow_up | Can my ex-boyfriend be covered by this? | RA 9262, Section 3 | RA 9262, Section 8; RA 9262, Section 33; IRR, Rule IV, Section 11; IRR, Rule IV, Section 13; IRR, Rule IV, Section 27 |

## Missed at top 5: `20261003-102656-sw250-ov0-hp` (2)

| Type | Question | Expected | Retrieved top 5 |
|---|---|---|---|
| both | How long do I have to file a criminal case for violence against women and their children? | RA 9262, Section 24; IRR, Rule III, Section 9 | RA 9262, Section 25; IRR, Rule III, Section 6; RA 9262, Section 7; RA 9262, Section 5; RA 9262, Section 6 |
| follow_up | Can my ex-boyfriend be covered by this? | RA 9262, Section 3 | RA 9262, Section 14; IRR, Rule IV, Section 13; RA 9262, Section 8; IRR, Rule IV, Section 11; RA 9262, Section 20 |

## Missed at top 5: `20261003-102704-sw250-ov1-hp` (3)

| Type | Question | Expected | Retrieved top 5 |
|---|---|---|---|
| irr_only | What kind of assistance can the Department of Health provide to women and children who are victims of violence? | IRR, Rule IX, Section 61 | IRR, Rule VI, Section 40; RA 9262, Section 40; RA 9262, Section 35; IRR, Rule VI, Section 39; RA 9262, Section 31 |
| both | How long do I have to file a criminal case for violence against women and their children? | RA 9262, Section 24; IRR, Rule III, Section 9 | RA 9262, Section 25; IRR, Rule III, Section 6; RA 9262, Section 7; RA 9262, Section 5; RA 9262, Section 6 |
| follow_up | Can my ex-boyfriend be covered by this? | RA 9262, Section 3 | RA 9262, Section 14; IRR, Rule IV, Section 13; RA 9262, Section 8; IRR, Rule IV, Section 11; RA 9262, Section 20 |

## Missed at top 5: `20261003-102714-sw400-ov0-nohp` (6)

| Type | Question | Expected | Retrieved top 5 |
|---|---|---|---|
| law_only | Where should I file an application for a protection order under this law? | RA 9262, Section 10 | RA 9262, Section 11; IRR, Rule IV, Section 20; IRR, Rule IV, Section 15; IRR, Rule IV, Section 21; RA 9262, Section 9 |
| both | Are temporary and permanent protection orders issued under this law enforceable throughout the Philippines? | RA 9262, Section 12; IRR, Rule IV, Section 22 | RA 9262, Section 8; IRR, Rule IV, Section 18; RA 9262, Section 16; IRR, Rule IV, Section 11; IRR, Rule IV, Section 29 |
| both | How long do I have to file a criminal case for violence against women and their children? | RA 9262, Section 24; IRR, Rule III, Section 9 | IRR, Rule III, Section 6; RA 9262, Section 25; RA 9262, Section 7; RA 9262, Section 6; RA 9262, Section 44 |
| follow_up | How long is it effective? | RA 9262, Section 15 | RA 9262, Section 8; IRR, Rule IV, Section 11; IRR, Rule IV, Section 18; RA 9262, Section 22; RA 9262, Section 16 |
| follow_up | Where do they file it? | RA 9262, Section 10 | RA 9262, Section 9; IRR, Rule IV, Section 21; RA 9262, Section 7; IRR, Rule IV, Section 15; RA 9262, Section 11 |
| follow_up | Can my ex-boyfriend be covered by this? | RA 9262, Section 3 | RA 9262, Section 14; IRR, Rule IV, Section 13; IRR, Rule IV, Section 11; IRR, Rule IV, Section 29; RA 9262, Section 8 |
