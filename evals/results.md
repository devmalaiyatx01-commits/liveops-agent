# Scenario evals

6 scenarios x 5 seeds, 20,000 simulated players each. Reproduce with `python -m liveops eval --policy <name> --seeds 5`.

## Policy: rules

Decision accuracy: **29/30**. False ships: **0**. Missed ships: 1. Guardrail rejections: 0. Fail-closed runs: 0.

| Scenario | Expected | Decisions (by seed) | Correct | Avg tool calls | Avg cost (USD) |
|---|---|---|---:|---:|---:|
| easier_early_levels | ship | ship, ship, ship, hold, ship | 4/5 | 7.0 | 0.000 |
| reward_chest | hold | hold, hold, hold, hold, hold | 5/5 | 7.0 | 0.000 |
| more_free_lives | hold | hold, hold, hold, hold, hold | 5/5 | 7.0 | 0.000 |
| new_onboarding | investigate | investigate, investigate, investigate, investigate, investigate | 5/5 | 3.0 | 0.000 |
| harder_levels | hold | hold, hold, hold, hold, hold | 5/5 | 7.0 | 0.000 |
| new_renderer | hold | hold, hold, hold, hold, hold | 5/5 | 7.0 | 0.000 |

Confusion (expected -> decided): hold->hold: 20, investigate->investigate: 5, ship->hold: 1, ship->ship: 4


## Policy: naive

Decision accuracy: **24/30**. False ships: **0**. Missed ships: 1. Guardrail rejections: 10. Fail-closed runs: 0.

| Scenario | Expected | Decisions (by seed) | Correct | Avg tool calls | Avg cost (USD) |
|---|---|---|---:|---:|---:|
| easier_early_levels | ship | ship, ship, ship, hold, ship | 4/5 | 3.0 | 0.000 |
| reward_chest | hold | hold, hold, hold, hold, hold | 5/5 | 3.0 | 0.000 |
| more_free_lives | hold | hold, hold, hold, hold, hold | 5/5 | 4.0 | 0.000 |
| new_onboarding | investigate | hold, hold, hold, hold, hold | 0/5 | 3.4 | 0.000 |
| harder_levels | hold | hold, hold, hold, hold, hold | 5/5 | 3.0 | 0.000 |
| new_renderer | hold | hold, hold, hold, hold, hold | 5/5 | 3.6 | 0.000 |

Confusion (expected -> decided): hold->hold: 20, investigate->hold: 5, ship->hold: 1, ship->ship: 4

