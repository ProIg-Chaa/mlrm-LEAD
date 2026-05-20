# Experiment 2-A Bad Event Attribution

- source: `output/experiments/20260516_183300/exp1_vstar_spike_type_parallel`
- spike rule: `H_t > local_mean(16) + 2.0 * local_std(16)`, min_entropy=`1.0`
- high-confidence wrong: tail-20 mean top1 >= `0.8` or any top1 >= `0.95`
- long output threshold: output_tokens >= `256`

## Sample-Level Summary
| method | samples | correct | wrong | long_output | high_conf_wrong | mean_len_correct | mean_len_wrong | mean_spikes_correct | mean_spikes_wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cot | 191 | 131 | 60 | 8 | 60 | 103.7 | 144.1 | 5.3 | 8.1 |
| lead | 191 | 139 | 52 | 6 | 52 | 109.8 | 156.6 | 5.6 | 7.6 |
| pure_soft | 191 | 112 | 79 | 33 | 79 | 113.9 | 412.2 | 5.5 | 9.8 |

## cot Event Attribution
| event | samples | output_tokens | spikes | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 131 | n=131 mean=103.72 med=85.00 p90=182.00 max=296.00 | 700 | 102 (14.6%) | 12 (1.7%) | 34 (4.9%) | 39 (5.6%) | 199 (28.4%) | 314 (44.9%) |
| wrong | 60 | n=60 mean=144.12 med=143.00 p90=218.00 max=311.00 | 488 | 37 (7.6%) | 28 (5.7%) | 24 (4.9%) | 16 (3.3%) | 147 (30.1%) | 236 (48.4%) |
| high_conf_wrong | 60 | n=60 mean=144.12 med=143.00 p90=218.00 max=311.00 | 488 | 37 (7.6%) | 28 (5.7%) | 24 (4.9%) | 16 (3.3%) | 147 (30.1%) | 236 (48.4%) |
| long_output | 8 | n=8 mean=286.12 med=283.00 p90=310.00 max=311.00 | 132 | 4 (3.0%) | 6 (4.5%) | 6 (4.5%) | 2 (1.5%) | 50 (37.9%) | 64 (48.5%) |
| all | 191 | n=191 mean=116.41 med=96.00 p90=198.00 max=311.00 | 1188 | 139 (11.7%) | 40 (3.4%) | 58 (4.9%) | 55 (4.6%) | 346 (29.1%) | 550 (46.3%) |

## lead Event Attribution
| event | samples | output_tokens | spikes | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 139 | n=139 mean=109.79 med=92.00 p90=199.00 max=317.00 | 772 | 118 (15.3%) | 24 (3.1%) | 40 (5.2%) | 36 (4.7%) | 203 (26.3%) | 351 (45.5%) |
| wrong | 52 | n=52 mean=156.62 med=139.50 p90=237.00 max=1024.00 | 394 | 33 (8.4%) | 20 (5.1%) | 18 (4.6%) | 13 (3.3%) | 129 (32.7%) | 181 (45.9%) |
| high_conf_wrong | 52 | n=52 mean=156.62 med=139.50 p90=237.00 max=1024.00 | 394 | 33 (8.4%) | 20 (5.1%) | 18 (4.6%) | 13 (3.3%) | 129 (32.7%) | 181 (45.9%) |
| long_output | 6 | n=6 mean=412.67 med=295.00 p90=317.00 max=1024.00 | 87 | 7 (8.0%) | 4 (4.6%) | 3 (3.4%) | 3 (3.4%) | 32 (36.8%) | 38 (43.7%) |
| has_soft_steps | 191 | n=191 mean=122.54 med=100.00 p90=215.00 max=1024.00 | 1166 | 151 (13.0%) | 44 (3.8%) | 58 (5.0%) | 49 (4.2%) | 332 (28.5%) | 532 (45.6%) |
| all | 191 | n=191 mean=122.54 med=100.00 p90=215.00 max=1024.00 | 1166 | 151 (13.0%) | 44 (3.8%) | 58 (5.0%) | 49 (4.2%) | 332 (28.5%) | 532 (45.6%) |

## lead Soft-Neighborhood Spikes
- soft window: `±8` generated tokens
| visual | relation | format | answer | diffuse_low_conf | other |
|---:|---:|---:|---:|---:|---:|
| 49 (20.4%) | 6 (2.5%) | 3 (1.2%) | 28 (11.7%) | 55 (22.9%) | 99 (41.2%) |

## pure_soft Event Attribution
| event | samples | output_tokens | spikes | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 112 | n=112 mean=113.90 med=74.50 p90=176.00 max=1024.00 | 615 | 70 (11.4%) | 26 (4.2%) | 34 (5.5%) | 33 (5.4%) | 185 (30.1%) | 267 (43.4%) |
| wrong | 79 | n=79 mean=412.23 med=143.00 p90=1024.00 max=1024.00 | 775 | 96 (12.4%) | 19 (2.5%) | 46 (5.9%) | 17 (2.2%) | 294 (37.9%) | 303 (39.1%) |
| high_conf_wrong | 79 | n=79 mean=412.23 med=143.00 p90=1024.00 max=1024.00 | 775 | 96 (12.4%) | 19 (2.5%) | 46 (5.9%) | 17 (2.2%) | 294 (37.9%) | 303 (39.1%) |
| long_output | 33 | n=33 mean=915.52 med=1024.00 p90=1024.00 max=1024.00 | 496 | 59 (11.9%) | 7 (1.4%) | 25 (5.0%) | 8 (1.6%) | 201 (40.5%) | 196 (39.5%) |
| all | 191 | n=191 mean=237.29 med=93.00 p90=1023.00 max=1024.00 | 1390 | 166 (11.9%) | 45 (3.2%) | 80 (5.8%) | 50 (3.6%) | 479 (34.5%) | 570 (41.0%) |

