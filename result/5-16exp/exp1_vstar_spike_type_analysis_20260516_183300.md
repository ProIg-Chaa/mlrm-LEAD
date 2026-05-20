# Experiment 1 Spike Type Analysis

- spike rule: `H_t > local_mean(16) + 2.0 * local_std(16)`, min_history=`4`, min_entropy=`1.0`
- topk field: `raw_topk`

## Overall
| method | samples | tokens | spikes | missing_topk | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cot | 191 | 22426 | 1188 | 0 | 139 (11.7%) | 40 (3.4%) | 58 (4.9%) | 55 (4.6%) | 346 (29.1%) | 550 (46.3%) |
| lead | 191 | 23596 | 1166 | 0 | 151 (13.0%) | 44 (3.8%) | 58 (5.0%) | 49 (4.2%) | 332 (28.5%) | 532 (45.6%) |
| pure_soft | 191 | 45643 | 1390 | 0 | 166 (11.9%) | 45 (3.2%) | 80 (5.8%) | 50 (3.6%) | 479 (34.5%) | 570 (41.0%) |

## cot By Correctness
| group | spikes | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 700 | 102 (14.6%) | 12 (1.7%) | 34 (4.9%) | 39 (5.6%) | 199 (28.4%) | 314 (44.9%) |
| wrong | 488 | 37 (7.6%) | 28 (5.7%) | 24 (4.9%) | 16 (3.3%) | 147 (30.1%) | 236 (48.4%) |

## cot By Generation Mode
| mode | spikes | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | 1188 | 139 (11.7%) | 40 (3.4%) | 58 (4.9%) | 55 (4.6%) | 346 (29.1%) | 550 (46.3%) |

## lead By Correctness
| group | spikes | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 772 | 118 (15.3%) | 24 (3.1%) | 40 (5.2%) | 36 (4.7%) | 203 (26.3%) | 351 (45.5%) |
| wrong | 394 | 33 (8.4%) | 20 (5.1%) | 18 (4.6%) | 13 (3.3%) | 129 (32.7%) | 181 (45.9%) |

## lead By Generation Mode
| mode | spikes | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | 1155 | 149 (12.9%) | 41 (3.5%) | 57 (4.9%) | 48 (4.2%) | 332 (28.7%) | 528 (45.7%) |
| soft | 11 | 2 (18.2%) | 3 (27.3%) | 1 (9.1%) | 1 (9.1%) | 0 (0.0%) | 4 (36.4%) |

## pure_soft By Correctness
| group | spikes | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 615 | 70 (11.4%) | 26 (4.2%) | 34 (5.5%) | 33 (5.4%) | 185 (30.1%) | 267 (43.4%) |
| wrong | 775 | 96 (12.4%) | 19 (2.5%) | 46 (5.9%) | 17 (2.2%) | 294 (37.9%) | 303 (39.1%) |

## pure_soft By Generation Mode
| mode | spikes | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_soft | 1390 | 166 (11.9%) | 45 (3.2%) | 80 (5.8%) | 50 (3.6%) | 479 (34.5%) | 570 (41.0%) |

