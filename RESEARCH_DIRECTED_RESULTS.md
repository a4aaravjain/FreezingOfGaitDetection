# Research-directed experiments - 2026-11-11

No candidate in this round improved the verified v2 research model. Coverage remains 7/16; nine episodes remain missed. Results are from reused development people, not an independent test.

## Sources and adaptations

[Fu, Lin and Zhang (2025), dynamic mode decomposition](https://doi.org/10.1038/s41598-025-88110-4) motivates measuring local motion dynamics. Its method uses person-specific thresholds, and its warning analysis uses the index at annotated transition times. Its discussion notes difficulty with stopping events. Our adaptation extracts delay-embedded mode/amplitude features only from the trailing observed three seconds and fits cross-person supervised classifiers; it does not reproduce their personalized method or reported results.

[PhysioGPN](https://pubmed.ncbi.nlm.nih.gov/40030782/) explicitly targets prediction at least two seconds before onset and motivates comparison of gait self-similarity. We tested a shared GRU encoder on the earlier and later halves of the observed window. This is a simplified single-ankle adaptation of a concept from the abstract, not a reproduction of the full architecture or its performance.

## Results

| Experiment | Combined early warnings /16 | Scored false starts | Unknown boundary alerts |
|---|---:|---:|---:|
| dmd_prediction_round_1: logistic | 7 | 0 | 1 |
| dmd_prediction_round_1: boosting | 7 | 0 | 1 |
| twin_gait_prediction_round_1: shared_gru_encoder | 7 | 0 | 1 |
| domain_agreement_round_1: median_three | 7 | 0 | 2 |
| domain_agreement_round_1: minimum_dense_extended | 7 | 0 | 1 |
| domain_agreement_round_1: geometric_three | 7 | 0 | 1 |

Two DMD classifiers, one shared-encoder model, and three agreement policies were evaluated. Some standalone heads were disabled for failing policy constraints. Combined counts include the retained v2 predictor; they are not new successes attributable to a disabled head. The median-agreement policy increased unknown alerts and was rejected. All other candidates lacked episode or lead improvement.

All three trained models passed save/reload score parity. Three DMD regression tests passed: stationary signals, isolation of one window from other windows, and nonzero finite features for dynamic motion. The shared model checkpoint restored the best validation PR-AUC with early stopping.

Train: people 1-14. Development checkpoint/policy selection and comparisons: people 15-18. People 19-22 were not used. The same two-second minimum lead, 2-5-second scoring interval, alert-start matching and false-warning constraints were retained. The published results were not counted as performance achieved by this project.

Source, training logs, candidate grids, predictions, merged replays and decision gates are saved in the three named run directories. New code: dmd_prediction.py, test_dmd_prediction.py and twin_gait_prediction.py. The existing ankles.py ensemble command still loads verified v2. Near-perfect sensitivity and minimal real-world false alarms remain unachieved.
