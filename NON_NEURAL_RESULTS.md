# Non-neural prediction experiments

Both alternatives use six seconds of past ankle data and the same requirement: a warning 2-5 seconds before annotated onset. Daphnet S01-S06 train the reference/model; S07/S08 select thresholds. These remain development results on historically exposed people.

Extra Trees uses 300 randomized trees and recent, previous and change features. The signal-change rule uses a training-distribution scale but no fitted FoG classifier. Neither method requires a neural network.

| Method | Highest-coverage diagnostic candidate | Unsuccessful warnings/hour | Selected policy |
|---|---:|---:|---|
| extra_trees | 6/38 | 58.4 | disabled |
| signal_change_rule | 8/38 | 170.5 | disabled |

Diagnostic candidates in this table maximize validation episode sensitivity within the fixed threshold grid; they are not recommended operating points. The success criterion remains >=90% episode coverage with <=1 unsuccessful warning/hour. Late warnings count as unsuccessful.

37 scientific regression tests passed. The new feature test checks sensitivity to temporal ordering, batch independence, finite results and rejection of incomplete histories. The tree model also passed a saved-model reload check.

This comparison tests whether a different model family helps; it cannot prove which architecture is universally best or demonstrate future clinical performance. The earlier GRU experiments already tested recurrent alternatives to convolutional networks. No reserved or historical-test model evaluation was added.

Reproduce with non_neural_experiments.py (existing run folders are protected), test_pipeline.py and summarize_non_neural.py. Exact training sources and dataset hashes are saved under runs/non_neural_round_1.
