# Temporal-model experiment results

All three prespecified temporal architectures completed. None produced a feasible validation policy. Diagnostic maximum-sensitivity candidates below violate the false-warning budget.

| Model | Epochs | Warning interval | Episodes warned | Unsuccessful warnings/hour | Feasible thresholds |
|---|---:|---|---:|---:|---:|
| gru_6s_h5 | 15 | 2-5 s | 6/38 | 78.3 | 0 |
| tcn_6s_h3 | 16 | 2-3 s | 4/38 | 63.0 | 0 |
| tcn_6s_h5 | 11 | 2-5 s | 9/38 | 145.9 | 0 |

All three saved models passed reload and real-stream probability parity checks. Validation people were S07/S08. No historical S09/S10 evaluation occurred.

The ideal-score diagnostic (which deliberately uses future labels and is NOT a deployable predictor) warns all 38 validation episodes without false starts at both horizons. This demonstrates that the current replay policy can support the task on this validation set when supplied ideal scores. It is not a model result or a mathematical upper bound.

TCNs preserve the final temporal state through causal residual convolutions; the GRU preserves sequence order. Six-second history and 2-5-second warning variants changed the available context and task window, so their values are not directly equivalent to the earlier 2-3-second benchmark.

The PowerShell training wrapper returned status 1 despite a completion message and completed artifacts for all three runs. Each run saved reload and stream verification results; no Python traceback appears in the captured console. The pooled streaming verification was subsequently launched through a subprocess wrapper and returned Python exit status 0.
