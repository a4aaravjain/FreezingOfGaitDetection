"""Summarize completed non-neural experiments without changing fitted policies."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
 out=ROOT/'runs/non_neural_round_1'
 results=json.loads((out/'results.json').read_text())
 lines=['# Non-neural prediction experiments','',
 'Both alternatives use six seconds of past ankle data and the same requirement: a warning 2-5 seconds before annotated onset. Daphnet S01-S06 train the reference/model; S07/S08 select thresholds. These remain development results on historically exposed people.','',
 'Extra Trees uses 300 randomized trees and recent, previous and change features. The signal-change rule uses a training-distribution scale but no fitted FoG classifier. Neither method requires a neural network.','',
 '| Method | Highest-coverage diagnostic candidate | Unsuccessful warnings/hour | Selected policy |','|---|---:|---:|---|']
 for r in results:
  b=r['diagnostic_highest_sensitivity_candidate']
  lines.append(f"| {r['name']} | {b['true_warning_events']}/{b['total_fog_episodes']} | {b['false_alarms_per_hour']:.1f} | {'disabled' if r['threshold']>1 else r['threshold']} |")
 lines+=['','Diagnostic candidates in this table maximize validation episode sensitivity within the fixed threshold grid; they are not recommended operating points. The success criterion remains >=90% episode coverage with <=1 unsuccessful warning/hour. Late warnings count as unsuccessful.', '',
 '37 scientific regression tests passed. The new feature test checks sensitivity to temporal ordering, batch independence, finite results and rejection of incomplete histories. The tree model also passed a saved-model reload check.', '',
 'This comparison tests whether a different model family helps; it cannot prove which architecture is universally best or demonstrate future clinical performance. The earlier GRU experiments already tested recurrent alternatives to convolutional networks. No reserved or historical-test model evaluation was added.', '',
 'Reproduce with non_neural_experiments.py (existing run folders are protected), test_pipeline.py and summarize_non_neural.py. Exact training sources and dataset hashes are saved under runs/non_neural_round_1.']
 (ROOT/'NON_NEURAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines[:12]))
if __name__=='__main__':main()
