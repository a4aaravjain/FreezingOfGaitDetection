# Latest research status

Completed 16 additional dense-network fits on 2026-10-16. Tested equal person/episode weighting, feature noise, training duration, fine-tuning copies of the saved original model, and earlier training targets to compensate for alarm delay. None beat the original 6/16 development result. The closest fine-tuned version warned before 5/16 episodes with zero scored false starts.

In a separate within-experiment train/calibration/assessment split, balancing improved the matched baseline from zero to one early warning with zero scored false starts. That episode was already detected by the original candidate. This is a limited experimental gain, not a new best model.

58 regression tests passed; all 16 fitted models passed checkpoint-reload prediction parity. The original model remains unchanged. See [DENSE_IMPROVEMENT_RESULTS.md](DENSE_IMPROVEMENT_RESULTS.md) for all results and the distinction between separate calibration and repeatedly reused development recordings.

The original 6/16 result remains exploratory, with all six successful warnings from one participant. No reliable improvement across people or real-world readiness has been demonstrated.
