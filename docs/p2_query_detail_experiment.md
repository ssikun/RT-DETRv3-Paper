# P2 Query Detail Enhancement Candidate

## Material Passport

- Type: code experiment plan and implementation record
- Status: CPU verified candidate; full training held pending efficacy evidence
- Date: 2026-10-12
- Repository: RT-DETRv3-Paper, exp/p2-enhance
- Evidence: prior VisDrone size, proposal recall, alpha, and gate diagnostics
- Intended outcome: small-object improvement with a smaller large-object cost

## Hypothesis and Evidence

The previous P2 path downsampled stride-4 detail into stride-8 P3. The P2 center
head was training-only. The main detector did not directly sample P2 values.
The 8-16 px bucket received little AP gain, while replacing encoder queries
with P2-ranked candidates reduced proposal recall. These observations motivate
testing direct detail access for existing queries. They do not establish that
resolution is the sole cause of the remaining errors.

## Implementation

Backbone P2 is projected to 64 channels with GroupNorm. P3/P4/P5 retain the
baseline neck and encoder. Existing query selection remains intact. After the
final decoder layer, four-head, four-point single-level deformable attention
samples P2 around each query's current predicted box. A zero-initialized linear
projection adds this detail to the query before its final class/box prediction.

For input-space box size s = sqrt(width * height), the residual gate is
clip((48 - s) / (48 - 16), 0, 1). It is full at <=16 px and zero at >=48 px.
The boxes are detached for this gate and sampling geometry. Thresholds refer
to resized, padded model input, not COCO's original-image area categories.
New normalization and attention run in FP32, including under AMP.

The branch refines queries already selected by the encoder. It cannot ensure
recovery of targets that have no usable query, and an incorrectly large
predicted box can turn off the detail gate. These are limits to test, not
evidence that the desired AP gain will occur.

This final-layer placement prevents a subsequent query self-attention layer
from spreading the added residual to other queries. Shared-parameter training
can still alter all scales. The gate does not guarantee unchanged APL.

The candidate configuration starts from the baseline and enables only this
new detail path. The old gate and training-only center head remain available
in separate configurations for ablation. The new branch exists at inference;
its latency and memory cost must be measured.

The branch adds 49,264 parameters for ResNet-18. CPU checks passed on
2026-10-12, including full-model synthetic training/backward and inference.
An additional real 640x640 validation-image forward with compatible COCO
pretrained parameters had active fraction 0.5733, mean gate 0.4415, zero initial
residual, and finite predictions. This is one-image implementation evidence,
not a dataset accuracy result. At zero output initialization, the first update
trains the output projection; P2 sampling gradients become nonzero once that
projection changes. Reported training gate statistics include all query groups.

## Verification and Experiment Gates

### Decision on 2026-10-12

Do not recommend a 40-epoch run based on CPU correctness checks alone. The
user requires substantial confidence before spending on full training. There
is no calibrated 60-70 percent success probability for this candidate, and
prior-work results do not establish an APS gain of one point for this model.
GPU access was subsequently supplied by the user for bounded testing. Full
40-epoch training remains on hold.

A bounded, frozen-baseline probe is implemented in
`tools/probe_p2_query_detail.py`. It is a separate diagnostic training protocol:

1. Load the verified baseline40 detection weights into the candidate. Verify
   complete original-parameter loading and measure epoch-zero predictions
   against the saved baseline, with the same preprocessing and 300 queries.
2. Freeze every original parameter and lock original BatchNorm running
   statistics. Optimize only `transformer.p2_query_detail` parameters. This
   keeps the original detector fixed while testing learnable P2 detail access.
3. Use the training set for at most five epochs, with a fixed seed and the
   baseline input/augmentation protocol. Evaluate before training and after
   every epoch; keep outputs in a new directory and preserve the source model.
4. A provisional continuation signal is APS >= baseline APS + 0.3 AP points,
   APL >= baseline APL - 0.5 AP points, and no overall AP decline. Report all
   epochs, branch residual magnitude, and inference cost. These thresholds
   screen expenditure; they do not establish the final one-point APS target.
5. If the probe gives no useful gain, lower this candidate's priority and do
   not extend it automatically. A negative frozen-model result does not prove
   that end-to-end training would fail. A positive result does not establish
   a 60-70 percent chance of meeting the paper target either.

The real-data GPU smoke passed on RTX 3090, Paddle 2.6.2 / CUDA 11.8, at
800x800 and batch size two. It verified loading all 571 original tensors,
49,264 trainable new parameters, 75 locked BatchNorm layers, identical initial
predictions, a nonzero new projection gradient, and unchanged original state
after optimizer steps. This establishes implementation behavior, not AP gain.

The five-epoch probe uses seed 3407, batch size one, new-branch learning rate
0.0001 with 200-step warmup, AMP training and FP32 Q300 evaluation. Original
parameters and BatchNorm statistics remain fixed. Evaluation before training
and after every epoch is retained, along with full epoch weights and CSV/JSON
results. The runner has a three-hour hard limit and is configured to shut down
the server after success or failure, as requested by the user. It cannot
automatically continue into full training.

The initial batch-size-two run was stopped at the user's request during its
first epoch, after full validation of the starting model. At that point GPU
occupancy was about 17.3 GB and logged peak allocated tensor memory was about
5.5 GB. These are distinct measurements; neither establishes the worst-case
memory across all training images. The batch-size-one run starts afresh from
the same baseline in `output/p2_query_detail_frozen_probe_20261012_bs1`.

1. Run tools/check_p2_query_detail.py on CPU: zero-init equivalence, correct
   input-space gate, small-query/P2 gradients, zero P2 gradient for gated-out
   queries, preserved encoder candidates, final-layer isolation, and full-model
   forward/backward.
2. On a GPU, run scripts/smoke_visdrone_p2_query_detail_3090.sh for the same
   checks and a 180-second real-data AMP training smoke.
   Check finite loss/gradients, active gate fraction, residual ratio, and memory.
3. Passing smoke establishes runtime compatibility only. Full training also
   requires an efficacy-based decision and a matched baseline protocol; save
   every evaluation and retain final/best checkpoints if it is undertaken.
4. Paper target: delta APS >= +1.0 AP point and delta APL >= -0.5 AP point.
   These are acceptance targets, not predicted outcomes.
5. If APS does not improve or the large-object cost persists, do not add more
   blocks to this candidate. Retain the negative result and reconsider the
   hypothesis. A positive run still needs repeatability and inference-cost data.

## Prior Work and Novelty Boundary

- Deformable DETR supplies the established sparse sampling mechanism:
  https://arxiv.org/abs/2010.04159
- CF-DETR already uses local information to refine object queries:
  https://ojs.aaai.org/index.php/AAAI/article/view/19893
- DRQ-DETR exposes a thin stride-4 value path for decoder access:
  https://github.com/shenguiyu/DRQ-DETR

Direct high-resolution query access is not claimed as new. The present
candidate tests final-layer placement and a query-wise predicted-scale gate
inside RT-DETRv3. A broader literature comparison and ablation are required
before describing this combination as a paper contribution.
