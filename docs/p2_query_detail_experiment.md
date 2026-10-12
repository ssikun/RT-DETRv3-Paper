# P2 Query Detail Enhancement Candidate

## Material Passport

- Type: code experiment plan and implementation record
- Status: CPU verified candidate; accuracy and GPU memory results pending
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

1. Run tools/check_p2_query_detail.py on CPU: zero-init equivalence, correct
   input-space gate, small-query/P2 gradients, zero P2 gradient for gated-out
   queries, preserved encoder candidates, final-layer isolation, and full-model
   forward/backward.
2. On a GPU, run scripts/smoke_visdrone_p2_query_detail_3090.sh for the same
   checks and a 180-second real-data AMP training smoke.
   Check finite loss/gradients, active gate fraction, residual ratio, and memory.
3. Only after smoke passes, train the candidate with the chosen baseline
   protocol, save every evaluation and retain final/best checkpoints.
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
