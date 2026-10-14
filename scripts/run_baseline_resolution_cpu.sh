#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
python_bin=${RTDETR_PYTHON:-/root/rivermind-data/venvs/paddle262-cu118-3090/bin/python}
run_dir=results/baseline40_resolution_cpu_20261014
mkdir -p logs "$run_dir"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export FLAGS_use_pinned_memory=false

finish() {
    status=$?
    trap - EXIT
    printf 'exit_code=%s\nfinished_utc=%s\n' "$status" "$(date -u +%FT%TZ)" \
        > "$run_dir/run.status"
    sync
}
trap finish EXIT

for size in 640 960; do
    config=configs/rtdetrv3/rtdetrv3_r18vd_visdrone_baseline.yml
    if [[ "$size" -eq 960 ]]; then
        config=configs/rtdetrv3/rtdetrv3_r18vd_visdrone_baseline_eval960.yml
    fi
    timeout --signal=TERM --kill-after=30s 21600s "$python_bin" -u \
        tools/evaluate_baseline_resolution.py -c "$config" \
        --output-dir "$run_dir/eval_${size}" \
        -o weights=eval_weights/baseline40_model_final.pdparams \
           EvalDataset.dataset_dir=dataset/visdrone2019 \
           EvalDataset.anno_path=annotations/val.json
done

"$python_bin" tools/analyze_coco_small_object_diagnostics.py \
    --annotations dataset/visdrone2019/annotations/val.json \
    --prediction "baseline640=$run_dir/eval_640/bbox.json" \
    --prediction "baseline960=$run_dir/eval_960/bbox.json" \
    --output-dir "$run_dir/size_diagnostics"
