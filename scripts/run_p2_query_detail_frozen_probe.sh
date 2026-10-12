#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
python_bin=${RTDETR_PYTHON:-/root/rivermind-data/venvs/paddle262-cu118-3090/bin/python}
run_id=${RTDETR_PROBE_ID:-20261012}
output_dir="output/p2_query_detail_frozen_probe_${run_id}"
status_file="logs/p2_query_detail_frozen_probe_${run_id}.status"
mkdir -p logs

finish() {
    status=$?
    trap - EXIT
    printf 'exit_code=%s\nfinished_utc=%s\noutput_dir=%s\n' \
        "$status" "$(date -u +%FT%TZ)" "$output_dir" > "$status_file"
    sync
    printf 'Probe finished with exit code %s; automatic shutdown in 60 seconds.\n' "$status"
    sleep 60
    shutdown -h now
}
trap finish EXIT

site_packages=$("$python_bin" -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$site_packages/nvidia/cudnn/lib:$site_packages/nvidia/cublas/lib:$site_packages/nvidia/cuda_runtime/lib:$site_packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export FLAGS_use_pinned_memory=false
export FLAGS_allocator_strategy=auto_growth

# The timeout bounds the paid screening run; it never extends into full training.
timeout --signal=TERM --kill-after=60s 10800s "$python_bin" -u \
    tools/probe_p2_query_detail.py \
    -c configs/rtdetrv3/rtdetrv3_r18vd_visdrone_p2_query_detail.yml \
    --baseline eval_weights/baseline40_model_final.pdparams \
    --output-dir "$output_dir" \
    --epochs 5 --seed 3407 --batch-size 2 --lr 0.0001 --device gpu --amp \
    -o TrainDataset.dataset_dir=dataset/visdrone2019 \
       TrainDataset.anno_path=annotations/train.json \
       EvalDataset.dataset_dir=dataset/visdrone2019 \
       EvalDataset.anno_path=annotations/val.json
