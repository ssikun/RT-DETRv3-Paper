#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
python_bin=${RTDETR_PYTHON:-/root/rivermind-data/venvs/paddle262-cu118-3090/bin/python}
site_packages=$("$python_bin" -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$site_packages/nvidia/cudnn/lib:$site_packages/nvidia/cublas/lib:$site_packages/nvidia/cuda_runtime/lib:$site_packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export FLAGS_use_pinned_memory=false
export FLAGS_allocator_strategy=auto_growth

"$python_bin" tools/check_p2_query_detail.py --device gpu
mkdir -p logs
set +e
timeout 180s "$python_bin" -u tools/train.py \
    -c configs/rtdetrv3/rtdetrv3_r18vd_visdrone_p2_query_detail.yml \
    --seed 3407 --amp \
    -o pretrain_weights=/root/rivermind-fs/rtdetrv3_r18vd_6x.pdparams \
       save_dir=output/p2_query_detail_3090_smoke \
       epoch=1 \
       TrainDataset.dataset_dir=dataset/visdrone2019 \
       TrainDataset.anno_path=annotations/train.json \
       TrainReader.batch_size=1 \
       TrainReader.worker_num=0 \
       TrainReader.use_shared_memory=False \
    2>&1 | tee logs/p2_query_detail_3090_smoke.log
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -ne 0 && "$status" -ne 124 ]]; then
    exit "$status"
fi
printf 'Smoke exit status: %s (124 means the planned 180-second timeout).\n' "$status"
