# VisDrone Baseline Commands

Train with console log saved:

```bash
mkdir -p logs

export OMP_NUM_THREADS=1
export FLAGS_use_pinned_memory=false
export FLAGS_allocator_strategy=auto_growth

python -u tools/train.py \
  -c configs/rtdetrv3/rtdetrv3_r18vd_visdrone_baseline.yml \
  -o pretrain_weights=weights/rtdetrv3_r18vd_6x.pdparams \
     TrainReader.batch_size=2 \
     EvalReader.batch_size=1 \
     TrainReader.worker_num=0 \
     EvalReader.worker_num=0 \
     TrainReader.use_shared_memory=False \
     EvalReader.use_shared_memory=False \
  --eval --amp 2>&1 | tee logs/visdrone_baseline.log
```

Evaluate the best checkpoint:

```bash
python tools/eval.py \
  -c configs/rtdetrv3/rtdetrv3_r18vd_visdrone_baseline.yml \
  -o weights=output/rtdetrv3_r18vd_visdrone_baseline/best_model/model.pdparams
```

Metrics are saved to:

```text
output/rtdetrv3_r18vd_visdrone_baseline/eval/bbox_metrics.txt
```
