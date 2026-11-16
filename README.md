# PCB defect detection + SPC on real data

Real-data demo for "Precision Electronic Defect Detection & SPC Engine".

- **Detection**: YOLOv8n on the PKU-Market-PCB dataset (693 real boards, 6 defect types), board-disjoint
  train/val split. Boards (~3000x1600 px, defects ~70 px) are cut into 640 px tiles; inference is tiled + NMS.
- **SPC**: EWMA / individuals charts and a weekly p-chart on the UCI SECOM semiconductor process dataset
  (1,567 runs x 590 sensors, real pass/fail labels).

## Run
```bash
pip install ultralytics pandas matplotlib torch
# data: https://huggingface.co/datasets/KeenForgeAI/PKU-Market-PCB-corrected  -> <pcb_root>
python prepare_tiles.py <pcb_root> <tiles_root> 0.1
python train.py                      # edit paths; 6 epochs on 4 CPU cores ~1.2 h
python infer_boards.py <pcb_root> <tiles_root>
# data/ : UCI SECOM (https://archive.ics.uci.edu/dataset/179/secom)
python spc_secom.py
```

## Results (see `figures/`)
- `val_metrics.json`: on 986 val tiles (all 786 tiles with defects + 200 background tiles, board-disjoint),
  mAP@0.5 = 0.947, mAP@0.5:0.95 = 0.47, precision 0.919, recall 0.943. Not a full-board evaluation.
- `board_*.jpg`: full-board predictions (white = ground truth).
- `spc_control_chart.png`, `spc_weekly_fail_rate.png`: SECOM sensor #59 (most correlated with fail label) shows a
  real persistent level shift that the EWMA chart flags. The SECOM sensors are anonymous and the data is not PCB images,
  so detection and SPC are shown on separate real datasets, not linked.
- Only 6 epochs, yolov8n, CPU: a baseline, not tuned.
