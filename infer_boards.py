"""Tiled YOLO inference on full PKU-Market-PCB boards from the board-disjoint val split + val metrics."""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from ultralytics import YOLO

PCB, TILES = Path(sys.argv[1]), Path(sys.argv[2])
model = YOLO("weights/best.pt")
names = model.names
m = model.val(data=str(TILES / "data_fast.yaml"), imgsz=640, device="cpu", plots=False, verbose=False, batch=16)
per = {names[int(c)]: dict(AP50=float(m.box.ap50[i]), P=float(m.box.p[i]), R=float(m.box.r[i])) for i, c in enumerate(m.box.ap_class_index)}
json.dump(dict(mAP50=float(m.box.map50), mAP50_95=float(m.box.map), precision=float(m.box.mp), recall=float(m.box.mr), per_class=per),
          open("figures/val_metrics.json", "w"), indent=2)
print(json.dumps(per, indent=1), m.box.map50)

def predict_board(img, conf=0.3):
    H, W = img.shape[:2]; boxes, scores, cls = [], [], []
    ys = [round(i * (H - 640) / (-(-H // 640) - 1)) for i in range(-(-H // 640))]
    xs = [round(i * (W - 640) / (-(-W // 640) - 1)) for i in range(-(-W // 640))]
    for y0 in ys:
        for x0 in xs:
            r = model.predict(img[y0:y0 + 640, x0:x0 + 640], imgsz=640, conf=conf, device="cpu", verbose=False)[0]
            for b, s, c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy()):
                boxes.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]); scores.append(float(s)); cls.append(int(c))
    if not boxes: return [], [], []
    keep = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()
    return [boxes[i] for i in keep], [scores[i] for i in keep], [cls[i] for i in keep]

val = [Path(x.strip()).stem for x in (PCB / "ImageSets" / "val_board.txt").read_text().split()]
pick = {}
for s in val:                      # one board per defect class
    pick.setdefault(s.split("_")[0] + "_" + s.split("_")[1] + "_" + s.split("_")[2] if s.count("_") > 2 else s, s)
chosen = [v for v in pick.values()][:6]
COL = {0: (0, 165, 255), 1: (255, 0, 255), 2: (0, 0, 255), 3: (255, 120, 0), 4: (0, 200, 0), 5: (200, 200, 0)}
summary = []
for stem in chosen:
    img = cv2.imread(str(PCB / "images" / f"{stem}.jpg")); H, W = img.shape[:2]
    gt = [list(map(float, l.split())) for l in (PCB / "labels" / f"{stem}.txt").read_text().splitlines()]
    pb, ps, pc = predict_board(img)
    out = img.copy()
    for c, cx, cy, w, h in gt:     # ground truth: thin white box
        cv2.rectangle(out, (int((cx - w / 2) * W) - 6, int((cy - h / 2) * H) - 6), (int((cx + w / 2) * W) + 6, int((cy + h / 2) * H) + 6), (255, 255, 255), 2)
    for b, s, c in zip(pb, ps, pc):
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COL[c], 4)
        cv2.putText(out, f"{names[c]} {s:.2f}", (int(b[0]), max(int(b[1]) - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, COL[c], 4)
    status = "NG" if pb else "OK"
    cv2.rectangle(out, (0, 0), (900, 110), (0, 0, 160) if pb else (0, 140, 0), -1)
    cv2.putText(out, f"{len(pb)} DEFECTS . {status}  (GT {len(gt)})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 5)
    cv2.imwrite(f"figures/board_{stem}.jpg", cv2.resize(out, (1800, int(1800 * H / W))), [cv2.IMWRITE_JPEG_QUALITY, 88])
    summary.append(dict(board=stem, gt=len(gt), pred=len(pb)))
    print(summary[-1])
json.dump(summary, open("figures/board_demo.json", "w"), indent=1)
