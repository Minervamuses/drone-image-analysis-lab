# Phase 03 — 短試跑的 NAFNet 輸出異常

2026-10-04，使用者確認 `20261004T124643.486598Z` 是 `--limit 4` 短試跑。來源為 `/mnt/c/Users/garyc/Downloads/20261004T124643.486598Z/`，CSV 與圖片保持原樣；完整核對證據見 ../build-log.md。

## 會影響下一步的確認事實

- 三份 CSV 的 4 圖×5 模型關聯、樣本數與逐圖平均一致，24 張 PNG 表頭均為 RGB 5280×3956。
- `outputs/NAFNet-GoPro-width64.pth/0881.JPG.png` 與 `0696.JPG.png` 有大片規則彩色區塊及棋盤紋；對應 inputs 沒有這些區塊。兩張 NAFNet 輸出的 PSNR 約為 8.0624 / 9.1145 dB。已查看的 Uformer/0881 對照圖沒有同樣現象。
- 所有 CSV 列仍為 ok：它只證明保存/數值計算完成，不能作為輸出內容正常的證據。原始結果、數值與模型均保留；不能以清晰度暴增宣稱改善，也不自動剔除模型或改摘要。

## 未確認與交接界線

2026-10-04 後續收到使用者貼回的 console：RTX A6000、torch 2.11.0+cu128、CUDA 12.8；同一短試跑完成 20 次保存/度量，總耗時 2096.7s，沒有記錄推論例外或 OOM。先前缺 console/裝置/耗時的未知已補齊；原始 JPG、checkpoint、實際 lab Git SHA、Spandrel 版本及精度旗標仍未回傳。本機無本次 checkpoint，未執行推論重現。

既有程式採 float32、支援 tiling 時使用 512 core/32 halo；入口未明確控制 TF32，而 `33b3960:lab/run.sh` 曾關閉 matmul/cuDNN TF32。[PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html#tensorfloat-32-tf32-on-nvidia-ampere-and-later-devices) 說明 float32 卷積仍可使用 TF32；[CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices) 提供 torch 2.9 之後的 `fp32_precision` 控制，且不建議混用新舊控制方式。console 未印實際旗標；以上只能支持一個精度對照，不證實異常根因，也不排除 checkpoint/模型的問題。

下一步只針對已異常的 none/0881 做下列對照，再判斷是否需要修改入口；不疊加 tiling、模型實作或依賴修正。原 run 與成績保持原樣，不把短試跑寫成驗收通過，也不自行啟動全量。若需要共用推論、依賴、縮圖裁切或代跑 GPU，沿用 PLANS 的授權/成本界線。現有 phase-03 的「無阻礙後才全量」仍適用，無需改變實驗定義。

## lab 單圖精度對照（尚未執行）

在 `/home/gary/test` 的 Linux shell 執行下列命令。只用已保存的 0881 輸入與現有 NAFNet 權重，兩次推論保持 loader、張量 dtype、尺寸及 tiling 相同，只改 cuDNN 卷積精度。預期產物為新 `evaluation/runs/nafnet_precision_<UTC>/` 的兩張 PNG 與 stdout；不跑 LPIPS/清晰度或其他模型，不改正式 run 的三份 CSV。

NAFNet 原四張模型循環為 244.9s（含度量），平均 61.225s/張僅供成本參考；IEEE FP32 可能較慢，本命令總耗時未實測。執行前 console 印 GPU、VRAM、磁碟與 cgroup 資源；不使用 CPU fallback。若現行卷積已是 IEEE，直接停止，精度假設需重新評估。

```bash
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" PYTHONPATH=src .venv/bin/python -u - <<'PY'
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from drone_sr.image_io import read_image, write_png
from drone_sr.inference import load_model, upscale

run = Path("evaluation/runs/deblur/20261004T124643.486598Z")
source = run / "inputs/0881.JPG.png"
checkpoint = Path("models/deblur/NAFNet-GoPro-width64.pth")
if not source.is_file() or not checkpoint.is_file():
    raise SystemExit("Missing original run input or existing NAFNet checkpoint; no download attempted")
if not torch.cuda.is_available():
    raise SystemExit("CUDA required; CPU fallback disabled")
print("Git:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "spandrel:", version("spandrel"))
print("GPU:", torch.cuda.get_device_name(0), "VRAM free/total bytes:", torch.cuda.mem_get_info())
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"],
      "NVIDIA_TF32_OVERRIDE:", os.environ.get("NVIDIA_TF32_OVERRIDE", "unset"))
print("Disk free bytes:", shutil.disk_usage("evaluation/runs").free)
for name in ("memory.max", "memory.current"):
    path = Path("/sys/fs/cgroup") / name
    if path.is_file():
        print("cgroup", name, path.read_text().strip())
current = torch.backends.cudnn.conv.fp32_precision
print("cuDNN conv:", current, "CUDA matmul:", torch.backends.cuda.matmul.fp32_precision)
if current == "ieee" or os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
    raise SystemExit("TF32 already disabled; stop before repeating the same precision conditions")
descriptor = load_model(checkpoint, role="deblur")
image = read_image(source)
print("Model:", descriptor.architecture.name, "device:", descriptor.device,
      "dtype:", descriptor.dtype, "tiling:", descriptor.tiling, "input shape:", tuple(image.shape))
out = Path("evaluation/runs") / ("nafnet_precision_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
out.mkdir(exist_ok=False)
try:
    for label, precision in (("current", current), ("ieee", "ieee")):
        torch.backends.cudnn.conv.fp32_precision = precision
        started = time.perf_counter()
        restored = upscale(image, descriptor)
        if restored.shape != image.shape:
            raise ValueError(f"Unexpected output shape: {tuple(restored.shape)}")
        destination = out / (label + ".png")
        write_png(restored, destination, source)
        print(label, "conv precision:", torch.backends.cudnn.conv.fp32_precision,
              "raw min/max:", restored.min().item(), restored.max().item(),
              "seconds:", round(time.perf_counter() - started, 1), "saved:", destination)
        del restored
        torch.cuda.empty_cache()
finally:
    torch.backends.cudnn.conv.fp32_precision = current
PY
```

目視核對 `current.png` 是否重現原始彩色方塊，再比較 `ieee.png`。若前者不能重現，不能把差異歸因於精度；若兩張都有同樣異常，精度假設沒有解決問題，不再重複此對照；若只有 IEEE 圖正常，才有依據提出入口的精度修正及其最小代表圖驗證。這兩張屬定位證據，不能當正式全量成績。
