# Drone SR / Deblur Lab

使用預訓練 Spandrel 模型批次處理圖片，保留原始圖片。實際處理與評估共用 `src/drone_sr/` 的讀寫、模型載入及推論；評估依目的分成以下入口。

## 選擇入口

| 目的 | 入口 | 工作內容 |
|---|---|---|
| 產生處理後圖片 | `python -m drone_sr` | SR、Deblur、SR→Deblur、Deblur→SR；不計算畫質指標 |
| 原尺寸來源的無參考評估 | `evaluation/run_evaluation.py --sr/--deblur` | 相同四種順序；輸出 PNG、四項清晰度指標及 Markdown 報告 |
| 合成退化的 SR 評估 | 同一入口加 `--legacy-sr` | 原圖降採樣 4×，比較 SR 與 bicubic 的 PSNR／SSIM／LPIPS |
| 固定 Deblur 實驗 | `bash lab/run.sh` | 四組模糊條件、遍歷 Deblur 權重、輸出三份 CSV |

`lab/run.sh` 是獨立實驗入口，不再轉呼叫 `run_evaluation.py`。評估指令、指標定義與結果限制見 [evaluation/README.md](evaluation/README.md)。

## 環境與安裝

目標為 **Linux／WSL Ubuntu 24.04、x86_64、Python 3.12**。在 repository 根目錄執行以下指令；Windows 使用者也須在 WSL 內使用 Git、Python 與 pip。

既有環境沿用 `.venv`。全新環境的步驟如下；這次修改未重新驗證乾淨環境安裝，首次下載包含數 GB 的框架／CUDA 套件，需先確認磁碟與網路成本。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-deps --progress-bar off \
  'https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl' \
  'https://download.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
.venv/bin/python -m pip install -r requirements-wsl.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

需要評估或 Lab 實驗時，再安裝其既有依賴：

```bash
.venv/bin/python -m pip install -r evaluation/requirements.txt
.venv/bin/python -m pip check
mkdir -p input output models/sr models/deblur evaluation/data/input
```

版本以 [requirements-wsl.txt](requirements-wsl.txt)、[pyproject.toml](pyproject.toml) 與 [evaluation/requirements.txt](evaluation/requirements.txt) 為準：Torch 2.11.0／Torchvision 0.26.0（CUDA 12.8 wheels）、Spandrel 0.4.2、extra arches 0.2.0、Pillow 12.3.0；評估另用 LPIPS 0.1.4、OpenCV headless 4.11.0.86、scikit-image 0.26.0 等。

一般 CLI／一般評估會選可用 CUDA，否則使用 CPU；CUDA 推論失敗不自動改用 CPU 重跑。**Lab 要求可用 CUDA，不能改用 CPU。** 執行前應確認 WSL 的 GPU 可見性、驅動相容性及實際可用記憶體，不能只由 GPU 型號推定可跑。

## 準備權重

程式不自動下載 SR／Deblur checkpoint。一般 CLI 必須明確提供所啟用階段的權重；下列來源沿用既有專案記錄，本次未重新下載或逐顆驗證。

| 模型 | 來源 | 說明 |
|---|---|---|
| Real-ESRGAN general x4v3 | [官方權重](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth) | SR 4×，可放在 `models/sr/` |
| FFTformer GoPro | [官方 releases](https://github.com/kkkls/FFTformer/releases) | Deblur |
| Uformer-B GoPro | [官方 repository](https://github.com/ZhendongWang6/Uformer) | Deblur |
| Restormer Motion Deblurring | [官方權重資料夾](https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK) | Deblur，extra arches |
| MPRNet Deblurring | [官方權重](https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view) | Deblur，extra arches |

SR 必須是 RGB、purpose=SR、scale>1；Deblur 必須是 RGB、purpose=Restoration、scale=1。Restoration 也可能是去噪模型，仍需核對權重來源與任務。

**Lab 會跳過所有檔名為 `NAFNet-GoPro-width64.pth` 的 checkpoint（含子目錄）**：既有短試跑出現彩色區塊／棋盤紋，使用者已授權排除。一般 CLI／一般評估沒有這項自動排除規則；歷史結果與權重保留。

## 實際處理圖片

至少指定 `--sr` 或 `--deblur`，旗標出現順序就是處理順序，每個模式最多一次。`--model` 是 `--sr-model` 的別名。

```bash
.venv/bin/python -m drone_sr --sr --sr-model models/sr/model.pth --input input --output output/sr
.venv/bin/python -m drone_sr --deblur --deblur-model models/deblur/selected.pth --input input --output output/deblur
.venv/bin/python -m drone_sr --sr --deblur --sr-model models/sr/model.pth --deblur-model models/deblur/selected.pth --input input --output output/sr-deblur
.venv/bin/python -m drone_sr --deblur --sr --deblur-model models/deblur/selected.pth --sr-model models/sr/model.pth --input input --output output/deblur-sr
```

`--input`／`--output` 可各自省略，預設 `input/`／`output/`；CLI 相對路徑以執行時目錄為準。輸入與輸出不得是同一資料夾或其目錄別名。

- 只掃第一層 JPG／JPEG／PNG／TIF／TIFF，副檔名不分大小寫；依檔名穩定排序，逐張處理。
- 依 EXIF 校正方向、轉 RGB；DJI MPO 取主影格。拒絕其他多影格來源與超過 8-bit 的 PNG／TIFF；不保留 alpha、EXIF 或 GIS metadata。
- 使用 FP32；階段間保留 tensor，只保存最終 PNG。通常大於 512 像素時使用 512 core／32 halo 分塊；不支援外部分塊的 Deblur descriptor 使用整張推論。
- 分塊仍需要完整輸入／輸出的 RAM；不支援尺寸或 OOM 會記錄失敗，不自動縮圖。先用少量代表圖確認資源與輸出。
- 單張失敗會繼續下一張；stdout 列來源→實際输出檔名，以及 Processed／Failed。成功或空輸入退出 0，處理失敗退出 1，argparse 參數錯誤退出 2。

## 所有圖片入口共用的撞名規則

各入口按自己的處理順序，以原始檔名去掉副檔名後加 `.png`。名稱已被既有檔案、目錄、符號連結或同批圖片占用時，使用第一個可用的編號：

```text
a.jpg → a.png
a.png → a(2).png
下一個 a.* → a(3).png
```

若 `a.png`、`a(2).png` 已存在，新的 `a.jpg` 從 `a(3).png` 開始；舊檔、輸入、權重與檔案別名保持原樣。失敗的圖片可能已保留批次內的名稱，編號不保證連續。

命名由 `image_io.unique_output_path` 在流程層分配；低階寫圖函式仍接收明確的目的路徑。Legacy 的 `hr/lr/bicubic/sr` 四個目錄採同一個檔名，報告列出實際名稱；新評估報告及 Lab CSV 保存實際路徑。Lab 不再使用 `a.JPG.png` 格式。既有歷史產物不重新命名。

此規則針對圖片。評估每次仍建立新 run，保留其既有時間戳命名及禁止重用 run／模型組目錄的規則。

## Lab：固定四組 Deblur 實驗

Lab 僅接受 `--help` 與 `--limit N`。**不帶 `--limit` 會處理全部輸入**，且每張都交給全部未排除的 checkpoint。

| 設定 | 固定行為 |
|---|---|
| 輸入 | `evaluation/data/input/` 第一層 PNG／JPG／JPEG |
| 權重 | 遞迴掃描 `models/deblur/` 的 .pth／.pt／.ckpt／.safetensors |
| 取樣 | 先按檔名排序，以 seed=923 洗牌，再取前 N 張 |
| 分组 | 循環分配 none／linear／trajectory／gaussian，每圖只屬一組 |
| 尺寸 | 保留原尺寸，不裁切或自動縮小 |

固定條件：none 不加模糊；linear 為 8 px／45° 直線；trajectory 為 90° 圓弧、最大 XY 跨度 8 px；gaussian 為 σ=2 px、13×13 核。合成模糊使用線性光與反射邊界。所有模型使用同一批圖片及分組。

Lab 必須事先準備 LPIPS 套件與 AlexNet backbone，禁止自動下載。預設 backbone 位置：

```text
models/torch-cache/hub/checkpoints/alexnet-owt-7be5be79.pth
```

可從 [PyTorch 官方 AlexNet 權重](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) 另行取得並放到該路徑，或以 `TORCH_HOME` 指向既有 Torch cache。LPIPS 的 `weights/v0.1/alex.pth` 校準權重則由已安裝的 lpips 套件提供；腳本會檢查兩者。

```bash
bash lab/run.sh --help
CUDA_VISIBLE_DEVICES=0 bash lab/run.sh --limit 1
# 確認首張成本、輸出及 cache 後，四張可涵蓋四組：
CUDA_VISIBLE_DEVICES=0 bash lab/run.sh --limit 4
```

`--limit 1` 只涵蓋 none 組；不是四組驗證，也不是只跑一顆模型。腳本會列裝置、可用 VRAM、host RAM、cgroup 限制及耗時。全量與長時間實驗需另行確認成本。

每次建立 `evaluation/runs/deblur/<UTC時間>/`：

| 產物 | 內容 |
|---|---|
| `inputs/`、`outputs/<checkpoint相對路徑>/` | 已分組模型輸入、每模型的最終 PNG |
| `full_reference.csv` | 原圖／輸入／輸出對應，PSNR／SSIM／LPIPS、狀態與錯誤 |
| `summary.csv` | 每模型三項均值、expected／success／failed、PSNR inf 數與 partial 狀態 |
| `sharpness.csv` | 原圖各一次、各模型輸出各一次的四項清晰度原值與有效性 |

Deblur 模型釋放後才載入 LPIPS。PSNR／SSIM 在 CPU float64 計算，LPIPS 使用 CUDA；比較對象是加模糊前的原圖與實際保存 PNG。摘要使用三項全參考分數皆成功的共同圖片；失敗列保留，不假裝全量平均。PSNR=inf 保留並納入 Lab 平均。

四項清晰度局部缺值不使有效的全參考成績失敗。任一全參考失敗最後退出 1；單張／單模型失敗仍繼續其他項目。CSV 的 ok 只表示處理與量測完成，不保證畫質改善。

## 模組與驗證狀態

`metric_defs.py` 是四指標清單與比值指標集合的唯一來源，不 import 其他模組；`blur_metrics.py` 計算，`summary.py` 統計，`report.py` 呈現，Lab 引用清單組 CSV 欄位。Lab 的 help 在 ML 套件載入之前處理。

2026-10-06 的命名／常數調整已用 CPU 小圖、替代模型、既有指標／報告測試及入口檢查驗證；未重新執行真實 checkpoint、GPU 推論或乾淨環境安裝。可重跑的相關檢查：

```bash
PYTHONPATH=src:tests:evaluation OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest \
  test_cli test_image_io test_runner.OrderedBatchTests test_runner.LegacyNamingTests \
  test_blur_metrics test_summary test_report \
  test_run_evaluation.EvaluationSelectionTests \
  test_run_evaluation.EvaluationModeTests \
  test_run_evaluation.ModeIntegrationTests
bash -n lab/run.sh
PYTHONDONTWRITEBYTECODE=1 bash lab/run.sh --help
```

`test_run_evaluation.LabArgumentsTests` 尚在測試舊 Lab 轉呼叫介面：本次發現 7 個 assertion failures（含子案例）與 1 個 error，在原始提交 `3359a7c` 也可重現，未列入上述選定檢查、未刪除或修改。部分其他 legacy 測試會載入模型或下載 AlexNet，不應直接把完整測試探索當成無權重檢查。

既有 Lab 四張真圖短試跑與代表圖觀察記錄於 [deblur/build-log.md](deblur/build-log.md)；NAFNet 的異常尚未定位，四模型全量結果仍不能由本次程式測試代替。生成的紋理不等於真實地物細節。

本 SR／Deblur 專案目前沒有可交付的 Dockerfile；獨立 `ocean-drone-release/` 的拼接／指標 Docker 不等於此流程的容器。舊指令與歷史測量可由 Git 歷史及既有 build-log 查閱，不能套用為目前入口的使用說明。
