# Drone Image Super-Resolution

## 目前一般 CLI：SR／deblur 與處理順序

以下為目前程式介面；後面的歷史執行紀錄與鎖定舊 commit 的 Colab 範例保留其當時語義。
本機只驗證 synthetic／mock，新增候選的真權重、GPU 及畫質尚待 lab 實測。

在 Linux／WSL 的 repo 根目錄，沿用下方 Python 3.12 安裝流程及固定版本。
必要套件為 Spandrel 0.4.2 加上 `spandrel_extra_arches==0.2.0`（已納入 requirements）。
已有環境更新：

```bash
.venv/bin/python -m pip install -r requirements-wsl.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
```

由使用者準備 `models/sr/` 與 `models/deblur/`，不自動下載／搬移權重。
至少指定一個模式，不能重複；只要求啟用階段的權重，`--model` 是 `--sr-model` 別名。

```bash
.venv/bin/python -m drone_sr --sr --sr-model models/sr/model.pth --input input --output output/sr
.venv/bin/python -m drone_sr --deblur --deblur-model models/deblur/selected.pth --input input --output output/deblur
.venv/bin/python -m drone_sr --sr --deblur --sr-model models/sr/model.pth --deblur-model models/deblur/selected.pth --input input --output output/sr-deblur
.venv/bin/python -m drone_sr --deblur --sr --deblur-model models/deblur/selected.pth --sr-model models/sr/model.pth --input input --output output/deblur-sr
```

階段間不寫中間圖片，只在最終輸出 PNG；FP32，512 core／32 halo，descriptor 負責 padding／裁回。
不適合外部分塊的 deblur descriptor 走完整圖片，因此大圖記憶體需求仍待 lab 確認，失敗不改用 CPU 重跑。
SR 要 RGB／SR／scale>1，deblur 要 RGB／Restoration／scale=1；Restoration 也可能是 denoise，
必須另核對來源。核心＋官方 extra registry 覆蓋 FFTformer、NAFNet、Restormer、Uformer、MPRNet，
這不代表五顆真權重已載入驗收。

每張失敗會續行；第二階段失敗不會覆寫既有成功檔，也不計新成功。
一般 CLI 可原子替換同名舊輸出，因此上述四模式使用不同輸出目錄。
Docker 使用本次原始碼重新建置時，需在下方 `docker run` 的 image 名稱後明確加上
`--sr --sr-model models/sr/model.pth`，models 掛載仍為唯讀；本輪沒有重建或重跑 Docker。


## Lab server 測試副本

此 repo 為 [`Minervamuses/drone-image-analysis`](https://github.com/Minervamuses/drone-image-analysis) 的測試副本，程式與既有測試來自 commit `725605585148581eb679836310befdf5612e2499`。只補上本節、`lab/run.sh` 與一張真實小樣本；下方原專案說明保留作為背景。

### 第一次 clone 與安裝

此 repo 公開可讀，lab server 使用 HTTPS clone／pull 不需要設定 GitHub SSH 金鑰。先只執行 clone，成功才切換目錄：

```bash
git clone https://github.com/Minervamuses/test.git && cd test
```

確認目前已在 `test/` repo 根目錄，再複製下面的完整安裝區塊，包含最外層括號。命令先檢查目錄，再於子 shell 內安裝；任何一步失敗就停止。若上一段 clone 失敗，不要繼續安裝。

```bash
(
set -e
test -f lab/run.sh
test -f requirements-wsl.txt
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-deps --no-cache-dir --progress-bar off \
  'https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl' \
  'https://download.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
.venv/bin/python -m pip install --no-cache-dir --only-binary=:all: --progress-bar off -r requirements-wsl.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip install -r evaluation/requirements.txt
.venv/bin/python -m pip check
)
```

安裝成功後，於同一個 repo 根目錄繼續「準備正式權重」。若 `test/` 已經存在，請先進入既有 repo 並 `git pull --ff-only origin main`，不要重複 clone。

沿用原專案的 Linux x86_64／Python 3.12／CUDA 12.8 套件，不修改依賴版本。套件下載約 4 GB，首次 LPIPS 另下載約 233 MB 的 AlexNet，請預留約 15 GB 安裝空間。安裝時間受網路影響，可能超過十分鐘。Server 需能連線 GitHub、PyTorch、NVIDIA 套件站與 PyPI。

使用者提供的 server 是 Ubuntu 24.04.2、Python 3.12.3、glibc 2.39，四張 RTX A6000（每張 49140 MiB）、driver 595.58.03。**cgroup 記憶體上限為 60 GiB**，不能將 `free -h` 顯示的 377 GiB 當成本程序可用量。測試預設選一張 GPU；CUDA wheel 是否能在該 server 實際執行，由下方入口的 GPU 檢查確認。

### 準備正式權重

Git 不包含模型權重。若已有正式 Compact 權重，將它放在 `models/model.pth`；否則在 repo 根目錄執行下面區塊，從原專案已指定的官方來源下載約 4.9 MB 並核對已記錄的 SHA-256。已有檔案會先核對，不覆寫；下載失敗不會留下名為 `model.pth` 的半成品。

```bash
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
from urllib.request import urlopen

target = Path("models/model.pth")
expected = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"
url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
if target.exists():
    data = target.read_bytes()
else:
    with urlopen(url, timeout=120) as response:
        data = response.read()
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit("Checkpoint SHA-256 mismatch; no file was written.")
if not target.exists():
    with target.open("xb") as stream:
        stream.write(data)
print("models/model.pth: SHA-256 OK")
PY
```

### 執行與後續 pull

```bash
# 第一次安裝與準備權重後：指定當下獲分配的 GPU 編號
CUDA_VISIBLE_DEVICES=0 bash lab/run.sh

# 後續更新同一個 lab clone
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES=0 bash lab/run.sh
```

入口先確認 CUDA、執行 `pip check`，再呼叫 `evaluation/run_evaluation.py`，預設使用 `--all` 依序評估 `models/` 第一層的所有 checkpoint，不要求固定的 `model.pth`。也可用 `bash lab/run.sh --model NAME.pth` 指定其中一顆。**CUDA 不可用會停止，不會改成 CPU 跑完整流程。** CPU 的數值計算預設使用 8 個 OpenMP／MKL threads，可透過 `OMP_NUM_THREADS`／`MKL_NUM_THREADS` 覆寫。模型與 LPIPS 使用可見 GPU 的第 0 張，並沒有多 GPU 平行處理。

預設使用 `lab/sample/` 的一張 **512×512** 圖，評估時先降採樣成 128×128，再還原成 512×512。每顆 checkpoint 都使用同一批圖片；請確認各輪 `measured` 是 1、`excluded` 是 0。這是小樣本驗證，**不是 4056×3040 大圖驗證**。測試套件與 GPU 決定性探測不再於每次評估前重跑；需要時可分別執行 `.venv/bin/python -m unittest discover -s tests`、`.venv/bin/python -m unittest discover -s evaluation` 或 `bash evaluation/gpu_checks/run_on_user_shell.sh`（測試及交接腳本仍使用 `models/model.pth`）。

若圖片已另行上傳 server，可改用自己的資料夾：

```bash
CUDA_VISIBLE_DEVICES=0 bash lab/run.sh --input /path/to/images --limit 1
```

`--limit` 限制每顆 checkpoint 使用的相同圖片批次。先以一張確認耗時與結果，再決定是否增加數量。4.7 GB 的本機原始資料集與歷次輸出沒有推上此 repo，`git pull` 不會取得它們。

每顆 checkpoint 的結果各自放在 `evaluation/runs/<timestamp>/`，包含標註模型名稱與 SHA-256 的 `report.md`、`hr/`、`lr/`、`bicubic/`、`sr/`。請回傳終端機輸出與 report，並檢視對應影像；小樣本通過不代表大圖效能或真實低解析影像的畫質已驗證。每次執行建立新目錄，舊結果保留且不納入 Git。

### 小樣本來源

`lab/sample/whaledrone_seek10s_x1536_y768_512.png` 為既有驗證使用的原始 PNG，263,958 bytes，SHA-256 `ad7d8815928ea78bb2243af8639541216e83a7444d78272d91451a2d7dd63faa`。來源為 [WhaleDrone: Los Cabos Humpback Whale UAV Dataset](https://huggingface.co/datasets/LucieLprt-Dvldr/WhaleDrone)，作者 Lucie Laporte-Devylder；資料集另列 Esther Jimenez、Hiram Rosales Nanduca 為共同貢獻者。依資料集的 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 授權，用於非商業研究測試並保留署名。

本專案的處理為：從 `DJI_20260114193309_0004_V.MP4` 第 10 秒取 RGB 畫面，裁切 `(x=1536, y=768, width=512, height=512)` 後存成 PNG。影片 SHA-256 為 `8dddd14150efee239002d536fa33446629cbde3979ce4e76429a23a4c1f56fda`。這張裁切是鏡像額外納入 Git 的小樣本；原始資料集與其他 `test-data/` 仍排除。

### 此副本的驗證狀態

2026-09-19 在本機 WSL、RTX 5070 Ti Laptop GPU，借用既有 `.venv`、Compact 權重與 AlexNet 快取實跑 `lab/run.sh`：`pip check`、33 項主程式測試、94 項評估測試全數通過；一張真實樣本兩條線都成功、0 張排除，LPIPS 與 SR 決定性皆為 True，產生 `report.md` 與 512×512 結果。已開啟來源與 SR 輸出確認構圖、尺寸及可解碼性；SR 水面細紋較平滑，不宣稱畫質提升。另以空的 `CUDA_VISIBLE_DEVICES` 確認入口在測試前退出，不改走 CPU。

Lab server 的安裝與 GPU 執行尚待使用者實跑；本次也未重新建立乾淨環境或重下載依賴。下方原專案的歷史驗證紀錄不是該 server 的測試結果。

---

本機、單一 Spandrel 推論流程：從 `input/` 或指定資料夾逐張讀圖，以專案內的 `models/model.pth` 放大，把同 stem 的 RGB PNG 寫進 `output/` 或指定資料夾，原圖保留不動。沒有 GUI，不訓練或微調，也沒有模型選擇參數。

交付預設為官方 `realesr-general-x4v3.pth`（Compact）。已完成的最大實例是一張真實 3840×2160 海面畫面處理為 15360×8640 PNG；SwinIR-M 另通過 512×512 真實裁切與最小分塊相容性。已驗證與未驗證的項目逐條列於「驗證狀態」。

## 涵蓋範圍

**會做：** 資料夾批次；逐張隔離失敗並給出 `Processed`／`Failed` 摘要與退出碼；自動選用 CUDA 或 CPU；超過 512 的圖自動分塊；讀入時依 EXIF Orientation 校正方向；拒絕多頁、高位深與浮點來源。

**不會做：** 不下載權重；不遞迴掃描子資料夾；不轉換影片；**本 pipeline 本身不計算任何畫質指標**，也不做模型排名；不保存 alpha、EXIF、GIS 等 metadata；不提供 Docker 映像。

PSNR／SSIM／LPIPS 由另一個獨立工具 [`evaluation/`](evaluation/README.md) 提供，它以合成退化自造真值，比較本 pipeline 與 bicubic 放大。該工具只 import `src/drone_sr/` 的公開函式，不改動它，產物也只寫在 `evaluation/runs/` 之下。

## 採用的環境與版本

目標為 Linux／WSL Ubuntu 24.04、Python 3.12、x86_64。使用 `.venv` 與 pip，透過 setuptools 的 editable 安裝連結 `src/drone_sr/`；模型位置固定錨定原始專案目錄，與執行命令的位置無關。

| 元件 | 版本／來源 |
|---|---|
| Python | 本機 3.12.3，Ubuntu 24.04.2 |
| pip | venv 內 24.0 |
| torch | 2.11.0+cu128，PyTorch 官方 CUDA 12.8 wheel |
| torchvision | 0.26.0+cu128，與 torch 配對的官方 wheel |
| spandrel | 0.4.2，PyPI |
| Pillow | 12.3.0，PyPI |
| setuptools | 81.0.0，`setuptools.build_meta` |
| NumPy／safetensors／einops | 2.5.3／0.8.0／0.8.2，PyPI 傳遞依賴 |
| Triton | 3.6.0，PyPI 傳遞依賴 |
| CUDA runtime／cuDNN | 12.8.90／9.19.0.56，NVIDIA 官方 wheel |

torch／torchvision 的版本配對依 [PyTorch 官方安裝說明](https://pytorch.org/get-started/previous-versions/)；採 CUDA 12.8 wheel 的依據是 [Blackwell 支援](https://pytorch.org/blog/pytorch-2-7/)。[requirements-wsl.txt](requirements-wsl.txt) 固定主要套件、Triton 與必要 CUDA wheel 的版本／官方來源；NVIDIA wheel 的 SHA-256 與 PyPI 對應檔案相同。NumPy、safetensors、einops 等其餘傳遞依賴由 pip 解析，上表記錄本次實際版本，並非完整 lock。

本機唯讀資源觀察為 NVIDIA GeForce RTX 5070 Ti **Laptop** GPU，driver 591.74，VRAM 12227 MiB，compute capability 12.0。這些硬體資訊本身不能證明 checkpoint 可推論；實測狀態見「驗證狀態」。

## 安裝

以下是已批准的 Linux x86_64／Python 3.12 安裝步驟，請在專案根目錄執行。其他 Python 版本需另選對應 wheel。GPU 需要 WSL 可見的相容 NVIDIA driver；套件會提供 CUDA runtime，不需另行安裝系統 CUDA Toolkit。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps --no-cache-dir --progress-bar off \
  'https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl' \
  'https://download.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
python -m pip install --no-cache-dir --only-binary=:all: --progress-bar off -r requirements-wsl.txt
python -m pip install --no-deps --no-build-isolation -e .
```

第一步只裝兩個官方 wheel，第二步補齊必要依賴，完成前請勿執行推論。

來源採 PyTorch 的 `download.pytorch.org`、NVIDIA 的 `pypi.nvidia.com`，其餘套件使用 PyPI。前者避開本機曾回傳 403 的 `download-r2.pytorch.org`；NVIDIA 來源則處理 PyPI cuDNN 下載過慢的實際問題。CUDA wheel 約 2.93 GB，另有 torch／torchvision 約 0.83 GB 與其餘套件，合計約 4 GB。安裝期間請保留約 15 GB 磁碟；本機完成後 `.venv` 實測約 6.7 GiB。本次沿用第一次已下載成功的兩個 wheel、從本機檔案安裝後再接上述依賴命令，花費 11 分 13 秒；沒有重跑一次全新環境建置，其他網路環境的時間未驗證。

## 準備與執行

1. 將下方指定的 Compact RGB SR `.pth` 權重準備為專案的 `models/model.pth`。程式不會下載模型。本機保留原檔名，並以相對 symlink `models/model.pth → realesr-general-x4v3.pth` 使用它。
2. 將圖片放在 `input/` 或指定資料夾第一層，接受 `.jpg`、`.jpeg`、`.png`、`.tif`、`.tiff`（大小寫皆可），不遞迴掃描。
3. 在專案根目錄、啟用環境後執行：

```bash
python -m drone_sr
```

指定資料夾時：

```bash
python -m drone_sr --input "/path/to/images" --output "/path/to/sr-results"
```

兩個參數各自可省略，預設為 `input/`、`output/`，相對路徑以執行時的工作目錄為基準；含空白的路徑請加引號。模型固定於原始專案內，不隨資料夾參數改變。

### 讀寫契約

- 檔名對應同 stem 的 PNG，例如 `input/DJI_001.JPG` 對應 `output/DJI_001.png`。輸出資料夾不存在會建立。
- 成功的 PNG 完整寫入後才替換同名舊結果；來源與其 symlink／hard link 不可被當作輸出覆寫。輸出檔權限為 `0600`（原子寫入的副作用）。
- **方向：** 讀入時先依 EXIF Orientation 把方向校正到像素上，輸出 PNG 不保留方向標記。
- **拒絕的來源：** 多頁 TIFF 與多影格 PNG；每通道超過 8-bit 的高位深來源；浮點影像。高位深依**容器編碼**判定（PNG 的 IHDR 位深、TIFF 的 `BitsPerSample`），不是只看解碼後的 Pillow mode，因此 16-bit 彩色也擋得住。1／2／4-bit 與調色盤圖片是合法輸入，照常處理。
- 其餘圖片一律轉成 RGB；不保存 alpha、EXIF、GIS 或其他 metadata。
- 啟動時自動選擇可用 CUDA，否則使用 CPU，並顯示 `Device`；CUDA 執行失敗會報錯，不會暗中改成 CPU 重跑大圖。

### 失敗處理與退出碼

- 缺輸入、輸出路徑是檔案、輸入與輸出為同一資料夾、缺模型或模型載入失敗會報錯並停止；空輸入顯示 `No supported images found in input/`（指定路徑則顯示該路徑）。
- 檔名穩定排序、逐張處理，模型只載入一次。
- 壞圖、推論或儲存失敗會列出檔名及原因，繼續下一張。多張輸入映射同名 PNG 時，衝突項全部記失敗；輸出指向任何輸入的 symlink／hardlink 也拒絕。
- 成功寫出的數量為 `Processed`，其餘為 `Failed`。全部成功或沒有支援的圖片時退出碼 0；設定錯誤或任一圖片失敗為 1。

## 自動分塊與資源

- 寬、高都不超過 **512** 時整張推論；任一邊超過 512 就分塊，無需額外參數。
- 每塊有效核心最多 **512×512**，四側各帶 **32 像素上下文**，模型最大接收 576×576；圖片外緣依實際範圍裁切。按模型倍率裁掉上下文，只把核心寫回一次，descriptor 自動處理最低尺寸／補邊／裁回。
- 完整輸入與大圖拼接結果留在 CPU RAM，GPU 一次只處理當前塊。4K 圖做 4× 時，完整 RGB float32 結果本體約 **1.48 GiB**，編碼另需副本；本機代表批次程序峰值 RSS 約 **5.44 GiB**。分塊不能消除完整輸出的 RAM 需求，請先用少量圖片確認其他尺寸。
- SwinIR descriptor 的 `DISCOURAGED` 表示可分塊但上下文可能改變結果。僅驗證過 192×176、內部核心 128 的小例；未驗證 SwinIR 的 512 核心大圖或所有場景接縫。大圖預設採已驗證的 Compact。

## 採用的權重

- 檔案：[realesr-general-x4v3.pth（官方下載）](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth)，4,885,111 bytes；[release v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0)，asset id 76259217，asset 更新時間 2022-08-30。
- 本機 SHA-256：`8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`。官方 API 未提供 digest；此值用於本次取得檔案的追溯與後續一致性檢查。
- 實際 descriptor：Compact／SRVGGNetCompact、RGB 3→3、原生 4×、1,213,296 個參數；本程式採 float32。只使用此單一 checkpoint，沒有混合另一個降噪權重。
- 官方 repository [授權文件](https://github.com/xinntao/Real-ESRGAN/blob/v0.2.5.0/LICENSE) 為 BSD-3-Clause。權重與資料不納入 Git；另一台機器需另外準備檔案。

### 開發者已驗證的第二 checkpoint

- [SwinIR-M real-world 4× 官方權重](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth)：`003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth`，67,129,861 bytes；GitHub release v0.0／asset 44142419，2021-09-06。
- 本機 SHA-256：`b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6`；官方 API digest 未提供。[官方 repository LICENSE](https://github.com/JingyunLiang/SwinIR/blob/main/LICENSE) 為 Apache-2.0。
- Spandrel 0.4.2 實測辨識為 SwinIR，RGB／4×／11,715,559 parameters；minimum=16、multiple=1。採 FP32；descriptor 不支援 FP16，tiling 為 DISCOURAGED。
- 真實 512×512 → 2048×2048 RGB PNG：CLI 約 **11.64 秒**；獨立一次上傳＋GPU 推論 **3.24 秒**，PyTorch 峰值 allocated **4.32 GiB**／reserved **6.65 GiB**（截至 forward，排除 CUDA context、其他程序和 PNG 階段）。此案例在本機 12 GB 筆電 GPU 通過，不等於全尺寸 SwinIR 驗證。
- 同一權重另通過 17×19 → 68×76、1×1 → 4×4 的尺寸檢查。已開啟真實 PNG，內容、色彩和尺寸正常。來源／輸出／命令／量測位於 `test-data/phase-03-swinir-20260917/`。
- 驗證時僅暫時將 `models/model.pth` 指向此檔，完成後恢復 Compact；沒有加入模型 CLI 選項。兩顆權重都保留，交付預設為使用者選定、已通過 4K 大圖且資源需求較低的 Compact，不宣稱畫質最優。

## 驗證狀態

### 已實際驗證

**2026-09-15 — 環境與程式契約**

- `pip check` 通過；專案 editable 安裝成功。
- 以**未訓練的極小 Compact 模型**與合成 5×7 RGB 圖實跑 `python -m drone_sr`：CPU 與 CUDA 各成功寫出 10×14 PNG，原圖 SHA-256 不變。CPU 子程序以 `CUDA_VISIBLE_DEVICES=''` 隱藏 GPU，確實走 CPU 分支；GPU 子程序自動選 `cuda:0`，wheel 包含 `sm_120`。各命令約 2.03／2.48 秒，僅是這個極小案例的耗時。
- 缺模型、不可載入模型、壞圖的實際 CLI 錯誤與計數符合預期；既有成功 PNG 在模型錯誤後保持不變。`--help`、空輸入與缺輸入檢查通過。

**2026-09-17 — 真實圖片與正式 Compact 權重**

- 從 WhaleDrone 影片第 10 秒畫面取 `(x=1536, y=768, w=512, h=512)` 海面裁切，執行真正的 `.venv/bin/python -m drone_sr`：`cuda:0`、Processed 1／Failed 0，產生 2048×2048 RGB PNG，影片／輸入／權重 hash 不變。
- CLI 含程序啟動約 **5.90 秒**；另一次同一 production 函式路徑量測，GPU 同步計時的上傳＋推論約 **0.220 秒**，PyTorch 峰值 allocated 約 **264 MiB**／reserved **284 MiB**（模型載入至推論結束，不含 CUDA context、其他程式或 PNG 輸出階段，不是整張顯卡用量）。沒有 warmup 或參數掃描，兩次結果像素一致。
- 預設與指定（含空白／相對路徑）資料夾的批次，各用兩張真實小裁切與中間一張故意損壞的圖片：GPU 均跑到底，Processed 2／Failed 1、退出碼 1 符合預期，兩種介面輸出 hash 相同，無關 output 保留。
- 隱藏子程序 CUDA 後，以同一正式權重與真正 CLI 跑 32×28 真實裁切，CPU 成功產生 128×112 PNG，約 2.11 秒（含啟動）；沒有拿 mock 當 CPU 證據。
- 素材座標、格式、雜湊、完整命令與 console 存於 `test-data/phase-01-compact-20260917/`、`test-data/phase-02-cli-20260917/`。這些目錄是單次驗證紀錄，不是正式 CLI 或可重跑 benchmark。

**2026-09-17 — V1 最終驗收（分塊與 4K）**

- 兩模型的真實 direct／tiled 小例均成功：Compact 640×576 → 2560×2304，走正式 512 核心自動分塊；SwinIR 192×176 → 768×704，使用內部強制 128 核心。完整圖與接縫交會裁切已檢視，這些海面樣本未見明顯拼接線；不要求兩條路徑每像素相同。
- Compact 真正 CLI 同批次處理完整 4K 圖、故意壞圖、32×28 真實小圖：**Processed 2／Failed 1／退出碼 1** 符合預期，壞圖後仍繼續。整批含啟動／推論／PNG 約 **37.96 秒**，不是單獨 GPU forward 的時間。
- 完整輸出為 **15360×8640 RGB PNG、117,800,469 bytes**，SHA-256 `bce5bfc0b9b347f06982e13f56c49ee6cc43ca910f1bf0c558887e4d613d7822`。原圖及無關輸出 hash 保留，驗收目錄的同名舊結果只在成功後替換。
- 完整 PNG 已解碼；影像檢視工具無法傳輸約 118 MB 原檔，因此人工檢查使用全圖縮覽、原尺寸接縫及最右／最下／右下裁切，未發現明顯拼縫、空白條、重影或裁切缺失。Pillow 對此 1.33 億像素結果會發出尺寸警告，但本次解碼成功；未停用圖片保護。
- 本機結果：[全圖預覽](<test-data/phase-04-tiling-20260917/full batch/full-preview.png>)、[完整 PNG（約 118 MB）](<test-data/phase-04-tiling-20260917/full batch/output/a_full.png>)。來源、完整命令、尺寸、雜湊與裁切存於 `test-data/phase-04-tiling-20260917/`。

**2026-09-18 — 讀圖正確性修正**

`read_image()` 的兩個已重現缺陷已修正，兩者都會讓程式正常結束、輸出可正常開啟，內容卻與來源意義不符：

- **EXIF 方向未套用。** 現於 mode 與單影格檢查之後、轉 RGB 之前呼叫 `ImageOps.exif_transpose()`。八個 Orientation 值（JPEG 與帶 `eXIf` chunk 的 PNG）經 `read_image()` → `write_png()` 後，輸出**逐像素**等於正確結果；Orientation 2／3／4 尺寸不變但像素會錯，只比尺寸看不出來。TIFF 由 Pillow 自行處理，未被旋轉兩次。
- **高位深拒絕不完整。** 原本只看 `image.mode`，而 Pillow 把 16-bit PNG colortype 2／4／6 與 16-bit RGB TIFF 映射成 `RGB`／`RGBA`，通過白名單後被靜默截成 8-bit（來源通道值 256／257／511 全變成 1）。現於任何解碼之前讀容器編碼，超過 8-bit 即以 `ValueError` 拒絕，該張記為 Failed 並繼續下一張。
- repo 內 34 張既有 8-bit 圖片經 `read_image()` → `write_png()` 的輸出位元組與修正前完全相同。
- 真實小批次以正式 Compact 權重與真正 `python -m drone_sr` 執行（**CPU**，該輪環境 `torch.cuda.is_available()` 為 False）：同一張真實海面裁切分別做成無 EXIF 的 64×48 JPEG、真正的 16-bit RGB PNG，以及像素已旋轉並標記 Orientation 6 的 JPEG。結果 `Processed: 2`／`Failed: 1`／退出碼 1，16-bit 那張列出原因且未產生輸出檔，三張來源 SHA-256 不變。方向圖輸出為 256×192，與未旋轉參考圖同向（未修正時會是 192×256），兩者平均差 0.60／255，差異來自旋轉後重新 JPEG 編碼與模型非旋轉等變。

**目前測試狀態**

- 完整 correctness suite **33／33 通過、無 skipped**（V1 的 29 項加上讀圖修正新增的 4 項），既有斷言未被放寬。較早各輪的測試計數見 git 歷史。
- 測試中的合成像素、未訓練小模型與 mock 用來驗證程式契約，**不證明真實 SR 品質或預訓練 checkpoint 相容性**。

### 尚未驗證與已知限制

- 2026-09-18 讀圖修正的所有檢查都在 **CPU** 執行，未在 GPU 上重跑。此修正只影響讀圖，與裝置無關，但沒有該輪的 GPU 觀察證據。
- 位深檢查涵蓋 PNG 與 TIFF 兩種容器；JPEG 以基線 8-bit 處理，未驗證 12-bit JPEG。
- 浮點圖片的拒絕來自兩條不同路徑：單通道 float（mode `F`）回報 `Unsupported image mode: F`；32-bit float **彩色** TIFF 則是 Pillow 連識別都失敗（`UnidentifiedImageError`），同樣記為該張 Failed，但訊息不是本程式發出的。
- 未壓縮 TIFF ＋ Orientation 5–8 ＋ mode `L`／`P`／`RGBA`／`CMYK` 時，Pillow 會轉置像素緩衝區卻未更新 `size`，輸出既非原圖也非正確方向。這是上游缺陷，本次未處理；本專案實際輸入為 8-bit RGB，碰不到此路徑。
- SwinIR 的 512 核心大圖與所有場景接縫未驗證。
- 畫質：已開啟原圖與輸出檢視，海面構圖、反光位置與色彩正常，但細紋較平滑；未證明新增紋理是真實地物細節。小裁切的 VRAM 有餘裕，不能據此承諾 4K 全圖或整段影片效能。
- 畫質量化：手上仍**沒有**配對且對齊的高解析度真值，因此本 pipeline 的輸出本身不附 PSNR／SSIM。[`evaluation/`](evaluation/README.md) 繞開這個缺口的方式是**自造真值**——拿高解析原圖降採樣成低解析輸入，再由本 pipeline 與 bicubic 各自放大回去與原圖比對。數字因此只在「純 bicubic 退化」這個前提下成立，不能當成真實低解析影像上的畫質排名。5 張 4056×3040 樣本（CPU）的實測是 **bicubic 在 PSNR 與 SSIM 上全勝、本 pipeline 在 LPIPS 上全勝**；這是 GAN 類 SR 在乾淨 bicubic 基準上的已知傾向，解讀前提見該工具的 README。
- 驗收資料只來自使用者指定的單一 [WhaleDrone](https://huggingface.co/datasets/LucieLprt-Dvldr/WhaleDrone) MP4（資料集標示 CC-BY-NC-4.0），沒有下載 SRT 或其他影片。結果是海面場景，沒有鯨魚／道路／屋頂細節驗收。
- 不含整段影片轉換與 Docker 映像。
- 未重新建立第二套乾淨環境驗證安裝。
- 圖片、權重與 `test-data/` 不納入 Git；另一台機器需自行準備 `models/model.pth` 與素材。

### 重跑檢查

```bash
python -m pip check
python -m unittest discover -s tests -p 'test_image_io.py' -v
python -m unittest discover -s tests -p 'test_inference.py' -v
python -m unittest discover -s tests -p 'test_cli.py' -v
python -m unittest discover -s tests -p 'test_tiling.py' -v
python -m unittest discover -s tests -v
```

讀圖修正的目標、階段文件與完整觀察證據見 [fix/GOALS.md](fix/GOALS.md) 與 [fix/build-log.md](fix/build-log.md)。已完成並移除的 V1 四階段計劃仍可由 git 取回，例如 `git show af6989e~1:build/build-log.md`。
