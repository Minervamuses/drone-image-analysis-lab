# 目前評測介面

新模式使用原尺寸輸入、不讀 GT、不做 legacy 降採樣。至少選 --sr 或 --deblur，出現順序就是處理順序：
```bash
.venv/bin/python evaluation/run_evaluation.py --deblur --limit 1
.venv/bin/python evaluation/run_evaluation.py --deblur --deblur-model models/deblur/selected.pth --limit 1
.venv/bin/python evaluation/run_evaluation.py --sr --sr-model models/sr/selected.pth --limit 1
.venv/bin/python evaluation/run_evaluation.py --sr --deblur --sr-model models/sr/selected.pth --limit 1
.venv/bin/python evaluation/run_evaluation.py --deblur --sr --sr-model models/sr/selected.pth --limit 1
```

含 deblur 未指定單顆時掃 models/deblur/ 第一層，SR 始終只選一顆。
--input 預設 evaluation/data/input/，--seed 可重現取樣，--runs-root 指定產物根目錄。
每次新建 run；report.md／per_image.md 互鏈並列每圖狀態。Phase 02 四指標暫標待量測，不當成量測通過。
舊模式必須加 --legacy-sr；舊 --model/--all 僅對該模式有效，模型目錄為 models/sr/。
以下原文件與驗證數字是 legacy 歷史，不能當成新 deblur 成績。Lab 新入口見 [README](../README.md)。

---

# evaluation — SR 線 vs bicubic 基線

一個獨立的評估工具，回答一個問題：

> 一張低解析的圖進來，走本專案的 SR，畫質是否真的贏過一般的放大手段（bicubic）？

本工具透過主 pipeline（`src/drone_sr/`）的公開函式載入所選 checkpoint 並推論；主程式仍預設使用 `models/model.pth`。所有產物寫在 `evaluation/runs/<timestamp>/` 之下，`input/`、`output/`、`models/` 皆為唯讀。

## 真值是自己造的

手上沒有配對且對齊的高解析度真值，所以本工具用**合成退化**：拿高解析原圖當真值，自己降採樣造出低解析輸入，再用兩種方式放大回去，與原圖比對。

```
原圖 ──解碼──▶ mod-crop ──▶ 【真值】
                   │
                   └─bicubic 1/4─▶ 【LR PNG（磁碟上的檔案）】
                                        ├─ Pillow bicubic 4× ─▶ bicubic 輸出 ─┐
                                        └─ drone_sr 未修改路徑 ─▶ SR 輸出 ────┤
                                                                              ▼
                                                            與真值比 PSNR／SSIM／LPIPS
```

**兩條線的輸入是磁碟上同一個 LR PNG 檔**，bicubic 線沒有任何路徑碰得到原圖。這是整份比較公平與否的關鍵，`test_bicubic.py` 用兩項檢查守著它：覆寫 LR 檔會改變輸出，以及輸出與獨立重算逐位元相同。

## 安裝

評估用的依賴另外記錄，**不動專案的 `pyproject.toml` 與 `requirements-wsl.txt`**：

```bash
.venv/bin/python -m pip install -r evaluation/requirements.txt
```

裝的是 `lpips==0.1.4` 與它的相依 `scipy==1.18.1`、`tqdm==4.70.1`。torch、torchvision、Pillow、numpy 由專案本身提供，這份檔案刻意不重複列出，所以安裝它不可能移動這些版本。

首次執行時 torchvision 會下載 LPIPS 用的 AlexNet backbone（約 233 MB）到 `~/.cache/torch`，需要網路一次。

## 執行

```bash
.venv/bin/python evaluation/run_evaluation.py --limit 5 --seed 20260919

# 指定 models/ 內的 checkpoint，只傳檔名
.venv/bin/python evaluation/run_evaluation.py --model realesr-general-x4v3.pth --limit 5 --seed 20260919

# 每顆 checkpoint 依序評估同一批圖片
.venv/bin/python evaluation/run_evaluation.py --all --limit 5 --seed 20260919
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--input` | `input/` | 原圖資料夾。只取**直接子項**中的 `.png`／`.jpg`／`.jpeg`（大小寫不敏感），忽略子資料夾與其他副檔名。 |
| `--limit` | `5` | 處理幾張。**預設刻意很小**，全量 738 張不在本工具的日常用法內（見「全量的成本」）。 |
| `--seed` | 無 | 給了就隨機取樣且可重現；不給就依檔名順序取前 N 張。 |
| `--runs-root` | `evaluation/runs` | run 目錄的位置。 |
| `--model` | `model.pth` | `models/` 第一層的 checkpoint 檔名，不能與 `--all` 同用。 |
| `--all` | 關閉 | 依檔名順序執行 `models/` 第一層的 `.pth`／`.pt`／`.ckpt`／`.safetensors`；指向相同檔案的 symlink 只執行一次。 |

`--all` 只取樣一次，每顆 checkpoint 都使用同一份圖片清單、相同退化流程與評估指標。每顆依序載入，完成後釋放模型，再執行下一顆；時間與輸出空間會隨 checkpoint 數增加。評估仍要求 RGB 4× 模型，無法載入或倍率不符會記錄失敗並繼續下一顆。

`bash lab/run.sh` 會先檢查 CUDA 與套件，預設以 `--all --input lab/sample --limit 1` 呼叫本入口；可加上 `--model NAME.pth`、`--input`、`--limit`、`--seed` 覆寫。GPU 交接檢查腳本仍使用預設模型，需另外執行。

退出碼：`0` 所有 checkpoint 均有可用成績；`1` 任一 checkpoint 執行失敗或沒有圖片在兩條線上都量到；`2` 參數、checkpoint 選擇或來源資料夾有問題。個別圖片失敗記錄於報表。

### 每次執行留下什麼

```
evaluation/runs/<UTC timestamp>/
├── report.md      checkpoint 名稱與 SHA-256、執行資料、逐張成績、平均、失敗清單
├── hr/            mod-crop 後的真值
├── lr/            降採樣後的低解析輸入（兩條線都讀這個）
├── bicubic/       bicubic 線的輸出
└── sr/            SR 線的輸出
```

**絕不寫進既有的 run 目錄。** 目錄名是 UTC 時間戳；同一秒內再跑一次會得到 `-2`、`-3`，底層的 `mkdir` 帶 `exist_ok=False`，所以任何情況下舊紀錄都不會被覆蓋或修改。本工具也不會自動刪除舊 run，何時清理由你決定。

`--all` 中每顆 checkpoint 各自建立一個上述 run 目錄及 `report.md`，輸出不互相覆蓋。報表只保留資料表，不再加入前言、解讀前提或結論段落；歷史報表保留原樣。

## 固定約定（改了就不能和舊數字並列）

| 項目 | 值 |
|---|---|
| 降採樣與放大 | Pillow `Image.Resampling.BICUBIC`，倍率 4 |
| mod-crop | 自右／下裁到寬高皆為 4 的倍數；**裁切後的原圖才是真值**，裁切不改動保留下來的像素 |
| 中間格式 | 全程 PNG。除了讀原始檔那一次，沒有第二次有損編碼、第二次縮放或色彩空間轉換 |
| 色彩空間 | RGB 三通道、8-bit、`data_range = 255`。**不是論文常見的 Y 通道** |
| PSNR | `10·log10(255²/MSE)`；`MSE == 0` 記為 `inf`，該張排除於 PSNR 平均，SSIM／LPIPS 照常納入 |
| SSIM | Gaussian window 11×11、σ=1.5、K1=0.01、K2=0.03；邊界 `valid`（不補邊）；Wang et al. 的加權有偏變異數；三通道各算後平均 |
| LPIPS | `lpips==0.1.4`、`net='alex'`、輸入正規化到 `[-1, 1]` |
| 平均 | 只涵蓋**兩條線都成功量到**的圖片；任一線失敗，兩邊都不計入 |

**不得為了讓分數好看而更動其中任何一條。** 那是換一個實驗，不是修 bug。

### 裝置

SR 線與 LPIPS 跟隨 `torch.cuda.is_available()`；**PSNR 與 SSIM 永遠在 CPU 以 float64 計算**，因為它們的輸入直接來自 Pillow，不搬到 GPU。後果：PSNR／SSIM 跨裝置可精確重現，SR 與 LPIPS 的數字則不可跨裝置並列比較。報告標頭分欄記載這件事。

## 結果怎麼讀

**三個指標要分開下結論。** PSNR 與 SSIM 衡量逐像素保真度，LPIPS 衡量感知相似度，方向可以相反而不矛盾。

本專案的模型 `realesr-general-x4v3` 是以真實世界複合退化（模糊、雜訊、壓縮）訓練的 GAN，而這裡的退化是**乾淨的 bicubic 降採樣**。這個組合下 GAN 類 SR 常見的結果就是 PSNR／SSIM 輸給 bicubic、LPIPS 明顯勝出。實測（5 張、CPU，run `20260918T184323Z`）正是如此：

| 指標 | SR | bicubic | 勝方 |
|---|---|---|---|
| PSNR | 26.6615 | 27.2016 | bicubic（5 張全輸） |
| SSIM | 0.691692 | 0.711496 | bicubic（5 張全輸） |
| LPIPS | 0.437991 | 0.570740 | **SR**（5 張全贏） |

目視檢查（兩張圖、四個版本、100% 檢視）補上數字看不出來的事：

- **稀疏高對比的小目標**（浪花、小物件）：SR 明顯比 bicubic 銳利，但斑點帶有真值沒有的橙褐色偏，較暗的一些直接消失。
- **密集細紋理**（植被、粗糙水面）：**SR 比 bicubic 更不忠實**——它把大片真實的細紋理抹成平坦色塊，看來是當成雜訊去掉了。

所以「SR 比較銳利」不是一致成立的敘述，也不要用單一平均概括整張圖。**SR 增加的細節是重建，不是還原；銳利不等於正確。**

## 全量的成本

**全量 738 張不在預設用法內。** 依實測（沙箱 CPU、單張 32.8 秒；產物體積隨畫面內容而異，實測每張 **42–53 MB**）：

| | 時間 | 磁碟 |
|---|---|---|
| 5 張（預設） | 約 2.7 分鐘 | 212 MB |
| 738 張（CPU） | **約 6.7 小時** | **約 31–39 GB** |
| 738 張（GPU） | 尚未量測；SR 與 LPIPS 會快得多，但 SSIM 與 I/O 恆在 CPU，所以不會快一個量級 | 同上 |

跑之前先確認磁碟空間，並記得本工具不會自動清理舊 run。

## 限制

- **樣本代表性：** 目前的驗收樣本是同一次飛行、同一片海域的連續影像。平均值不能外推成「本專案在各類場景的表現」。
- **真值來自 JPEG：** `input/` 是 DJI 的 MPO JPEG，真值本身已經是有損解碼的結果。這不影響兩條線的對等性（兩邊比的是同一個真值），但絕對數值受此影響。
- **只在 bicubic 退化下成立：** 本工具不處理真實的低解析影像，結論不可外推成真實退化下的畫質排名。
- **SSIM 沒有第三方交叉核對：** `scikit-image` 與 `torchmetrics` 不在授權依賴內。改以一份刻意寫慢的獨立重新推導核對（逐視窗走訪、二維 kernel），三組輸入下差距 ≤ 1.3e-15。這證明快速路徑算的是預期的定義，**不**證明該定義與其他工具一致。
- **PNG 原圖只以合成 fixture 驗證：** `input/` 目前 0 張 PNG。
- **GPU 路徑的資源數字尚未取得：** 見 `evaluation/gpu_checks/`。

## 檢查

```bash
.venv/bin/python -m unittest discover -s evaluation
```

94 項檢查，涵蓋退化契約、兩條線的對等性、三個度量的性質、平均的納入規則與報告完整性。與專案既有的 `tests/` 分開，互不掃到。

> 沙箱 session 內 `~/.cache/torch` 可能唯讀，此時需加 `TORCH_HOME=<可寫目錄>` 前綴。一般的使用者 shell 不需要。

## GPU 交接

Coding agent 的沙箱 session 看不到本機 GPU，所以 GPU 上的資源數字與決定性檢查要在你自己的 shell 跑：

```bash
bash evaluation/gpu_checks/run_on_user_shell.sh
```

一次做完 LPIPS 全尺寸成本與決定性、SR 線成本與決定性、一次小樣本真實執行與 `report.md`，最後印出一段可貼回的輸出。另有一支更短的 `probe_lpips_full_size.sh`（約 30 秒），只回答「LPIPS 在 4056×3040 上放不放得進 VRAM」。

## 計劃文件

`GOALS.md`（目標與固定約定）、`PLANS.md`（順序與授權）、`build-log.md`（實際發生的事與觀察證據）、`phases/`、`context/`。**`build-log.md` 是執行狀態的唯一來源。**
