# SR／deblur 原尺寸四指標評測

本次 lab 固定 deblur-only，預設 1 張。新 evaluation 至少要一個模式旗標，重複旗標報錯；
出現順序即執行順序，不自動展開其他順序或 1＋3N 矩陣。
含 deblur 未指定單顆时，掃 `models/deblur/` 第一層 .pth/.pt/.ckpt/.safetensors，排序並 resolve 去重。
SR 始終只選一顆明確 `--sr-model PATH`，不固定 ATD、不掃 SR 目錄。

```bash
.venv/bin/python evaluation/run_evaluation.py --deblur --limit 1
.venv/bin/python evaluation/run_evaluation.py --deblur --deblur-model models/deblur/fftformer_GoPro.pth --limit 1
.venv/bin/python evaluation/run_evaluation.py --sr --sr-model models/sr/selected.pth --limit 1
.venv/bin/python evaluation/run_evaluation.py --sr --deblur --sr-model models/sr/selected.pth --limit 1
.venv/bin/python evaluation/run_evaluation.py --deblur --sr --sr-model models/sr/selected.pth --limit 1
```

`--input` 預設 `evaluation/data/input/`；原尺寸 PNG/JPG/JPEG，第一層、固定一次取樣。
`--seed` 固定隨機取樣；`--runs-root` 預設 evaluation/runs/；相對參數以 repo root 解讀。
每次新建 run、保留歷史結果。每組 ID 納入順序／checkpoint 路徑／SHA；同 stem 圖片明確拒絕，不讓它們互覆寫。
模型依組載入與釋放，階段共用 tensor，不寫中間有損檔。
安裝、模型來源、GPU 前提及 lab 首張命令見 [repo README](../README.md)。

## 四指標與有效性

使用共用 read_image 的 EXIF／RGB／8-bit 規則，轉 uint8 RGB 後以 OpenCV COLOR_RGB2GRAY 取得 float64 灰階 0–255。
不將 RGB 當 BGR、不 resize、不加強、不換 JPEG。基準每張在同 run 只算一次；
後值重新解碼**實際交付的最終 PNG**。沒有 GT／PSNR／SSIM／LPIPS 或合成退化。

| 指標 | 固定算法 | deblur 逐張變化 | 朝較清晰方向 |
|---|---|---|---|
| laplacian_variance | OpenCV Laplacian CV_64F, ksize=1，變異數 | 後÷前 | >1 |
| tenengrad | 3×3 Sobel 梯度平方的平均 | 後÷前 | >1 |
| cpbd | 來源向量化 CPBD；64×64 blocks、原常數與 Canny 規則 | 後−前 | >0 |
| crete_roffet_blur | skimage blur_effect, h_size=9 | 後−前 | <0 |

比值的前值為 0 記 N/A，保留原值，不加 epsilon。
CPBD 前後任一方 edge_count=0 時比較無效；有可量邊緣的真 0 分有效，
逐張保留 edge_count／valid_blocks／mean_edge_width，摘要列處理後邊緣消失張數。
Crété 非有限結果記不可量測，不把來源的 fallback 1.0 當有效分數；
在固定 skimage 0.26.0 下，平坦 64×64 圖由函式本身回傳有限 1.0，與原工具相同，
小至 3×3 的非有限案例另有測試。低紋理分數的解讀仍有限制。

每項獨立記有效性及失敗。處理失敗與指標失敗分開；
一項失敗不刪除成功圖片或其他分數。明確執行錯誤回傳非零，正常 N/A 不表示推論失敗。
摘要先逐張算 ratio／delta，再取中位數及朝清晰比例；平手留在有效分母，列反向／排除原因及有效／總數。
四項不平均成總分，中位數不表示對稱抵消後的總改善量。
多模型只在該項、該比較的共同有效圖片比較，列樣本完整來源及覆蓋率。
全體共同集合空就不產生全體排名，仍保留其他模型的有效配對比較。

SR／combine 只列原始前後分數和尺寸，變化／摘要標「跨尺寸不適用」；
沒有 bicubic／ATD 額外基線或假 GT。本次只驗證這些程式路徑，沒有其真實評測。
雜訊、過銳化與假紋理可能提高分數，清晰度相關變化不等於真實細節恢復。

## 產物與來源

- `report.md`：環境、版本、模型 SHA／架構／device／descriptor 尺寸與 tiling、逐模型耗時、
  CUDA peak allocated/reserved（可用時）與 process_max_rss_kib（整個程序截至該組完成的累積峰值，非單模型獨占峰值）、
  各項摘要、失敗與樣本入口。
- `per_image.md`：input/output、尺寸、順序、模型身分、未先四捨五入資料生成的四項前後／變化／validity/reason/debug。
  兩份互鏈，摘要直接從同一批完整精度紀錄計算，只有顯示時格式化。
- 歷史 runs_summary.md、runs_extracted.json、既有 runs 與 GT 不覆寫。

核心來源為使用者提供的 Downloads/metrics 工具（2026-10-02），已納入 `blur_metrics.py`，
執行不依賴 Downloads、pandas、原工具 CSV API 或 ThreadPoolExecutor。
必要調整：RGB tensor 的灰階轉換、各項錯誤隔離、CPBD 無邊緣標記、Crété 非有限不套有效 fallback。
CPBD 原通知完整保留，授權見 [LICENSE-CPBD.txt](LICENSE-CPBD.txt)，授權檔僅移除原尾端空白。
來源 SHA-256：

| 原檔 | SHA-256 |
|---|---|
| blur_metrics/metrics.py | ccdc3be71793d2098225978150e5310f7079b5530ae0ce8894d6bfdbe1096a37 |
| blur_metrics/preprocessing.py | d7c643a61519d372edadabe97e75acce8c3607cd92a7cfc5ac6ada900a4cfbbc |
| LICENSE-CPBD.txt（原始） | f42ed0aacb17b2236f3b8260180f652032d80630f95b6b5dd360ed4d9e2a777a |

## 本機可重跑的無權重檢查

```bash
.venv/bin/python -m pip check
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:evaluation \
  .venv/bin/python -m unittest test_blur_metrics test_summary test_report test_run_evaluation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests \
  .venv/bin/python -m unittest test_cli test_inference test_tiling test_image_io
bash -n lab/run.sh
```

來源核心數值比對在本機已實際執行；另一台機器沒有原 Downloads 工具時僅該外部比對案例 skip，
獨立已知數值、有效性、順序和報告檢查仍可執行。不要直接跑全部 legacy 測試：
部分會初始化 AlexNet／載入真 SR 權重，未列入本次無模型驗收。
精確結果、曾失敗的檢查與修正見 [build-log](../sr-deblur/build-log.md)。

## 保留的 legacy SR

明確指定 `--legacy-sr` 才執行下方歷史流程；
`--model NAME`／`--all` 僅用於 legacy，搜尋位置現在是 `models/sr/`：

```bash
.venv/bin/python evaluation/run_evaluation.py --legacy-sr --all --input lab/sample --limit 1
.venv/bin/python evaluation/run_evaluation.py --legacy-sr --model model.pth --input lab/sample --limit 1
```

以下原文件與分數是歷史證據，其舊入口／路徑以本節覆蓋，不能當新 deblur 成績。

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
