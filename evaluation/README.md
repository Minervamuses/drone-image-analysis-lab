# 評估入口與指標

評估共用 `src/drone_sr/` 的模型與推論，不另外實作 SR 或 Deblur 引擎。環境、權重来源及固定 Lab 實驗見 [主 README](../README.md)。

## 一般評估：SR／Deblur 與處理順序

```bash
.venv/bin/python evaluation/run_evaluation.py --deblur --limit 1
.venv/bin/python evaluation/run_evaluation.py --deblur --deblur-model models/deblur/selected.pth --limit 1
.venv/bin/python evaluation/run_evaluation.py --sr --sr-model models/sr/selected.pth --limit 1
.venv/bin/python evaluation/run_evaluation.py --sr --deblur --sr-model models/sr/selected.pth --limit 1
.venv/bin/python evaluation/run_evaluation.py --deblur --sr --sr-model models/sr/selected.pth --limit 1
```

旗標出現順序就是推論順序，每個模式最多一次；不自動展開其他順序。

| 參數 | 預設／規則 |
|---|---|
| `--input` | `evaluation/data/input/`，只讀第一層 PNG／JPG／JPEG |
| `--limit`、`--seed` | 預設 1 張；無 seed 按名稱取樣，有 seed 固定隨機取樣 |
| `--sr-model` | 有 SR 就必須明確指定一顆 |
| `--deblur-model` | 可指定一顆；否則掃 `models/deblur/` 第一層 checkpoint，排序並按實體路徑去重 |
| `--runs-root` | `evaluation/runs/`，每次建立新 run |

相對路徑參數以 repo root 解讀。所有模型組使用同一份取樣；每組 ID 包含順序、模型路徑及 SHA-256。一般評估不自動套用 Lab 的 NAFNet 排除清單。

來源不做合成退化；依共用讀圖規則校正 EXIF、轉 8-bit RGB。推論階段間保留 tensor，只在最後保存 PNG。原圖 baseline 每張只算一次，後值重新解碼實際保存的 PNG；此入口不算 PSNR／SSIM／LPIPS。

### 四項清晰度指標

唯一清單與比值集合位於 [metric_defs.py](metric_defs.py)，計算位於 [blur_metrics.py](blur_metrics.py)。轉換方式為 RGB uint8 → OpenCV COLOR_RGB2GRAY → float64、0–255。

| 指標 | 算法 | Deblur 前後變化 | 朝較清晰方向 |
|---|---|---|---|
| laplacian_variance | OpenCV Laplacian，ksize=1，變異數 | 後÷前 | >1 |
| tenengrad | 3×3 Sobel 梯度平方的平均 | 後÷前 | >1 |
| cpbd | CPBD，64×64 blocks、來源常數與 Canny 規則 | 後−前 | >0 |
| crete_roffet_blur | skimage blur_effect，h_size=9 | 後−前 | <0 |

只有單獨 `--deblur` 計算變化與改善摘要。SR 或兩階段組合只列前後原值／尺寸，變化標「跨尺寸不適用」。

- 比值前值為零時記 N/A，不加 epsilon；CPBD 前後任一方無可量測邊緣時不做有效比較，有邊緣的零分仍有效。
- 各指標獨立記 valid／status／reason／debug；CPBD 保留邊緣統計，Crété 非有限結果不當有效分數。
- 摘要從逐張 ratio／delta 取中位數及朝清晰方向比例，平手保留在有效分母。四項不合成總分。
- 多模型按各項共同有效圖片比較，列實際樣本與覆蓋率；全體共同集合空時不產生全體排名，仍可保留有效配對。
- 雜訊、過銳化與假紋理也可能提高分數；清晰度變化不等於真實細節恢復。

### 產物與失敗

每次在新 run 寫入 `report.md`、`per_image.md` 與各模型組的 PNG。兩份報告互鏈，保存完整 input/output 路徑、原始與輸出尺寸、模型 SHA／架構／device、耗時、資源及量測結果。

圖片依各流程的處理順序分配 `a.png`、`a(2).png`、`a(3).png`；不再拒絕同 stem 的不同輸入，也不覆蓋已占用的檔名。模型組目錄仍禁止重用，歷史 run 不改寫。

單張或模型失敗會記錄，成功 PNG 保留。明確指標執行錯誤使退出碼非零；正常不可量測的 N/A 不當推論失敗。一般評估退出碼：0 為沒有處理／指標錯誤，1 為執行失敗，2 為輸入或參數配置錯誤。

## Legacy：合成退化的 SR 對照

只有明確加 `--legacy-sr` 才執行。此流程要求 **RGB 4× SR 模型**，不能與 `--sr`／`--deblur` 混用。

```bash
TORCH_HOME="$PWD/models/torch-cache" .venv/bin/python evaluation/run_evaluation.py \
  --legacy-sr --model model.pth --input lab/sample --limit 1
TORCH_HOME="$PWD/models/torch-cache" .venv/bin/python evaluation/run_evaluation.py \
  --legacy-sr --all --input lab/sample --limit 1
```

`--model` 是 `models/sr/` 第一層的檔名，預設 `model.pth`；`--all` 掃該層 checkpoint 並按實體路徑去重，不能與 `--model` 同用。Legacy 預設輸入 `input/`、limit=5；seed、runs-root 與一般評估同義。範例中的 `lab/sample/` 包含既有 512×512 真實裁切樣本。

LPIPS 使用 AlexNet；與禁止下載的 Lab 不同，legacy 在 cache 缺失時可能由 torchvision 下載 backbone。先準備 cache；`TORCH_HOME` 可讓兩入口共用既有權重。

流程為：原圖解碼 → 右／下 mod-crop 到 4 的倍數 → bicubic 降採樣 4× → 同一張磁碟 LR PNG 分別交給 SR／bicubic → 與裁切後原圖比較。

| 約定 | 行為 |
|---|---|
| 輸入與輸出 | 中間圖均為 PNG，不再做有損編碼 |
| PSNR | RGB、data_range=255；相同圖得 inf；legacy 摘要排除非有限 PSNR 配對並記數 |
| SSIM | RGB 通道平均；Gaussian 11×11、σ=1.5、K1=0.01、K2=0.03、valid 邊界、加權有偏變異數 |
| LPIPS | net=alex，RGB 正規化到 [-1,1] |
| 裝置與納入 | PSNR／SSIM 為 CPU float64；SR／LPIPS 同裝置；只納入兩線皆完成量測的圖片 |

每個 checkpoint 獨立建立一個 run：

```text
evaluation/runs/<UTC時間>/
├── report.md
├── hr/
├── lr/
├── bicubic/
└── sr/
```

四個圖片目錄使用同一個編號檔名；任一目錄已有同名產物就往下一號分配，避免部分舊結果被覆蓋。`report.md` 同時列來源檔名與「輸出檔名（hr/lr/bicubic/sr）」。

Legacy 的個別圖片失敗會記錄；若該 checkpoint 仍有可用成績，退出碼可為 0。模型層級失敗或沒有任何完整成績為 1，選模／輸入問題為 2。不要只靠退出碼判定所有圖片成功。

分數只在這個合成 bicubic 退化條件下成立；原圖也不是真實低解析輸入的配對 HR。不能用它直接宣稱真實空拍場景的畫質排名。

## Lab 與上述評估的差別

`bash lab/run.sh` 固定做 Deblur 四組實驗，沒有 `--sr`、`--deblur-model`、`--input`、`--seed` 或 `--runs-root` 參數，只有 `--limit`／help。它遞迴找權重、使用 seed=923、缺 CUDA 或 LPIPS cache 就停止；無 limit 時處理全部圖片。

Lab 產生 `full_reference.csv`、`summary.csv`、`sharpness.csv`，不產生此處兩份 Markdown 報告。PSNR=inf 在 Lab 均值中保留；legacy 的 inf 摘要規則不同。四項清晰度只輸出原值與有效性，不作 ratio／delta 摘要。具體路徑、權重排除與執行命令見 [主 README](../README.md)。

## 來源、授權與檢查

清晰度計算核心來自使用者提供的 metrics 工具（2026-10-02），目前直接保存在 `blur_metrics.py`，執行不依賴 Downloads 或獨立的 `ocean-drone-release/`。

保留的調整包括 RGB tensor 包裝、各指標錯誤隔離、CPBD 無邊緣標記，以及 Crété 非有限結果的無效標記。CPBD 通知與授權見 [LICENSE-CPBD.txt](LICENSE-CPBD.txt)。

| 原檔 | SHA-256 |
|---|---|
| metrics.py | ccdc3be71793d2098225978150e5310f7079b5530ae0ce8894d6bfdbe1096a37 |
| preprocessing.py | d7c643a61519d372edadabe97e75acce8c3607cd92a7cfc5ac6ada900a4cfbbc |
| LICENSE-CPBD.txt（原始） | f42ed0aacb17b2236f3b8260180f652032d80630f95b6b5dd360ed4d9e2a777a |

本次小圖／替代模型檢查與已知舊 Lab 測試失敗見 [主 README](../README.md)。不要直接執行完整 legacy 測試當成無權重驗證：部分測試載入真實模型或 LPIPS，外部來源比對在來源工具缺失時會 skip。

`gpu_checks/` 是保留的 SR／LPIPS GPU 診斷工具，可能處理 4056×3040 圖與既有 checkpoint，並非一般入口的啟動檢查。本次未執行，執行前須另外確認模型、cache 與資源成本。歷史實驗證據見 [SR／Deblur 紀錄](../sr-deblur/build-log.md) 與 [Deblur Lab 紀錄](../deblur/build-log.md)；本次未重算其數字。
