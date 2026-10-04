# deblur — 目標與實驗約定

## 目的與原則

完成一次可在 lab server 執行的 deblur 實驗：將圖片均分為四組，各圖只接受一種處理，再把同一批處理圖送入全部未經使用者授權排除的 deblur checkpoint，留下可自行分析的數據。

2026-10-04 使用者決策：最小展開、能跑優先；直接覆寫 `lab/run.sh`，本次入口專做 deblur，不處理 SR。沿用已存在的函式與依賴，不考慮未來擴展、不建立新框架、不做嚴格測試。每個有意義的修改步驟完成必要檢查後立即 commit。Git 執行及授權由 [PLANS.md](PLANS.md) 統一規定。

2026-10-04 追加使用者授權：「跳過NAFNet不管，這邊就備註有問題以及我授權跳過，繼續」。已知 `NAFNet-GoPro-width64.pth` 短試跑有彩色方塊/棋盤紋，後續執行跳過此檔名的 checkpoint（含子目錄），不再診斷或修正 NAFNet。既有五模型結果與異常證據原樣保留；後續三份 CSV 僅記錄參與執行的模型，不把跳過記成推論成功或重新改寫舊成績。

## 範圍與非目標

- 新實驗的正式程式變更限 `lab/run.sh`；可以嵌入 Python 並 import 既有讀圖、推論與指標函式。
- 基準回退的檔案範圍見第一階段；不重寫共用 SR 程式，也不維護本次入口的 SR 相容模式。
- 不做訓練、模型搜尋、SR/combine、參數掃描、90% 通過門檻、排名選王、bootstrap、反光增強、額外雜訊、JPEG 再壓縮或 Wiener 對照。
- 不新增服務、通用 registry、cache/resume、測試框架、Docker/Colab 工作或五套原廠 runner。
- 本機沒有本次資料集與 checkpoint。使用者於 lab server 準備並執行；不為本機驗收下載權重或資料。

## 輸入與四組分配

預設從 repository root 的 `evaluation/data/input/` 第一層讀取 PNG/JPG/JPEG；承接現有平坦資料目錄，不再整理遙測 CSV、搬檔或篩選 GT。使用 `models/deblur/`（`models/` 內的 deblur 子目錄）下全部 .pth/.pt/.ckpt/.safetensors checkpoint，遞迴搜尋並依相對路徑排序；不掃入 SR 權重。

先將圖檔依名稱排序，以固定 seed=923 洗牌，再按索引循環分配四組。N 張圖只產生 N 張模型輸入，各組數量最多差 1；N<4 時允許空組。分配僅做一次，所有模型使用完全相同的圖片與分組。單張失敗記錄原因，不重新洗牌或補圖。

下列為本版固定起始參數，屬實作預設，不代表最佳值；不自動調參：

| 組別 | 一張圖的處理 |
| --- | --- |
| none | 不加模糊，保留解碼後的 RGB 像素 |
| linear | 8 px、45° 的固定直線運動模糊核 |
| trajectory | 固定 90° 圓弧軌跡，最大 XY 跨度 8 px，均勻取樣形成核 |
| gaussian | Gaussian σ=2 px，13×13 核 |

模糊核權重總和為 1，運動核以質心置中；沿用現有 sRGB/linear 轉換方式做純模糊，邊界用反射。原圖本身的反光保留，不額外增強。所有中間與輸出圖用 PNG。原圖只讀，GT 永遠是「加模糊前的原圖」，不是模型輸入。

**維持原尺寸，不自動縮圖或裁成 512×512。** 沿用既有 Spandrel 載入與推論的尺寸／tiling 行為；不支援的尺寸或 OOM 記錄為該模型失敗，不以偷偷縮圖換取成功。需修改共用推論時先回報具體必要性。

## PSNR／SSIM／LPIPS 成績

每個未排除的模型都處理全部 N 張已分組圖片。將實際保存的 model 輸出與對應 GT，在相同尺寸、完整畫面上計算既有 RGB PSNR/SSIM/LPIPS。沿用 `evaluation/metrics.py` 與 `evaluation/perceptual.py` 的數值約定。

主摘要是**每個模型跨四組、全部圖片的三個算術平均**；不以每組平均再平均，不以只挑模糊組或成功範例取代使用者要求的全體結果。不增加分組排名、勝率或自動結論。

正常情況分母是 N。若個別圖／模型失敗，保留失敗列；摘要以三項全參考分數都取得的共同成功圖片計算，明列 n_expected、n_success、n_failed，且標記 partial，不能冒充全部圖片平均。完全沒有成功圖片時均值留空。完全相同影像的 PSNR=inf 是有效結果：保留 inf；含此值的算術平均亦為 inf，另記 psnr_inf_count，不擅自排除或換成任意大數。

## 四項清晰度原始資料

對每張 GT 計算一次，對每張 model 輸出各計算一次；不要求替人工模糊輸入再加一套此類量測。

| 欄位 | 使用的既有實作 | 方向／範圍 |
| --- | --- | --- |
| laplacian_variance | Laplacian 響應變異數 | 高＝較清晰；0～∞ |
| tenengrad | Sobel 梯度能量平均 | 高＝較清晰；0～∞ |
| cpbd | Narvekar & Karam CPBD | 高＝較清晰；0～1 |
| crete_roffet_blur | scikit-image blur_effect | 高＝較模糊；0～1 |

直接重用 `evaluation/blur_metrics.py::measure_tensor`，保留 `evaluation/LICENSE-CPBD.txt`。只輸出原始值與有效性資料，不呼叫其比較／摘要函式。無有效邊緣的 CPBD 保留函式原值及 invalid 標記；其他不能計算的值留空並記錄原因，不補零。

## 預期成果與產物

每次執行寫入新的 `evaluation/runs/deblur/<UTC時間>/`，保留模型輸入與各模型輸出，並只產生下列三份成績 CSV：

| 檔案 | 最小資料欄位 |
| --- | --- |
| full_reference.csv | image、condition、blur_params、model、original_path、input_path、output_path、psnr、ssim、lpips、status、error |
| summary.csv | model、n_expected、n_success、n_failed、psnr_mean、ssim_mean、lpips_mean、psnr_inf_count、status |
| sharpness.csv | image、kind（original/model_output）、model、condition、path、laplacian_variance、tenengrad、cpbd、crete_roffet_blur、status、invalid_metrics、error |

`sharpness.csv` 是四項指標的唯一成績檔：只有表頭與資料列，沒有說明段落、方向表、分析、平均、比率、評語或建議。欄位中的識別資訊、狀態及錯誤為資料。GT 列 model 留空，不按模型重複 GT 列。四項指標的局部缺值不影響已有效的 PSNR/SSIM/LPIPS。

檔名使用原始名稱含副檔名再加 .png；模型目錄使用 checkpoint 的相對路徑含副檔名，避免同 stem 覆蓋。不提交圖片、checkpoint 或整批輸出到 Git。

## 成功條件

- [ ] 四組只分配一次、數量差不超過 1；全部未排除的模型消費同一批 N 張輸入，NAFNet 跳過理由與使用者授權有紀錄。
- [ ] GT、模型輸入與輸出可由 CSV 對應；原圖、權重與歷史結果未被改寫。
- [ ] 每模型取得三個全體平均；失敗、缺值、inf 與樣本數如實呈現。
- [ ] 原圖與模型輸出的四項指標單獨存入純資料 sharpness.csv。
- [ ] 修改按步驟 commit；lab 實際跑過並檢查代表輸出後，才聲稱實驗完成。模型改善與否不是程式驗收門檻。

## 未知與來源

無需使用者現在裁決的新產品決策。圖數、尺寸、lab 資源、套件／權重／LPIPS cache 是否可用及總耗時，留給第三階段的短試跑確認。本機缺資料不阻擋第二階段完成。

來源：[專案指引](../AGENTS.md)、本次使用者要求，以及 Git 的 d805d9e、35522e6、c6a8ffc、8766f1d、33b3960。這些舊計劃只供技術取材，不繼承與本次要求衝突的舊流程或授權。
