# SR／deblur 可選模式與無參考評測 — 目標

## 目的與最新決策

本計劃於 2026-10-02 建立，同日依使用者最新決定修訂。交付是本機可驗證、可提交、供使用者同步到 lab 的程式；不是本機真實模型實驗。
一般入口保留 SR、deblur 及兩種 combine 順序；evaluation 也能選擇要執行的模式。本次 `lab/run.sh` 固定 deblur-only，遍歷 lab 的 deblur 權重，以指定工具的四項無參考指標評估。
使用者自行 push、lab pull、下載權重及執行真實評測；本機不得下載 SR／deblur／LPIPS 預訓練權重，不以缺少它們阻擋本機交付。
本次決策取代舊版「固定 ATD、1＋3N 全矩陣、GT 配對、指標未決、lab 四路驗收是開發完成前提」。舊文件與報告只作歷史證據。

## 預期成果與介面

| 旗標及出現順序 | 一般 CLI／新 evaluation 的處理 |
|---|---|
| `--sr` | 僅 SR |
| `--deblur` | 僅 deblur |
| `--sr --deblur` | SR → deblur |
| `--deblur --sr` | deblur → SR |

- 一般入口與新 evaluation 都至少需要一個模式旗標，同一旗標不可重複；無旗標於載入模型前報錯。舊 evaluation 僅在明確 `--legacy-sr` 下沿用原流程。
- 一般入口採 `--sr-model PATH`、`--deblur-model PATH`，只要求啟用階段的權重；舊一般入口 `--model` 可保留作 `--sr-model` 別名，但不能省略模式旗標。
- 新 evaluation 的 SR 階段必須指定一顆 `--sr-model PATH`，不掃 SR 目錄、不固定 ATD。含 deblur 的模式若有 `--deblur-model PATH` 就只用該顆，未指定則掃 `models/deblur/` 第一層。
- 選 deblur 時有 N 組；選 SR 時一組；選單一 combine 順序時有 N 組。一次只執行所選順序，不自動展開 1＋3N，不順便計算另一順序。
- 本次 lab 腳本固定 `--deblur`，預設 `--limit 1`；可指定單顆 deblur 或輸入／取樣數。SR／combine 的程式能力以本地 synthetic／mock 驗證，本輪不做它們的真實評測。
- 一般入口與新 evaluation 共用解碼與有序 tensor 推論；中間階段不轉存 PNG／JPEG，最後才寫 PNG。

## 模型、素材與真實執行分工

所有執行路徑相對 repository root，不寫死任何人的部署路徑。

| 內容 | 位置與規則 |
|---|---|
| SR 權重 | `models/sr/`，一般 SR／使用者日後選 SR 評測時使用 |
| deblur 權重 | `models/deblur/`，lab 由使用者下载下列候選權重 |
| 真實評測輸入 | `evaluation/data/input/`，原尺寸 PNG／JPG／JPEG，第一層、固定一次取樣 |
| 評測產物 | `evaluation/runs/` 下的新 run；組合 ID 區分順序、權重与圖片，不能只靠權重 stem |
| 既有 GT 目錄 | 保留目錄及已有檔案，本次不讀、不要求、不刪除 |

模型來源與架構支援已查閱；實體 checkpoint 的載入、GPU、尺寸及分塊效果仍待 lab 確認，不作本機交付前提：

| 使用者指定候選 | 官方檔名／定位 | 預定載入器 |
|---|---|---|
| FFTformer GoPro | `fftformer_GoPro.pth`；[官方 releases](https://github.com/kkkls/FFTformer/releases) | Spandrel 0.4.2 核心 |
| NAFNet-GoPro-width64 | `NAFNet-GoPro-width64.pth`；[使用者／官方下載](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view) | 核心 |
| Restormer Motion Deblurring | `motion_deblurring.pth`；[使用者／官方資料夾](https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK) | 官方 spandrel_extra_arches |
| Uformer-B GoPro | 官方測試定位 `GoPro/Uformer_B/models/model_best.pth`；[官方 repo](https://github.com/ZhendongWang6/Uformer) | 核心 |
| MPRNet Deblurring | `model_deblurring.pth`；[使用者／官方下載](https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view) | 官方 spandrel_extra_arches |

不同模型同名時由使用者使用可辨識檔名；報告仍記錄路徑、架構、SHA-256，不能據檔名推定模型內容。Uformer 的外部下載可能需要登入，不承諾匿名可下載。
原 `models/model.pth` 不在本機搬移；程式、仍使用的 legacy 路徑與說明必須支援使用者搬到 `models/sr/` 後的配置。

## 四指標契約

演算法來源是使用者指定的 `/mnt/c/Users/garyc/Downloads/metrics`。實作時將必要核心與 CPBD 授權納入 repo；lab 不得依賴該本機絕對路徑。
本次不要求 GT，不計 PSNR／SSIM／LPIPS，不製造模糊／清晰配對；舊 legacy 評測的既有三指標與結果另行保留。

| 指標 | 固定量測方式 | deblur 逐張變化 | 朝較清晰方向 |
|---|---|---|---|
| laplacian_variance | OpenCV Laplacian ksize=1 的變異數 | 後 ÷ 前 | > 1 |
| tenengrad | 3×3 Sobel 梯度平方的平均，不用總和 | 後 ÷ 前 | > 1 |
| cpbd | 原工具向量化 CPBD，保留常數與 debug 統計 | 後 − 前 | > 0 |
| crete_roffet_blur | skimage blur_effect，h_size=9 | 後 − 前 | < 0 |

- 以同一解碼／EXIF／RGB／位深規則取得輸入，明確轉灰階 0–255；輸出量測對象為實際交付的最終 PNG。不得將 RGB 當 BGR，不為計分 resize、加強或換 JPEG。
- 每張輸入的基準指標在同一 run 算一次，給各 deblur 共用；這只是該 run 的資料重用，不加持久化 cache。
- 每個指標各自彙總逐張變化的中位數、朝較清晰方向的比例、平手／反向張數、有效數／總數及排除原因；平手留在有效分母。四指標不得平均成總分。
- 比值的前值為 0 時記 N/A，保留原始前後值；不用任意 epsilon。中位數只是穩健摘要，不解讀成對稱抵消後的總改善量。
- CPBD 前後任一方沒有可量邊緣時，該指標比較記不可量測；有可量邊緣的真實 0 分仍有效。記錄前後 edge_count、valid_blocks，不刪掉其他指標，也不隱藏處理後邊緣消失的張數。
- Crété 的非有限結果不得默默替換成有效 1.0；保留原工具算法，但將此 fallback 明示為不可量測並附原因。
- 解碼／推論／寫圖失敗與各個指標失敗分開。單一指標失敗不能丟棄成功圖片或其餘可用分數；沒有有效分母就寫 N/A。
- 多模型對照限同一指標、同一比較的共同有效圖片，列樣本數與覆蓋率；全體共同集合为空就不產生全體排名，不因一顆壞模型清空其他有效配對比較。
- 使用者日後明確選 SR／combine 時，可記最終輸出的四項原始值和尺寸；不將原尺寸 input 與放大輸出算變化倍率。該類前後摘要記「跨尺寸不適用」，不自動增加 bicubic／ATD 基線或假 GT。
- 指標反映清晰度相關變化，不等於去模糊成功率或真實細節恢復。主報告保留雜訊、過銳化、假紋理、低紋理限制及代表圖片目視入口。

## 報告與成功條件

同一批未四捨五入量測產生 `report.md` 與 `per_image.md`，互相連結。主報告含環境、模型、模式、樣本、彙總、失敗／不可量測、限制與樣本連結；逐筆檔含 input、output、模型／順序、各指標前後值／變化／有效性與原因。
舊 `runs_summary.md` 等分析不覆寫；沒有 lab 證據不產生「某模型最好」的結論。

- [x] 一般 CLI 四條路徑、旗標順序和新 evaluation 模式選擇有本地可觀察驗證。
- [x] 本次 lab 命令只選 deblur；沒有 SR 權重、GT 或 LPIPS 預訓練權重仍能完成該路徑的本地整合測試。
- [x] 五種候選的預定載入支援與所需套件已接入；不把 mock／架構偵測當真權重驗收。
- [x] 四指標計算、逐張變化、有效性、共同樣本與兩份報告有必要本地檢查。
- [x] 原圖／權重安全、逐張失敗隔離、同名衝突與新 run 不覆寫保持正確。
- [x] 本機三階段、說明與每批改動的 commit 完成；使用者可自行 push、lab pull 後準備模型與圖片。
- [x] 真實模型／lab／GPU／畫質及未跑的 SR／combine 評測如實列為未驗證，不阻擋本機交付，不聲稱已通過。

## 範圍、非目標與保留行為

範圍：一般 CLI／載入／共用推論，新 evaluation 的選擇／runner／數據／四指標／摘要／兩份報告，lab bash、必要依賴與說明、既有 unittest 的最小延伸。
不做訓練、模型搜尋、全矩陣或全資料集實跑、GT 流程、五套原廠 runner、通用 registry／服務／並行框架、cache／resume、Docker／Colab 重做。
原圖、權重和歷史產物只讀；保留 EXIF、DJI MPO、RGB、拒絕高位深與覆寫防護。一般 CLI 成功後仍可原子替換同名舊輸出，失敗留舊檔但不能當成新成功；四模式比較範例指定不同 output。
現有 SR 512 core／32 halo 保留。deblur 按 descriptor 處理 padding／cropping 與 tiling 限制；沿用 FP32，不預設 FP16 可用。Restoration 類別也含 denoise，不能取代權重來源核對。
執行授權、每次 commit 及續作規則由 [PLANS.md](PLANS.md) 集中管理。

## 未知與待決事項

沒有需要使用者中途裁決的產品決策。依賴精確版本、局部介面和測試細節由執行代理按此目標自行落實並記錄。
lab 的檔案內容、實際 GPU／資源、checkpoint 載入與圖像效果尚未驗證，由使用者在交付後執行所提供命令；不是本機 phase 的前置依賴。

## 來源

- 2026-10-02 使用者本輪：只採指定四指標、本次 lab deblur-only、架構可選模式、本機改好由本人 push／lab pull／下載權重；啟動即授權必要工作、每次改動 commit、禁止 push／切 branch。
- [專案指引](../AGENTS.md)、[正常入口](../src/drone_sr/__main__.py)、[推論](../src/drone_sr/inference.py)、[評測入口](../evaluation/run_evaluation.py)、[lab 入口](../lab/run.sh)。
- [Spandrel 0.4.2 descriptor](https://github.com/chaiNNer-org/spandrel/blob/v0.4.2/libs/spandrel/spandrel/__helpers/model_descriptor.py)、[extra arches 0.2.0](https://pypi.org/project/spandrel-extra-arches/0.2.0/)、[官方 extra registry](https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel_extra_arches/spandrel_extra_arches/__helper.py)。
