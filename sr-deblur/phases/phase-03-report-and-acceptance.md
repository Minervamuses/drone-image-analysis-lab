# Phase 03 — 四指標、兩份報告與本機交付

## 來源與目標

讀 [GOALS.md](../GOALS.md) 的四指標契約、[PLANS.md](../PLANS.md)、前兩階段證據、指定 metrics 原始碼及 evaluation 的 `metrics.py`、`perceptual.py`、`summary.py`、`report.py` 與 tests。
成果：deblur 原圖／輸出的四項真實量測、逐張變化與有效性，形成主報告／逐筆報告；完成本機代表整合與使用者 lab 操作交付。
既有檔名保留以免斷連結；本 phase 已不以 lab 真實驗收作完成條件。

## 範圍、依賴與非目標

Phase 02 Complete。四指標、方向與統計已在 GOALS 決定，沒有要等使用者裁決的前提。
只取工具純量測核心及必要灰階／有效性處理，整合至專用 `evaluation/blur_metrics.py` 或同等小範圍既有模組；保留 CPBD notice／license。
不搬其 CLI／CSV／pandas／ThreadPoolExecutor，不新增 generic metric registry，不變更原 legacy 三指標語義。
必要新 evaluation 套件依 PLANS 自行安裝、固定並檢查，不另問批准；不安裝或下載預訓練模型來完成測試。

## 實作與量測驗證

1. 先讀指定來源 `/mnt/c/Users/garyc/Downloads/metrics`；以檔案 SHA 記錄引入的版本。納入 repo 的核心與 license 必須可獨立於 Downloads 運行，README 交代來源與改動。
2. 固定 GOALS 的 OpenCV kernel、CPBD 參數與 Crété h_size=9。復用其數值算法；只為區分不可量測／失敗及一致 RGB 轉換做必要調整，不另研發指標。
3. 對同一解碼的原圖及交付 PNG 算四值，base 每張一次供 deblur 共用；每個指標獨立有效性與錯誤。CPBD debug、Crété fallback、零分母及尺寸處理嚴格依 GOALS。
4. 逐張算 ratio／delta 再統計，不用兩批平均或中位數相除代替。平手保留分母，無有效樣本記 N/A；不同模型比較時寫明共同樣本與覆盖率，不能把失敗當改善。
5. SR／combine 只有日後明確選擇才執行；本機驗證其接上相同四指標接口與「跨尺寸變化不適用」語義，不為此執行真 SR、全矩陣或另加基準。
6. 主報告只由本 run 的量測、狀態與可證實規則產生；保留樣本目視連結、失敗／不可量測、指標限制。舊「只能表格」tests 限 legacy，新契約可有事實說明，不能用模板捏造畫質結論。
7. `per_image.md` 能對回每張 input、輸出、模型／模式／順序、四個原始前後值、變化、validity／reason、CPBD debug；不重複放完整逐筆表到主檔。兩檔同源未四捨五入量測、互鏈及數量一致。
8. 同步新 CLI、模型目錄、安裝依賴與 lab 流程的 README；不把舊歷史成績當新 deblur 證據。每批改動驗證、記錄後 commit。

## 最小必要測試

沿用 unittest，可新增一個 `evaluation/test_blur_metrics.py`，它只驗證四指標需求，不建立框架或大樣本基準：
- 一張有邊緣的微小 RGB 圖（CPBD 至少 64×64）經新核心與來源核心量測，對固定算法應一致；來源載入／import 不得意外觸發其 CSV API。來源沒有測試也不能當正確性保證。
- 已知 identity 前後分數相等、ratio=1／delta=0；灰階／RGB色序、CPBD 真實 0 vs 無邊緣、Crété 非有限、零分母、單指標失敗皆有可區分結果。
- 用手工已知測量值驗證逐張後中位數、方向／平手／有效分母、共同樣本與兩報告統計，不要求所有指標對任何人工模糊圖都單調。
- 新 module 不寫額外 CSV、不啟新工作池；新路徑即使無 LPIPS 權重也能完成，不偷偷下載。
- 同一小圖＋非交換 synthetic descriptors 經正式共用推論、CLI 與 evaluator 保存 PNG，核對相同條件最終像素一致；deblur 接實際四指標與 report/per_image，並測一張失敗不抹掉另一張成功。

於 repo 根目錄、Linux .venv：
```bash
.venv/bin/python -m pip check
PYTHONPATH=src:evaluation .venv/bin/python -m unittest test_blur_metrics test_report test_summary test_run_evaluation
PYTHONPATH=src:tests .venv/bin/python -m unittest test_cli test_inference test_tiling test_image_io
bash -n lab/run.sh
git diff --check
```

`test_blur_metrics` 是本 phase 預定新增，不是目前已存在／已跑的測試。代表整合案例納入上述既有 test_run_evaluation／test_cli，不另做新 harness。
收尾檢查之前已通過且未受影響的測試無須反覆跑。完整 evaluation suite 的真權重／LPIPS 案例不作本次無模型完成條件，明列未跑原因。

## lab 操作交付（本機不執行、不等待）

交付說明必須涵蓋：
1. 使用者自行 push，lab pull；lab 依 repo 說明建／更新 .venv 並安裝新核心＋評測依賴。GPU 驅動／CUDA／可用 VRAM／RAM 以 lab 實際唯讀檢查為準。
2. 使用者自行 `mkdir -p models/sr models/deblur evaluation/data/input`，把 GOALS 五候選下載到 deblur，準備少量真實輸入；不要求 ATD 或 GT，不需要搬其他模型湊單顆測試。
3. 先以 `lab/run.sh --deblur-model PATH --limit 1` 選一顆小樣本，再以腳本預設遍歷已放入的 deblur。這是預定新介面，實作後用 help 與 shell 參數測試核對成可照做命令，不能把計劃範例稱已在 lab 成功。
4. lab 腳本在第一個昂貴推論前檢查 GPU／模型與輸入，印出模式、選中模型／圖片數和 output。壞模型記失敗並續行；空模型／無 GPU 明確報錯，不默默 CPU 長跑。
5. 在報告記錄實際 checkpoint／SHA、套件版本、裝置、輸入／輸出尺寸、逐模型耗時與可取得的資源用量。首次小樣本開圖檢查清晰度、雜訊、色偏與 tile 邊界；先按實測成本決定使用者後續批量指令。
6. Uformer 下載權限、真權重載入、GPU相容、實際內存／時間、畫質均明列尚未本機驗證。任何後續 lab 結果只在實際取得後追加 build-log，不改寫本機證據成真模型驗收。

## 驗收條件

- [ ] 四指標核心與必要依賴落地，來源／CPBD授權完整；無 Downloads 執行依賴或高階 API 副作用。
- [ ] 逐張 ratio／delta、有效性與摘要例子正確；不可量測不填假分數，單一失敗不丟其餘資料。
- [ ] 本地代表整合串過 deblur、真實指標、安全輸出與兩份報告；CLI／evaluation 同條件輸出一致。
- [ ] SR／combine 模式架構與跨尺寸標示有 mock／synthetic 證據，本輪沒做其真實評測。
- [ ] lab deblur-only 命令／模型來源／資料放置／依賴／首張檢查有可操作說明，未驗證限制明列。
- [ ] 相關便宜回歸有結果，所有本次改動 commit，無 push／切 branch，既有修改保留。
- [ ] GOALS 本機交付成功條件與 PLANS 完成標準逐項對照後達成；缺 lab 真實結果不阻擋本項。

## 恢復、證據與完成

必要本地檢查失敗按 PLANS 修正，不擴成模型／GPU掃描；失敗輸出與原因保留，不清空舊 run。
build-log 記實際命令、樣本種類、量測比對、報告位置、commits、未驗證範圍。完成後交付 commit 清單、說明連結與精確 lab 命令；本機任務到此結束，不等待使用者同步或真權重下載。
