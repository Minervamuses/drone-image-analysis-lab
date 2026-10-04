# Phase 03 — lab 短試跑與一次正式實驗

## 目標、範圍與非目標

在 lab 實際取得 [GOALS](../GOALS.md) 的輸出與數據。由使用者準備資料／checkpoint、同步已提交程式並執行；本機不補下載。遠端連線、push 與代跑 GPU 的界線以 [PLANS](../PLANS.md) 為準。這是一次實驗的交付，不是模型研究或調參階段。

## 前置與未知

第二階段 Complete。lab 的 repository 已有本次提交、現有 .venv、`evaluation/data/input/`、`models/deblur/` 和 LPIPS 所需權重 cache。只核對本次需要的 GPU、可用 VRAM/RAM/磁碟、圖數與尺寸、checkpoint 數與套件是否能載入；不跑完整診斷套件。

未確定的是原尺寸各模型是否能在 lab 執行以及耗時；用下面唯一一輪短試跑確認，不先假定 GPU 型號或 CPU 回退可接受。若目前無 lab 執行管道，記錄待使用者執行，本階段不得標 Complete，也不阻擋已完成的本機交付。

## 執行與最小驗證

在 lab 的 Linux shell、repository root：

```bash
bash lab/run.sh --limit 4
```

有至少四張圖時應各分一組；不足四張則以實際數量執行，不補圖。只檢查 CSV 列數／關聯／平均與少量實際圖：GT 未改、模型輸入符合所分組別、輸出同尺寸且不是黑圖／損壞檔。四項 CSV 只有資料，沒有自動分析。不要求模型贏過輸入，不設定畫質門檻。

記錄一輪的耗時、可用資源與失敗情況，據 N 與模型數作粗估。若 OOM／依賴缺失／讀圖失敗，先回報具體原因並作直接修正；原尺寸、公共推論或依賴的變更依 PLANS 處理。任何程式修正仍需在本機 WSL 檢查後立即 commit，再由使用者同步，不在 server 無紀錄地改程式。

短試跑結果先追加至 ../build-log.md，保持本階段 In progress，並在本機 WSL 用 `git add -- deblur/build-log.md`、`git diff --cached --check`、`git commit -m "docs: record bounded lab deblur run"` 形成追蹤點。無阻礙後，由使用者啟動一次正式全量：

```bash
bash lab/run.sh
```

正式結果以新 run 的全部 N 張為準，不能把四張短試跑成績當全量。代理如要代跑，須先按 PLANS 告知成本並取得相應執行授權。沒有新原因不重跑全資料集。

## 驗收

- 各模型在 full_reference.csv 有 N 個對應記錄（成功或明確失敗），summary.csv 每模型一列；用一個模型的逐圖數據核對平均與分母即可。
- sharpness.csv 有每張成功讀取原圖的一列，以及各成功輸出的一列；四個指標值／缺值狀態可對應圖片，不要求四項全部非空。
- 有代表圖的實際目視與三份檔案的存在／內容證據。處理錯誤不能以 exit code 或語法通過掩蓋。
- 個別模型失敗如實揭露；完全沒有有效模型結果時，不宣稱實驗成功完成。沒有畫質改善也照實交付，不擴張成新模型搜尋。

## 證據、commit 與完成

取得結果後只把執行日期、lab 環境、程式提交、run 路徑、N／模型數、耗時、檔案核對及失敗／未驗證項目追加至 ../build-log.md；數據分析留給使用者。不把圖片、權重、整批 CSV 或舊輸出加入 Git。

在本機 WSL 提交本次紀錄：

```bash
git add -- deblur/build-log.md
git diff --cached --check
git commit -m "docs: record observed lab deblur experiment results"
```

達到上述觀察後標第三階段 Complete。報告實際結果和限制後停止；不延伸為後續優化或研究。
