# Phase 02 — 可選評測模式與 deblur-only lab 入口

## 來源與目標

讀 [GOALS.md](../GOALS.md)、[PLANS.md](../PLANS.md)、Phase 01 證據、`lab/run.sh` 與 evaluation 的 `run_evaluation.py`、`sources.py`、`sr_line.py`、`runner.py`、`runs.py`、`report.py` 及相關 tests。
成果：evaluation 選一條處理路徑、對固定樣本輸出可追溯結果；本次 lab 預設只處理 deblur，不需要 ATD、GT、LPIPS 或其他 SR 權重。

## 範圍、依賴與非目標

Phase 01 必須 Complete。本地用 tiny images、synthetic descriptors／mock checkpoint 選擇，不要求實體模型。
在既有 evaluation 模組修改選模、調度、結果狀態與兩檔骨架，不建立另一個推論核心。Phase 03 負責四指標實值與統計。
不建配對 GT pipeline、不製造 blur／降採樣、不跑全矩陣、不下載模型、不實測 SR／combine。保留 legacy 合成 SR 路徑及歷史結果。

## Preflight 與實作方向

1. 核對現有 imports、來源取樣、run 唯一性與失敗處理。新模式不能經過 legacy 固定 mod-crop／1/4 degradation，也不能因 top-level perceptual import、metadata 或測試初始化要求 LPIPS。
2. 新模式採 GOALS 的有序 flags 及 `--sr-model PATH`、`--deblur-model PATH`。舊合成 SR 評測以明確 `--legacy-sr` 選入，互斥新模式，保留舊 `--model NAME`／`--all` 的原意，搜尋位置更新至 `models/sr/`。不把 `--all` 偷換成新 deblur 意義。
3. 新模式沒有 flags 就報错，SR-only 指定一顆 SR，deblur-only 不探查 SR 檔案。含 deblur 時可選單顆，或掃 `models/deblur/` 第一層 .pth／.pt／.ckpt／.safetensors，穩定排序、resolve 去重；不支援／壞檔記失敗，不能靜默過濾。
4. N=2 時 deblur 有兩組、選任一 combine 順序也只有兩組、SR-only 一組；不得產生七組。空 deblur 目錄是明確配置錯誤，不能退回 SR 或宣稱成功。
5. 同一 run 固定一次取樣供選定模型共用，直接用 Phase 01 共用解碼與 tensor 執行。模型逐顆處理並釋放，單顆失敗繼續其他候選；某張失敗繼續其他圖，最終有實際執行錯誤則非零。
6. 每次新建 suite run，輸出子目錄以不相撞的組合 ID 包含模式、順序、權重身分。輸入同 stem 造成同名輸出時明確拒絕，不覆蓋任何輸入／舊 run。既有成功輸出不能充當本 run 成功。
7. 結果能表示 image／mode／order／checkpoint／output、推論狀態，以及各指標 value／validity／reason。此 phase 骨架標「待量測」，不是缺 GT；不填假數字或宣告未來指標已通過。
8. lab/run.sh 移除 `--all` 與 ATD 前提，在腳本明列本次 `--deblur`、input 為 `evaluation/data/input/`、limit=1；GPU／.venv／pip check 沿用。腳本拒絕透傳 SR／combine／legacy 模式避免誤擴本次範圍；其他模式由使用者直接呼叫 Python evaluator 選擇。
9. lab 可傳 `--deblur-model PATH` 只檢查一顆；沒有此參數才遍歷全部。保留指定 input、limit、seed 的能力。此單顆功能必須在本 phase 完成，不能把「搬走其他權重」或只用普通 CLI 當作替代。
10. 模型目錄、程式預設與活躍說明引用一致；不搬使用者檔案、不刪既有 GT 目錄。歷史文件中的舊路徑是記錄，不全面搜索替換。

## 必要驗證

在 repo 根目錄、Phase 01 的 Linux .venv：
```bash
PYTHONPATH=src:evaluation .venv/bin/python -m unittest test_run_evaluation test_runs test_sources test_report test_runner.SelectSourcesTests test_runner.DeviceMemoryTests test_sr_line.ScaleAssertionTests
bash -n lab/run.sh
git diff --check
```

在相關既有 test 檔補少量 selection／順序／失敗／模式隔離案例，以 fake loader 驗證模型呼叫；驗證 deblur-only 在沒有 SR、GT、LPIPS 權重時可執行，無網路下載。
以 stub Python 的便宜 shell 檢查驗證 lab 預設參數、拒絕 SR flags、保留單顆選擇；不能只靠 bash -n 推定執行參數正確。直接沿用 unittest，不新建 shell 測試框架。
舊 tests 依明確 legacy 入口调整，保留舊合成語義檢查；兩份報告骨架與此 phase 狀態一致，不重寫歷史報告。
必要檢查失敗先修，不進 Phase 03；每批驗證、紀錄並 commit。

## 驗收條件

- [x] 所選模式的組數／順序符合 GOALS，無隱藏全矩陣，所有組共用樣本。
- [x] deblur-only 不載 SR／LPIPS、不看 GT、不降採樣，單顆／全部選擇可預期。
- [x] 空目錄、壞權重、單圖失敗、不支援模型均可定位；其他有效工作保留，錯誤退出碼正確。
- [x] run／輸出名稱不相撞，重跑不覆寫舊 run，兩份骨架可追溯。
- [x] lab 腳本固定 deblur，GPU 檢查仍在；本機的 shell 參數測試通過但不宣稱 GPU 已測。
- [x] legacy 路徑明確保留，自己的變更已提交。

## 恢復、證據與交接

素材與權重只讀；失敗 run 保留，不拿刪除／重跑掩蓋問題。build-log 記命令、模式映射、計數及失敗證據、commits。重要介面反證自主修訂下一 phase。
本機驗收完成後自動進 Phase 03；指標已確定，不再停下詢問，不等待 lab。
