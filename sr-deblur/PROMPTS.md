# SR／deblur — 可重複使用的提示

## 啟動或續作整體執行

> 請執行 repository 根目錄 `sr-deblur/` 的計劃。我的啟動指令代表已授權完成這份計劃所需的一切本機工作；依 `sr-deblur/PLANS.md` 自主落實必要依賴、環境、CLI、資料／報告結構、跨檔修改、驗證與修正，不停下等待我再裁決。每一批改動都 commit，禁止 push 或切換分支。
>
> 先讀適用 AGENTS.md、`sr-deblur/GOALS.md`、`sr-deblur/PLANS.md`、`sr-deblur/build-log.md`，再依 build-log 與依賴順序選第一個非 Complete 的 `sr-deblur/phases/` 文件，以及相關 context／code_review 和 live code／tests。使用者在 PLANS 記錄的明確授權優先於較一般的重複確認要求。
>
> 先唯讀確認工作樹與 WSL/Linux 工具鏈，保存既有修改；進行該 phase 最小實作、必要測試、失敗診斷／修正、證據記錄與 scoped commit。環境尚缺時自行按 PLANS 準備，不能把已授權安裝再變成等待許可。必要檢查通過並提交後，自動繼續下一個 phase。
>
> 依 GOALS 的最新契約工作；不要恢復舊版的固定 ATD／1＋3N／GT／未定指標或等待 lab 驗收 gate。本機不下載預訓練權重、不跑真模型；這些是使用者在 lab 的後續操作，不是本機缺失前提。
>
> 四指標與順序的必要檢查失敗先修，不將失敗當通過。遇反證自主修訂未開始 phase；只為重要新發現建 context，只在做過實際程式審查後建 code_review。真實環境障礙按 PLANS 診斷與記錄，不能靠要求使用者重複授權來處理。
>
> 每批變更含修正／說明／紀錄都依 PLANS 提交，只 stage 本次自己的差異，不納入無關既有修改。不要等待我 push、lab pull 或下載權重。持续至三 phase 的本機完成條件全部達成；最後報告 commits、必要檢查、lab 執行方式及真模型尚未驗證的範圍。

## 執行單一階段

> 請只執行我指定的 `sr-deblur/phases/` 階段。先讀相同 durable sources，確認其依賴；按 PLANS 的既有授權完成該階段實作、檢查、紀錄及每批 commit，然後停止。單階段範圍是我此次指令的限制，不是計劃的逐階段審批要求。

## 純審查

> 對照 `sr-deblur/GOALS.md`、`sr-deblur/PLANS.md`、指定 phase、實際 diff 與 build-log 審查。分清模式選擇、本次 deblur-only 預設、四指標數學／有效性、原圖保護、歷史 legacy 相容與 Git 邊界。只回報有證據的問題；純審查不改檔或實作。

## 新證據出現時修訂計劃

> 讀目標、路線、build-log、受影響 phase 與 live code。指出反證、保留已觀察證據，自主修訂本計劃內受影響的技術路線與未開始工作並 commit，不重新把已決指標或授權標為待使用者決定。若當前請求只是修改計劃，完成文件檢查與提交後停止，不順便開始 Phase 01。

## 最終本機整合審查

> 逐項對照 GOALS 的本機成功條件及 PLANS 完成標準，查驗實際本地命令、代表 synthetic 圖片輸出、真實四指標計算、兩份報告與相關回歸。确认 lab 腳本預設 deblur-only，SR／combine 只驗證架構而未實跑，全部改動有 commit 且未 push／切 branch。交付使用者在 lab 準備權重、安裝依賴及首次小測的說明，不等待遠端結果、不宣稱真實模型畫質已通過。
