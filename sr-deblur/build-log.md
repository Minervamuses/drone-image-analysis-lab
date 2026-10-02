# SR／deblur — Build Log

本檔是 runtime phase 狀態及實際實作／驗證證據的唯一來源。未來命令與驗收要求在 phase 文件，不當成已執行結果。

## 階段摘要

| Phase | 狀態 | 開始 | 完成 | 證據 | 阻塞 |
|---|---|---|---|---|---|
| 01 — 本機環境與三模式 | Not started | — | — | — | 尚未啟動實作，環境可依既有授權自行準備 |
| 02 — 可選模式與 deblur lab | Not started | — | — | — | 等待 01 的本機驗收 |
| 03 — 四指標、報告與交付 | Not started | — | — | — | 等待 02 的本機驗收 |

狀態只使用 Not started、In progress、Blocked、Complete。只有必要本機驗收有觀察證據才標 Complete；不複製另一份 phase 狀態。
lab 真實權重／GPU／圖片品質尚未驗證；依最新分工屬使用者後續操作，不是上述本機 phase 的阻塞。

## 證據規則

- 寫明實際 checkout、命令、環境、結果、產物與 commit；分開記本地 synthetic／mock、指標數值檢查、歷史報告及 lab 真實結果。
- 缺項與失敗如實記錄；不記秘密、完整資料集、例行冗長輸出。重大錯誤追加更正。
- 重要實作新理解才寫 context，實際程式審查才寫 code_review；每批紀錄／修正也依 PLANS 提交。
- 不為記錄某 commit 自身 SHA 無限新增 commit；當次名稱與 Git 歷史可追溯，下次事件再引用前次 SHA。

## 活動紀錄

原始建檔記錄保留：尚無實作活動或新功能驗證證據。最初計劃撰寫及使用者要求的空資料夾準備不代表任一 phase 開始或完成。

### 2026-10-02 — 依最新決策修訂計劃，未啟動實作

- 使用者將本次 lab 評測改為 deblur-only 與指定四項無參考指標；移除本次 GT、ATD 全矩陣及等待 lab 四路驗收的要求，保留可選模式架構。
- 使用者自行 push／lab pull／下載權重；啟動即授權本計劃必要工作，每批改動 commit，禁止 push／切 branch。
- 已唯讀確認 WSL root、main／7a7c6ee 基線及既有未提交修改；本機缺 .venv。官方來源核對五候選的 core／extra_arches 支援，但未下載或載入真權重。
- 本次只修訂原七個計劃 Markdown，狀態全部維持 Not started；不得把文件檢查當作程式／模型測試。
- 文件結構、內容一致性與範圍檢查結果在實際執行後追加；計劃尚未實作。

未來實作事件格式：日期時間／phase、狀態變更、實際範圍、確切檢查與結果、產物／commit、限制／阻塞、下一個符合依賴的動作。

### 2026-10-02 — 修訂文件檢查

- skill 的 validate_harness.py：repo 為本專案、plan-root=sr-deblur、minimal／medium、harness-only／strict／json，七個實際修改路徑各列 proposed-path 與 allowed-path；修正 Phase 01 章節標題後回報 0 errors、0 warnings、valid=true。
- 獨立 fresh-agent 由 PROMPTS 及七文件走讀：沒有阻擋本機完成的問題，未發現殘留裁決／權重／lab 等待 gate。
- 七檔相對 Markdown 連結及尾端空白檢查通過；SHA-256 對照顯示僅這七檔改變，其餘 73 個原有非忽略檔案內容不變。git diff --check 通過。
- 以上僅文件驗證；未執行程式測試、安裝依賴、下載模型或推論。三 phase 維持 Not started。
