# SR／deblur — 執行計劃

## 概要與資訊唯一來源

- 計劃根目錄：`sr-deblur/`；沿用三個 `phases/` 檔案，不增加流程框架。
- 目標、模式／四指標契約與候選來源：[GOALS.md](GOALS.md)。
- 執行模式：**Autonomous within authorization envelope**。使用者說「執行此計劃」即授權完成本計劃需要的一切本機工作；不得逐階段等待同意。
- 專案形態 minimal；風險 medium，因介面、模型支援及評測資料會變更。
- 本檔擁有路線、授權、停止／恢復及完成標準；`build-log.md` 唯一擁有 runtime 狀態／觀察證據；`PROMPTS.md` 是入口；各 phase 擁有具體工作及檢查。重要發現才建 `context/`，實際程式審查才建 `code_review/`。

## 已確認 repository 基線

2026-10-02 修訂前的唯讀觀察，不是新功能驗收：
- active root `/home/minervamuses/drone-image-analysis-lab`；`main`，HEAD `7a7c6ee8859754d5162bec032133b5bd7d2a103e`。AGENTS 內沒有 -lab 的路徑是舊位置。
- Windows 主機透過 `wsl.exe -d Ubuntu-24.04` 使用 Bash／Git／Python 3.12.3。專案尚無 `.venv`；Python >=3.12、torch 2.11.0、torchvision 0.26.0、Spandrel 0.4.2、Pillow 12.3.0；WSL requirements 為 CUDA 12.8。
- 一般入口只收單一 `--model`，載入器只收 SR scale>1。evaluation 現為 bicubic 1/4 再 4× 重建；入口 top-level import LPIPS；lab 預設注入 `--all`。
- 本機只有原 Compact 權重，新增模型目錄及評測資料目錄為空；不必在本機補真權重。
- 指定 metrics 的純核心依賴 NumPy、OpenCV、scikit-image；高階 API 會寫 CSV、import pandas、預設多執行緒，不能直接照搬。
- 官方 Spandrel 核心支援 FFTformer／NAFNet／Uformer；官方 extra_arches 註冊 Restormer／MPRNet。來源查閱不等於實體權重實跑。
- 整份 evaluation/test_runner.py 會初始化 LPIPS，test_sr_line.py 部分案例載真權重；不能直接當無模型本機全套。

修訂前工作樹，保留非本計劃改動：
```text
 M README.md
 M src/drone_sr/image_io.py
 M tests/test_image_io.py
?? .dockerignore
?? Dockerfile
?? colab/
?? evaluation/data/
?? evaluation/runs_extracted.json
?? evaluation/runs_summary.md
?? sr-deblur/
```

## 使用者授權與 Git 規則

### 授權來源與效力

2026-10-02 使用者明示：「計劃中不要停下來等我裁決，執行即代表授權一切」，且「每次改動皆 commit，只禁止 push、切 branch」。
此為本計劃範圍內的預先授權，優先於 AGENTS／skill 中要求此類工作另行詢問的一般規則；不修改 AGENTS。啟動後代理自行完成必要技術選擇與修正，不再次詢問指標、CLI、檔案數、依賴、環境、commit 或例行測試許可。
本次請求是修訂計劃，不是啟動 Phase 01；本輪計劃文件也依每次改動 commit 的要求提交。

### 已授權的本機工作

- 修改本計劃涉及的正式程式／測試／文件，含超過三個正式檔、必要 CLI／評測資料結構及報告格式變更；優先用現有模組。
- 自行建立／修復專案 Linux Python 3.12 `.venv`，按 README 的已固定版本與現有套件管理流程安裝依賴、更新直接相關 requirements／pyproject；不借用不符合專案 Python 版本的其他環境。
- 保留 Spandrel 0.4.2，加入 `spandrel_extra_arches==0.2.0`；其套件 metadata 要求 Spandrel >=0.3.2。完成必要註冊與無預訓練權重的支援檢查。
- 加入評測需要的 `opencv-python-headless` 與 `scikit-image`，沿用既有 NumPy／SciPy；版本由執行時相容性與 pip check 決定並固定。不要為搬整套工具新增 pandas 或另一個並行 runner；legacy 既有依賴保留。
- 可新增一個專用 `evaluation/blur_metrics.py`、對應最小 unittest 檔及 CPBD 授權檔，移入必要量測核心；這是具體四指標需求，不建立指標 registry。
- 便宜本地檢查、依賴下載／安裝即使超過十分鐘亦已授權；先告知實際預估與資源，不以此等待回覆。此授權不改變本輪「本機不下載預訓練權重、不做真模型／GPU 實驗」的分工。
- 維護本 bundle、必要記錄及相關 README／evaluation README／lab 說明；Docker 只更新受 CLI 直接影響的用法與未驗證標示。
- 所有命令使用 WSL 工具鏈，全部變更在目前分支完成。

### 每次改動皆 commit

「一次改動」是可獨立說明的一批程式／測試／文件變更，不是每次按鍵。每批完成相應驗證並更新紀錄後立即 commit，再開始下一批；修正、後續文件／紀錄變動也各自提交，不能只留到全部結束一次提交。
commit 僅納入自己這批的檔案／hunk，先核對 staged diff；不得 `git add .` 把使用者既有改動、素材或權重一起提交。和既有修改共用檔案時，以 scoped staging 保留其原有 unstaged 差異。
程式失敗的中間改動若需保存，commit 必須明示 WIP／失敗，phase 不得標 Complete。不要為了紀錄 commit 自身 SHA 反覆產生紀錄 commit；可由 Git 歷史和後一筆紀錄追溯。
push 和切換分支明確禁止，由使用者自行 push，lab 自行 pull；本機代理不替使用者操作 lab，也不因此停下等遠端結果。

## 界線、失敗與自動恢復

- 無指標／GT／模型名稱／一般技術選擇的待裁決 gate，無逐 phase 審批 gate；lab 權重、圖片與 GPU 缺席不阻擋本機開發驗收。
- 真實 checkpoint 與資料由使用者在 lab 準備；本機只跑 synthetic／mock 和四指標的小圖檢查，不改成下載真權重或 CPU 長跑。
- 真正執行障礙（WSL 不可用、套件無法取得、權限／commit 身分不可用、必要測試失敗）先自行調查與作範圍內修正，不當成缺使用者授權。保留證據；可獨立完成的工作继续。
- 沿用兩次聚焦修正仍失敗就停止該路徑擴大的成本界線；修訂被證據推翻的技術做法，未解的必要檢查仍阻擋依賴它的 phase。若確實無法完成則誠實回報未完成與原因，不假裝成功，也不以重問許可代替診斷。
- 不因「授權一切」擴張為別的專案、破壞素材／既有修改、付費服務、模型搜尋、一般架構重寫或繞過更高層工具限制；這些不是達成本計劃所需工作。push／切 branch 不以其他 Git 操作繞過。
- 所有失敗以 scoped 修復恢復，不 reset／clean 使用者工作樹、不清掉舊 run。

## 階段路線圖

| Phase | 可觀察成果 | 依賴 | 文件 |
|---|---|---|---|
| 01 | 可用本機環境；一般 CLI 三模式／有序串接／候選架構支援 | 無 | [phase-01-modes.md](phases/phase-01-modes.md) |
| 02 | evaluation 可選模式，本次 lab 固定 deblur；資料／輸出／狀態可追溯 | 01 | [phase-02-evaluation.md](phases/phase-02-evaluation.md) |
| 03 | 四指標、兩份報告、本地整合與 lab 操作交付 | 02 | [phase-03-report-and-acceptance.md](phases/phase-03-report-and-acceptance.md) |

所有 phase 的必要驗收都是本機可取得的證據；真權重相容性、lab 資源與畫質另列交付後驗證，不設第四個阻擋開發交付的 phase。
01／02 不必等候真實模型；02 的暫時未量測記錄由 03 完成四指標，不再等待使用者選指標。
每 phase 必要檢查通過、紀錄並 commit 後，自動選下一個有效 phase。不得把例行進度回報當結束點。

## 計劃維護與完整驗證

從 build-log 找第一個依賴完成且本身未 Complete 的 phase；不設另一份 current-phase。必要檢查失敗先修，不帶著錯誤往下走。
技術假設失效時自主修訂本檔與未開始 phase，保留穩定目標與既有觀察。舊事件追加更正，不能抹掉失敗證據。
本機代表整合需小圖片＋synthetic 非交換模型，串過正式 CLI／evaluation／安全輸出／真實四指標／兩份報告。這證明軟體路徑，不證明預訓練模型畫質。
全套只在相關且便宜時收尾一次；涉及真權重或 LPIPS 下載者改跑明確無模型部分並列限制，不擅補下載。

## 整體完成標準

- [x] 三 phase 在 build-log 為 Complete，有確切命令與成功／失敗邊界證據。
- [x] GOALS 的本機交付成功條件均覆蓋；deblur-only 不依賴 ATD、GT 或 LPIPS，SR／combine 選擇與順序有測試。
- [x] 四指標與逐張／彙總契約、兩報告、模式隔離及舊行為回歸有結果。
- [x] 所有本次變更已有分批 commit，未 push、未切 branch，既有無關修改仍保留。
- [x] lab 命令、模型／素材放置、依賴安裝、首張小測及失敗原因可由使用者照做；未知 lab 資源與下載可用性清楚揭露。
- [x] 完成聲明只寫「本機架構、測試與交付完成」，不寫真實五模型／GPU／去模糊效果已驗收。無需等待使用者 pull／下載／回傳分數。
