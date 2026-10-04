# deblur — 執行計劃

## 概覽與資料歸屬

計劃根目錄：`deblur/`。穩定要求見 [GOALS.md](GOALS.md)；本檔只管順序、授權與完成界線。採 **授權範圍內自主執行**，本機完成可做的工作後交由使用者在 lab 跑一次正式實驗。專案形態 minimal；風險 medium，原因是有 Git 基準回退與本機無法實證的 GPU 路徑。

[build-log.md](build-log.md) 唯一持有執行狀態／觀察證據；各 `phases/` 文件持有步驟與最小檢查；[PROMPTS.md](PROMPTS.md) 提供啟動／續作入口。只有發現會改變後續做法的事實才建 `context/phase-NN-context.md`；只有實際做審查才建 `code_review/phase-NN-review.md`，均非例行必備文件。

## 已確認基準

規劃時唯讀檢查：2026-10-04，repo 為 `/home/minervamuses/drone-image-analysis-lab`，分支 main，HEAD `33b396019a9294efbda02a440d9a3b68eac69568`，工作區乾淨，fix 分支不存在。WSL Ubuntu-24.04 的 Git 為 /usr/bin/git，現有 .venv 使用 Python 3.12.3。

回退目標為 `8766f1d966c2c2b95e7598ff32871ce5bb2e9476`，即 33b3960 的父提交。此時 lab/run.sh 為 64 行，已具有：
- 35522e6 的 deblur/Spandrel extra arches 載入。
- c6a8ffc 的四項指標、CPBD 授權與 OpenCV/scikit-image 依賴。
- 8766f1d 的 DJI MPO 主影格修正，以及原有 PSNR/SSIM/LPIPS。

因此不退到尚無上述能力的 d805d9e。新腳本可以直接 import 現有工具，不呼叫膨脹的 runner/report 邏輯。lab 真實安裝狀態尚未確認。

## 執行授權

本次 authoring 只建立此計劃。使用者啟動執行後，已授權：
- 在 WSL 建立／checkout 名為 `fix` 的新分支，依第一階段回退；保留 main。
- 每個有意義的修改步驟，最小檢查後立即 commit；包含初始計劃、回退、實驗腳本及 lab 結果紀錄。只讀動作不需製造空 commit，不跨多個修改步驟累積提交。
- 直接覆寫 lab/run.sh，重用既有函式，更新本計劃紀錄，執行列明的便宜檢查；不要求每個 phase 再批准。
- 所有 Git 命令在本機 WSL 執行；Windows 只作 wsl.exe 呼叫入口。計劃內 bash 指令皆以 repository root 為工作目錄。
- 使用者於本次執行另已明確要求「幫我push」；本次 fix 交付可推送至既有 `origin` 的 `fix`，供 lab 同步。只做正常 fast-forward push，不 force push、不改 main/upstream，也不擴張成其他遠端操作。

只有下列情況需要新決策或停止：
- fix 已存在且來源不明、HEAD 已變動導致回退範圍不同，或存在會被覆蓋的使用者修改；先報具體衝突，不能強制重設。
- 需要修改 AGENTS.md、擴張正式程式範圍、增修依賴／manifest、縮圖裁切、換模型、導入新架構或改變 GOALS 的實驗定義。
- 除上述明確授權的 `origin/fix` push 與其既有 Git 認證外，merge、其他遠端寫入/憑證用途、下載新權重／資料、付費呼叫及刪除使用者資料均未授權。
- 代理代跑未知或超過約十分鐘的 GPU／全量工作：先取得 lab 的圖數、資源與短試跑成本，告知預估後取得執行同意。使用者可自行執行所列 lab 命令。
- 兩次聚焦修正仍失敗，或一次昂貴嘗試無效；記錄實際結果，不自行改成大規模實驗。

## 階段路線圖

| 階段 | 結果 | 依賴 | 計劃 |
| --- | --- | --- | --- |
| 01 | fix 分支保存計劃，回到膨脹前程式基準並 commit | 無 | [phase-01-baseline.md](phases/phase-01-baseline.md) |
| 02 | 單一 lab/run.sh 跑通四組、所有模型與三份 CSV；完成本機最小檢查及 commit | 01 | [phase-02-experiment.md](phases/phase-02-experiment.md) |
| 03 | lab 短試跑、一次正式實驗及原始結果紀錄 | 02 | [phase-03-lab-run.md](phases/phase-03-lab-run.md) |

第三階段不能以本機 mock 冒充完成。資料／server 尚不可用時，本機交付可完成，第三階段明列等待 lab 執行證據，不因此追加本機下載或框架。

## 2026-10-04 — NAFNet 授權跳過

使用者明確要求跳過 NAFNet 並繼續。後續入口在 checkpoint 清單中排除檔名 `NAFNet-GoPro-width64.pth`（含子目錄），stdout 記錄已知輸出異常與使用者授權；其餘 checkpoint、分組、原尺寸與三份 CSV 約定照常。這是明確授權的單一模型例外，不新增選模介面或自動淘汰規則。

取消 NAFNet 單圖精度對照/根因定位作為下一步，已知兩張彩色方塊證據與原始五模型 run 保留。先完成入口排除、最小語法/help/差異檢查並立即 commit；補看其餘模型的少量既存輸出後，由使用者在 lab 執行一次正式全量。NAFNet 未修復不再阻擋正式實驗，但全量結果尚未取得時 Phase 03 仍不能標 Complete。

## 計劃維護與驗證尺度

採語法檢查、必要的差異閱讀，以及 lab 四張圖的一輪短試跑。不上完整測試套件、不新增測試檔／fixture／benchmark 框架、不要求嚴格 TDD、統計驗收或每階段獨立審查。

必要檢查失敗時只修當前步驟；失敗不得寫成通過。證據推翻未開始步驟時，先改本檔和受影響 phase；穩定目標只能由使用者更改。build-log 保留失敗與修正紀錄，不覆寫歷史。每步 commit 主旨與檢查寫入 log，SHA 由 git log 查核，避免為記錄自身 SHA 反覆提交。

## 整體完成

所有階段在 build-log 有觀察證據並標 Complete；GOALS 的成績資料能對應真實輸出。個別模型失敗如實保留，不自動淘汰或宣稱成功；使用者授權跳過的 NAFNet 保留異常與授權紀錄，不要求根因定位或修復。沒有任何有效模型結果時，第三階段不能宣稱實驗成功完成。未取得 lab 證據時只能回報「程式已交付、實驗待跑」。本次實驗與紀錄完成就停止。
