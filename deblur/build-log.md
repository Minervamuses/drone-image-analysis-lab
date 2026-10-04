# deblur — 執行紀錄

本檔唯一持有執行狀態與觀察證據；預定工作見 PLANS.md 與各 phase。

## 階段狀態

| 階段 | 狀態 | 開始 | 完成 | 證據 | 阻礙 |
| --- | --- | --- | --- | --- | --- |
| 01 — 分支與基準回退 | Complete | 2026-10-04 | 2026-10-04 | fix 保存計劃；基準檔案差異為空；回退範圍與保留檔核對通過 | 無 |
| 02 — 單次實驗腳本 | In progress | 2026-10-04 | — | 已完成入口與單次分組/前處理的語法及 help 檢查 | 模型/全參考 CSV 步驟尚待實作 |
| 03 — lab 執行 | Not started | — | — | — | lab 資料、權重與執行證據尚未取得 |

## 活動與證據

初始狀態僅有計劃文件；後續執行證據按時間追加。不得把計劃文件的結構檢查當成實驗驗收。

執行後才追加日期（Asia/Taipei）、步驟、實際修改、命令與結果、commit 主旨、失敗／未驗證項目及產物路徑。狀態使用 Not started、In progress、Blocked、Complete。第三階段的數據分析由使用者自行進行；此紀錄只確認執行情況。

### 2026-10-04 — Phase 01 / 步驟 1：保存計劃

- 唯讀確認：repo `/home/minervamuses/drone-image-analysis-lab`；WSL Ubuntu-24.04，Linux 6.6.87.1；Git `/usr/bin/git`；現有 `.venv/bin/python` 為 Python 3.12.3，pip 亦屬該 Linux venv。
- 起始 `git status --short --branch` 為 `main...origin/main`，只有本次 `deblur/` 未追蹤；HEAD `33b396019a9294efbda02a440d9a3b68eac69568`，父提交 `8766f1d966c2c2b95e7598ff32871ce5bb2e9476`；fix 尚不存在，無既有修改需覆蓋。
- 已讀適用 AGENTS.md、GOALS、PLANS、build-log 與 phase-01；沒有既存 context/code_review。`rg` 不在 PATH，使用既有 `find`/Git 作最小探索，未安裝工具。
- 已執行 `git checkout -b fix`；保存七份既有計劃文件及此紀錄。提交前執行 `git diff --cached --check`。
- Commit 主旨：`docs: record minimal deblur experiment plan`。回退及模型執行尚未進行。
- 初次唯讀檢查的 shell 變數傳遞造成讀檔失敗（`cat: '': No such file or directory`）；改用明確路徑後已完整讀取，未更動專案檔案。

### 2026-10-04 — Phase 01 / 步驟 2–3：回退基準

- 計劃提交為 `cfd8557`；提交後 `git status --short --branch` 為乾淨的 fix，main 保留。
- 再次核對 repo、Linux Git/Python、Git 狀態與 33b3960 的檔案清單後，執行 `git revert --no-commit 33b396019a9294efbda02a440d9a3b68eac69568`，無衝突。
- 依 phase-01 使用 `git restore --source=33b3960 --staged --worktree` 保留三個資料 `.gitignore`、`evaluation/runs_extracted.json` 與 `evaluation/runs_summary.md`；未碰圖片、權重、venv 或歷史 runs。
- 實際回退：刪除該次新增的 `.dockerignore`、`Dockerfile`、`colab/drone_sr_colab.ipynb`；還原 `README.md` 與 `lab/run.sh`。不是整棵 tree 回退；保留上述五個資料路徑與 deblur 計劃。
- `git diff 8766f1d -- lab/run.sh src evaluation/metrics.py evaluation/perceptual.py evaluation/blur_metrics.py evaluation/LICENSE-CPBD.txt pyproject.toml requirements-wsl.txt` 無差異；共用載入/推論、MPO 修正、全參考/清晰度指標與 CPBD 授權均保留。
- `git diff --cached --name-status` 僅有五個預定回退檔案；`git diff --cached --check` 通過。此紀錄納入同一回退提交，未跑模型或完整測試。
- Commit 主旨：`revert: restore pre-expansion baseline and preserve experiment data`。提交後核對分支、乾淨工作區與 git log，再交接 phase-02。

### 2026-10-04 — Phase 02 / 步驟 1–2：入口與前處理

- 第一階段回退提交為 `935b6c0`；提交後 fix、乾淨工作區與三筆 git log 已確認。讀 phase-02，重新確認 WSL repo、Git、Python 與當前 `lab/run.sh` 後才實作。
- 唯一正式程式修改為 `lab/run.sh`：既有 venv 執行嵌入 Python；只接受 `--limit N`/help；名稱排序後 seed=923 洗牌、limit 後循環分組，每圖只產生一張原尺寸 PNG，保留原副檔名。
- 重用 `read_image`/`write_png`/`measure_tensor`。sRGB/linear 公式與質心置中的 bilinear rasterization 取材自 Git 中舊入口；固定 8px/45° 直線、90°/8px 圓弧與 σ=2/13×13 Gaussian，反射邊界純模糊，none 不改解碼 RGB；沒有裁切、額外雜訊、反光增强或再壓 JPEG。
- GT 清晰度每圖一列，保留局部缺值與 CPBD 原值/invalid；前處理錯誤保留原分組與原因。全參考 CSV 與模型循環將在下一個修改步驟完成，此時未宣稱完整入口可跑實驗。
- 已讀既存相關 `evaluation/context/phase-04-context.md`：沿用 CPU float64 PSNR/SSIM 與完整尺寸 LPIPS 的既有成本限制，不改算法。
- 本機 `evaluation/data/input/` 只有 `.gitignore`，`models/deblur/` 為空；未下載、未 import ML 套件作驗收、未跑圖片/GPU/mock/全套測試。
- 檢查：`bash -n lab/run.sh`、既有 `.venv/bin/python` 的 heredoc `ast.parse`、`bash lab/run.sh --help`（無資料/權重仍正常顯示）、`git diff --check` 均退出 0；已閱讀 diff，沒有其他正式程式變更。
- 編輯工具第一次拒絕同一路徑的 delete/add patch（未更動檔案）；改用單一 update patch 後成功，非模型或程式檢查失敗。
- Commit 主旨：`feat: prepare evenly assigned deblur inputs`。模型載入、推論、保存/度量與三份最終 CSV 尚未驗證。
