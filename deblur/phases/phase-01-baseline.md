# Phase 01 — fix 分支與基準回退

## 目標、範圍與非目標

在 fix 保存本計劃，再用可追蹤 commit 回到 8766f1d 的程式基準。此階段不寫新實驗、不改 AGENTS.md。依 [GOALS](../GOALS.md) 保留資料、checkpoint 與歷史結果，Git 授權見 [PLANS](../PLANS.md)。

## 前置與唯讀確認

讀取 ../build-log.md，唯讀核對 WSL repo、git status、分支與 git log。首次啟動應為 main@33b3960、fix 不存在，且只有本次 deblur/ 計劃待提交；若只多了使用者已提交的本計劃，辨識後不重複提交。

中斷續作時，若已在來源可確認的 fix，按實際 Git 狀態接續：計劃已 commit 就跳過步驟 1；回退已暫存／進行中就檢查其差異與保留檔，接續處理，不重做 revert；回退已 commit 且 lab/run.sh 等於基準就只補驗收與紀錄。來源不明的既有 fix、額外變更或衝突先回報。禁止 reset --hard、git clean 或強制 checkout。

## 實作步驟與 commit

以下命令在 WSL 的 repository root 執行；從 Windows 呼叫時以 wsl.exe -d Ubuntu-24.04 --cd 搭配實際 repo 路徑，不使用 Windows Git。

1. 建立分支，將 build-log 的此階段標 In progress 並記錄已觀察的分支狀態，保存計劃，形成第一個追蹤點：

```bash
git checkout -b fix
git add -- deblur/
git commit -m "docs: record minimal deblur experiment plan"
```

2. 撤回大型實驗 commit，但保留其資料忽略檔與歷史數據：

```bash
git revert --no-commit 33b396019a9294efbda02a440d9a3b68eac69568
git restore --source=33b3960 --staged --worktree -- \
  evaluation/data/input/.gitignore \
  evaluation/data/ground_truth/deblur/.gitignore \
  evaluation/data/ground_truth/sr/.gitignore \
  evaluation/runs_extracted.json \
  evaluation/runs_summary.md
```

這會回退 .dockerignore、Dockerfile、README.md、colab/drone_sr_colab.ipynb、lab/run.sh 的該次變動；不另行調整 Docker／Colab。保留五個上述資料相關路徑與 deblur/ 是明確例外，因此不宣稱整棵 tree 逐位元等於 8766f1d。未追蹤的圖片、權重、.venv 與舊 runs 不動。

3. 完成下列檢查、把實際結果寫入 build-log，立即提交回退：

```bash
git add -- deblur/build-log.md
git diff --cached --check
git commit -m "revert: restore pre-expansion baseline and preserve experiment data"
```

## 最小驗證與驗收

- `git diff 8766f1d -- lab/run.sh` 無差異；確認共用 loader、metrics、perceptual、blur_metrics、CPBD 授權與 MPO 修正仍在。
- `git diff --cached --name-status` 在提交前符合上述回退範圍及本計劃紀錄；不用完整測試。
- commit 後 `git branch --show-current` 為 fix，`git status --short` 乾淨，`git log -3 --oneline` 可追溯。
- 若 revert 衝突，先停在此步；可用 `git revert --abort` 回到回退前狀態，但保留已提交計劃。不要靠 reset 刪除變更。

## 證據與交接

將命令、實際變更範圍、commit 主旨與驗收結果寫入 ../build-log.md。達到以上觀察後標 Complete，交給第二階段；沒有實作驗收前不前進。不建立獨立回退框架或例行 review 文件。
