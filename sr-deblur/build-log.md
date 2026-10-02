# SR／deblur — Build Log

本檔是 runtime phase 狀態及實際實作／驗證證據的唯一來源。未來命令與驗收要求在 phase 文件，不當成已執行結果。

## 階段摘要

| Phase | 狀態 | 開始 | 完成 | 證據 | 阻塞 |
|---|---|---|---|---|---|
| 01 — 本機環境與三模式 | Complete | 2026-10-02 | 2026-10-02 | 下方 Phase 01 51 項本機測試及 pip check | 無；真模型另待 lab |
| 02 — 可選模式與 deblur lab | Complete | 2026-10-02 | 2026-10-02 | 下方 62 項必要／新 runner 檢查及 shell stub | 無；指標接入於 03 |
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

### 2026-10-02 — Phase 01 Complete：環境、有序 CLI 與候選載入支援

- 實際 checkout：/home/minervamuses/drone-image-analysis-lab，main，開始 HEAD b86b879；Windows agent 透過 wsl.exe -d Ubuntu-24.04，Linux Python 3.12.3／Git 2.43.0。AGENTS 舊根路徑不沿用。
- 先保存既有修改／未追蹤檔案的 SHA-256 與備份於 /tmp/sr-deblur-baseline-3p7_dbr3。README 僅 stage 本次新增區塊；image_io.py、test_image_io.py、Docker／Colab／素材／歷史分析不 stage。
- 新建 .venv。先依 README 安裝 torch 2.11.0+cu128／torchvision 0.26.0+cu128，再安裝 requirements-wsl.txt 與 extra arches 0.2.0，editable install。PyPI wheel 下載慢，兩次受控中止下載後重用完整 wheel；Triton 同版由 download.pytorch.org 官方來源取得，SHA-256 6f5928e6d44c34a97bbe164cceddc0ef2007121c89ebcfba5415cf452de7ee9f。最終以 /tmp/sr-deblur-wheels/requirements-local.txt 安裝相同固定版本／核對 NVIDIA 既定 hashes，沒有換版本或預訓練權重下載。
- 安裝命令結尾：.venv/bin/python -m pip install --no-deps --no-build-isolation -e .；.venv/bin/python -m pip check → No broken requirements found。NumPy 2.5.3、Pillow 12.3.0、Spandrel 0.4.2。
- 一般 CLI 新增有序 --sr／--deblur、--sr-model（--model 別名）、--deblur-model；只載啟用模型。共用 run_stages 直接串接 tensor、檢查每阶段 shape／finite；同名／source alias／checkpoint alias 保護，第二階段失敗不覆寫舊成功檔。
- 讀安裝版本確認 extra install(ignore_duplicates=True)、五候選 registry，FFTformer multiple_of=32、Uformer multiple_of=128/square、Restormer/MPRNet multiple_of=8。官方 descriptor 自行 padding／cropping，clamp 後做階段邊界 finite 檢查；不繞過 descriptor。
- 實際唯讀程式審查找到：把 DISCOURAGED 一律 direct 會改變既有 SwinIR SR 分塊。已限縮新 tiling 規則至 x1 restoration，保留 SR 512 core／32 halo，新增 synthetic 回歸。
- 必要驗證（CUDA_VISIBLE_DEVICES 為空，OMP_NUM_THREADS=2、MKL_NUM_THREADS=2）：
  - PYTHONPATH=src:tests .venv/bin/python -m unittest test_cli test_inference test_tiling → 39 tests OK。
  - PYTHONPATH=src:tests .venv/bin/python -m unittest test_image_io → 12 tests OK（含使用者既有 MPO 測試／未提交差異）。
  - .venv/bin/python -m drone_sr --help、git diff --check → 通過。
- 以上是 CPU tiny synthetic／mock 與安全寫檔證據；沒有真權重／GPU 推論，沒有檢驗五模型畫質或分塊接縫。一般 CLI／Docker 用法已在 README 前段同步，Docker 未重建。
- 本批 commit：feat: add ordered SR and deblur inference modes（SHA 由 Git 歷史追溯）。完成提交後自動開始 Phase 02。

### 2026-10-02 — Phase 02 Complete：可選 evaluation 與 deblur-only lab

- 接續 Phase 01 commit 35522e6，main／相同 WSL .venv，沒有切 branch 或 push。
- 新 evaluator 明確選一條有序模式；N=2 時 deblur／任一 combine 各 2 組、SR-only 1 組。SR 必須選單顆，deblur 可選單顆或第一層排序／resolve 去重掃描；sample 每 run 固定一次。
- 新 runner 使用共用解碼／tensor inference／PNG，不讀 GT、不降採樣、不匯入 LPIPS。模型逐組釋放、保存逐模型／逐張錯誤與正確非零退出碼；ID 含順序與 resolved checkpoint 路徑／SHA，拒絕同 stem 輸入輸出衝突，run/組目錄均不重用。
- report.md／per_image.md 同源互鏈、列模型身分、模式、輸入／輸出與失敗；四指標仍明確待量測，非假數據。本階段不宣稱四指標已驗收。
- lab/run.sh 固定 --deblur，預設 evaluation/data/input/／limit=1；允許單顆 checkpoint/input/limit/seed/runs-root，拒絕其他模式與縮寫。保留 CUDA/pip preflight，加入可取得資源資訊。legacy 僅 --legacy-sr，models/sr/，舊 --model/--all 語義保留。
- 安裝 evaluation/requirements.txt 的既有 lpips 0.1.4／SciPy 1.18.1／tqdm 4.70.1。單連線下載慢，SciPy 使用同一官方 PyPI 檔案分段取得並核對官方 SHA-256，再從本地完整 wheel 安裝；沒有下載 AlexNet 或 SR/deblur checkpoint。
- 初次入口檢查：2 項 legacy 測試因 lpips 尚未安裝而 error，另修正 sparse mock 缺少組 ID 的測試資料；新模式/lab 11 項通過。完成依賴後重新跑必要集合，未把先前失敗視作成功。
- 最終必要檢查（CUDA_VISIBLE_DEVICES 空、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2）：
  - PYTHONPATH=src:evaluation .venv/bin/python -m unittest test_run_evaluation test_runs test_sources test_report test_runner.SelectSourcesTests test_runner.DeviceMemoryTests test_runner.OrderedBatchTests test_sr_line.ScaleAssertionTests → 62 tests OK，含真實 shell stub 參數檢查。
  - .venv/bin/python -m pip check → No broken requirements found。
  - bash -n lab/run.sh、evaluation/run_evaluation.py --help、git diff --check → 通過。
  - CUDA_VISIBLE_DEVICES= bash lab/run.sh --limit 1 → 預期 exit 1／CUDA unavailable，在推論前停止，無 CPU fallback。
- fresh-agent 對 selection/count/order、兩報告骨架、檔案安全、失敗隔離、LPIPS 分離及 lab 參數做唯讀程式審查，無新增阻擋發現。
- 本批 commit：feat: add selectable evaluation and deblur-only lab entry。README 仍只 stage 本次差異；既有使用者 I/O／測試、Docker／Colab／素材／歷史分析未納入。
- 本地測試均 synthetic／mock，沒有真權重或 GPU inference；完成提交後自動開始 Phase 03 四指標及交付。
