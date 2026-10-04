# deblur — 執行紀錄

本檔唯一持有執行狀態與觀察證據；預定工作見 PLANS.md 與各 phase。

## 階段狀態

| 階段 | 狀態 | 開始 | 完成 | 證據 | 阻礙 |
| --- | --- | --- | --- | --- | --- |
| 01 — 分支與基準回退 | Complete | 2026-10-04 | 2026-10-04 | fix 保存計劃；基準檔案差異為空；回退範圍與保留檔核對通過 | 無 |
| 02 — 單次實驗腳本 | Complete | 2026-10-04 | 2026-10-04 | 兩個實作步驟；Bash/Python 語法、help 與差異檢查通過 | 真實推論/度量/CSV 內容留待 lab 驗證 |
| 03 — lab 執行 | In progress | 2026-10-04 | — | 已有四張短試跑/console；NAFNet 兩圖異常；使用者明確授權跳過並繼續 | 待入口排除 NAFNet、其餘代表圖核對與 lab 正式全量結果 |

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

### 2026-10-04 — Phase 02 / 步驟 3–5：模型與原始成績表

- 前處理提交為 `de2a80b`，提交後工作區乾淨。再次唯讀確認 repo、WSL Git/Python、Git 狀態與當前入口後，繼續同一正式程式檔；未改套件、共用模組或測試。
- 遞迴排序 `models/deblur/` 的 .pth/.pt/.ckpt/.safetensors；逐 checkpoint 使用既有 `load_model(..., role="deblur")`/`upscale` 處理完全相同的 records 與已保存輸入。沿用既有尺寸/tiling 行為，確認輸出等於输入形狀，沒有自動縮圖或裁切。
- checkpoint 相對路徑含副檔名成為輸出模型目錄，圖片原名稱含副檔名再加 `.png`；每次新建帶微秒的 UTC 目錄且禁止覆蓋既有 run。輸入、原圖、權重與歷史結果不改寫。
- 執行前確認既有依賴、CUDA、Torch AlexNet cache 與 lpips 套件內 v0.1/Alex 校準權重；缺失明確退出。下載函式在本入口禁用，不自動安裝或 CPU 推論。裝置、VRAM、RAM/cgroup 與實際圖片尺寸/時間只在 lab 執行時顯示。
- 每個模型先保存所有輸出、釋放 deblur 模型/tensor，再建立既有 `PerceptualMetric(cuda:0)`。重讀實際保存的 PNG 與 `read_image` 解碼的原圖，完整尺寸計算 CPU float64 RGB PSNR/SSIM 與 GPU LPIPS；RGB 值 round 到既有 [0,255] 約定。
- `full_reference.csv` 保留每圖/模型對應、前處理/載入/推論/度量失敗與已取得的局部分數；各項全參考度量獨立記錯，後续模型仍處理。`summary.csv` 以三項皆取得的共同成功列作每模型算術平均，明列 expected/success/failed；只要有失敗即 partial，零成功均值留空。有效 PSNR=inf 與其平均保留，另有 psnr_inf_count。
- `sharpness.csv` 只有欄位與原始資料；GT 一次、model output 各一次；保留 CPBD 無邊緣時的原值與 invalid/reason，其餘缺值不補零；不呼叫比較/分析/摘要函式，也不影響有效全參考成績。不存在的輸出以 failed 列保留原因。
- 腳本只寫三份成績 CSV，保存中間輸入和輸出 PNG；每模型完成更新三表，已記錄失敗不清空其他模型结果。任一全參考失敗時最後退出 1；有完整成功結果則依資料呈現，不自動排名或宣稱改善。
- 檢查：`bash -n lab/run.sh`、heredoc `ast.parse`、`bash lab/run.sh --help`、`git diff --check` 均通過；已閱讀模型、配對、均值與 CSV 的 diff。差異閱讀發現零成功摘要亦應 partial，提交前已修正並重跑 AST/Bash 語法與 diff check（均退出 0），沒有程式檢查失敗。
- 唯一正式程式修改仍是 `lab/run.sh`，另更新本 log。沒有新增框架、context/review 文件、下載、安裝、GPU/mock 或全套測試；本機沒有實際圖片/模型輸出/成績 CSV，不能宣稱實驗完成。
- Commit 主旨：`feat: run one minimal deblur experiment with raw metric tables`。第二階段只達到計劃的本機程式交付門檻；真實 checkpoint 相容性、尺寸、OOM、LPIPS cache 可用性及實際成績尚未驗證。

### 2026-10-04 — Phase 03：等待 lab，停止於計劃交付界線

- 第二階段最終程式提交為 `e111517`，提交後 fix 工作區乾淨。依依賴順序讀 phase-03；本階段沒有實際啟動 lab 短試跑或正式實驗。
- 再次唯讀確認 WSL repo、Git/Python、Git 狀態/提交及當前輸入/模型檔案；`find evaluation/data/input models/deblur -type f` 只有輸入 `.gitignore`，即本機指定資料 0 張、deblur checkpoint 0 個。本次沒有使用者提供的 lab 執行管道，不以其他本機資料或權重替代。
- `git rev-parse main` 仍為 `33b396019a9294efbda02a440d9a3b68eac69568`。五個指定保留資料路徑相對 33b3960 的差異為空；src、全參考/清晰度共用程式、CPBD 授權、AGENTS.md 與 manifests 相對 8766f1d 的差異亦為空。
- lab 同步本次已提交程式、準備既有 venv/資料/checkpoint/LPIPS cache 後，於 repository root 先執行 `bash lab/run.sh --limit 4`。此命令尚未實測；保留 console 的 run 路徑、裝置/資源、圖尺寸、耗時及失敗原因，核對三 CSV 與少量原尺寸實際圖片。
- 一輪短試跑的資源/成本、CSV 關聯與代表输出確認後，由使用者執行一次 `bash lab/run.sh`。全量命令尚未實測；圖數、checkpoint 數、lab GPU/VRAM/RAM/磁碟、套件/cache 與總耗時未知，未替使用者估算或代跑。
- 待取得 lab 證據後再追加短試跑/正式結果並依 phase-03 提交；不把語法/help 通過、退出碼或本機程式交付當成實驗完成。本階段保留 Blocked，沒有有效模型結果可供分析。
- 本機交付完成，按 PLANS 的缺資料/server 停止界線停下；沒有 push/遠端寫入、下載、安裝、額外測試或 GPU 工作。
- Commit 主旨：`docs: record pending lab deblur execution`。本次結論為「程式已交付、實驗待跑」。

### 2026-10-04 — Phase 03：收到並核對四張 lab 短試跑

- 使用者提供 `/mnt/c/Users/garyc/Downloads/20261004T124643.486598Z`，並明確確認是 `--limit 4` 短試跑，不是正式全量。run 名稱標示 UTC 2026-10-04 12:46:43.486598，即 Asia/Taipei 20:46:43.486598；CSV 的 lab 原路徑為 `/home/gary/test/evaluation/runs/deblur/20261004T124643.486598Z/`。
- 續作唯讀確認：同一 WSL repo、Linux Git/Python、fix@`fb33d1a`，工作區乾淨；重新讀 AGENTS、GOALS、PLANS、build-log 與 phase-03。lab 實際 checkout SHA、GPU、VRAM/RAM/磁碟、套件版本與總耗時沒有隨結果回傳，未以檔案時間推估成本或假定硬體。
- 原始交付只有三份 CSV、4 張 inputs PNG、20 張 outputs PNG；没有原始 JPG、checkpoint 或 console。所有來源只讀，未搬檔、改寫成績、重跑模型或下載。本機 `models/deblur/` 仍無 checkpoint。
- `full_reference.csv` 20 列、`summary.csv` 5 列、`sharpness.csv` 24 列（4 GT + 20 model_output）；欄位符合 GOALS，沒有說明段落。全參考 20 列與清晰度 24 列皆為 ok，error/invalid_metrics 空白，三項分數及四項指標皆有值；本次無 inf 或失敗列可驗證特殊情況。
- 四組各一張：0881.JPG=none、0002.JPG=linear、0696.JPG=trajectory、0997.JPG=gaussian；五個模型的 image/condition/blur_params/original_path/input_path 完全相同，各模型恰好 4 個 image，沒有重複 model-image。模型為 NAFNet-GoPro-width64.pth、Uformer_B_GoPro.pth、fftformer_GoPro.pth、model_deblurring.pth、motion_deblurring.pth。
- 以 WSL 現有 Python 的 csv/標準函式直接核對每模型逐圖算術平均：五個模型的 PSNR/SSIM/LPIPS mean 與 summary 差異皆為 0，分母皆為 4，expected/success/failed 皆 4/4/0。這是短試跑平均，不能代替全量平均。
- 核對 CSV 路徑映射到回傳 run 後，所有 inputs/outputs 檔案存在；名稱含原副檔名，sharpness 的 model-image/condition/path 與 full_reference 一致，GT 不按模型重複。讀取 24 張 PNG 表頭皆為 RGB、5280×3956。
- 實際目視：inputs 的 0881.JPG.png（none）與 0696.JPG.png（trajectory），NAFNet 的同兩張輸出，以及 Uformer 的 0881.JPG.png。輸入沒有彩色方塊；NAFNet 兩张輸出有大片規則彩色區塊與棋盤紋，確認是可解碼但明顯異常的內容。已查看的 Uformer 對照圖未見同樣現象，沒有宣稱其餘所有圖片均已目視通過。
- NAFNet 的 0881.JPG / 0696.JPG PSNR 分別為 8.062397571287418 / 9.114510565912244 dB，輸出 laplacian_variance 分別為 57664.25125337871 / 54295.31281784097。CSV 的 ok 表示輸出及度量取得，不能掩蓋實際內容異常；保留原始列、模型與數值，不自动排除 NAFNet、不改分母或設畫質門檻。
- 唯讀追查現有 `inference.py`/`tiling.py` 與已安裝 Spandrel NAFNet：共用 loader 使用 float32，支援 tiling 時沿用 512 core/32 halo；此入口沒有明確設定 TF32。這些是現有程式事實，尚不能確定異常來自 tiling、GPU 數值或 checkpoint；沒有用推測做程式修正。影響下一步的事實見 `context/phase-03-context.md`。
- 驗證僅使用 CSV 核對、PNG 表頭/檔案關聯與上述代表圖目視；沒有回算全部指標、跑完整測試或新增驗證框架。原始 JPG 不在回傳資料且 lab 路徑本機不可用，無法獨立核對 GT 雜湊、blur 合成或重新計算 GT 全參考成績。
- 本次只提交 log 與影響後續執行的 context，commit 主旨 `docs: record bounded lab deblur run`。短試跑已取得真實證據；全量尚未執行，NAFNet 異常與 lab 成本/環境待查，Phase 03 保留 In progress，未宣稱整體實驗完成。

### 2026-10-04 — Phase 03：補入 lab console 與下一個定位步驟

- 使用者貼回同一 run 的 console，命令為 `/home/gary/test` 下的 `bash lab/run.sh --limit 4`。run 路徑、4 張 5280×3956 圖、seed=923、四組分配與 5 個 checkpoint 均吻合已回傳 CSV；此為使用者提供的 console 證據，未新增或假稱存在 console 檔案。
- 再次唯讀確認 WSL repo、`/usr/bin/git`、現有 Python 3.12.3、fix@`0605f42` 與乾淨工作區；重讀適用 AGENTS、GOALS、PLANS、phase-03、log/context、當前入口及相關載入/tiling/NAFNet 程式。本機指定輸入仍只有 `.gitignore`，沒有 deblur checkpoint；未執行模型。
- lab 確認為 torch `2.11.0+cu128`、CUDA runtime `12.8`、NVIDIA RTX A6000、`CUDA_VISIBLE_DEVICES=0`；啟動時 VRAM free/total = `45366444032/50899386368` bytes（約 42.25/47.40 GiB）。這是 lab 裝置，並非本機筆電 GPU；啟動值不是峰值量測。
- Console 原始 RAM 資訊：MemTotal `396091568 kB`、MemFree `41278028 kB`、MemAvailable `323821248 kB`；cgroup memory.max `64424509440`、memory.current `62796476416` bytes。cgroup 上限 60 GiB，current 約 58.48 GiB，上限與 current 當下差約 1.52 GiB；host MemAvailable 不能當容器可用量，current 也不是此程式獨占用量，不能由此推定 OOM。
- PSNR/SSIM 在 CPU float64，LPIPS 在 cuda:0，使用既有 `/home/gary/test/models/torch-cache/hub/checkpoints/alexnet-owt-7be5be79.pth` 與 lpips v0.1/Alex 校準權重。每模型 4/4 saved、4/4 complete scores；console 沒有載入、推論、度量或 OOM 錯誤。torchvision pretrained/weights 與 torch.meshgrid indexing 提示是棄用警告，沒有證據把它們當作彩色方塊原因，未升級依賴。
- 各模型架構／模型循環耗時：NAFNet `244.9s`、Uformer `335.3s`、FFTformer `571.1s`、MPRNet（model_deblurring.pth）`346.8s`、Restormer（motion_deblurring.pth）`448.3s`。總耗時 `2096.7s`，約 34 分 57 秒；各模型時間包含推論、保存與度量，不能當成純推論 profiling。
- 五模型循環合計 `1946.4s`，其餘 `150.3s` 包含啟動/前處理等；同條件全部五模型每張約 `524.175s`（8.74 分），只能作粗略外推，正式 N 尚未知，未估整批總時數、啟動全量或重跑這輪 35 分鐘工作。
- 唯讀 Git 比較確認被回退的 `33b3960:lab/run.sh` 曾明確關閉 matmul/cuDNN TF32；現入口未設定。PyTorch 官方 numerical_accuracy/CUDA 說明指出 float32 張量不排除 TF32 卷積；但 console 沒印實際精度旗標，這只是候選原因，不是已確認根因。來源連結及固定其他條件的單圖對照命令放入既存 context。
- 下一步限定已異常的 `0881.JPG.png`、同一 NAFNet checkpoint、現有 loader/upscale/原尺寸 tiling，比較現行 cuDNN 卷積精度與 IEEE FP32。只準備供使用者在 lab 執行的命令；不改正式程式、公共介面、共用推論、依賴或分組/成績定義，不產生新測試框架，不自行跑 GPU。命令記錄 Git SHA/Spandrel 版本/實際精度並保存兩張 PNG 到新的診斷目錄；原 run 不修改。
- 此定位命令尚未在 lab 執行；以 NAFNet 四張含度量平均 `61.225s/張` 作成本參考，兩次單圖推論的 IEEE 耗時尚未知，不保證兩分鐘內完成。若現行精度已為 IEEE，命令在推論前停止，不做相同條件的重複工作。
- 本步只改 `deblur/build-log.md` 與 `deblur/context/phase-03-context.md`；文件差異檢查與 context 命令的 Bash/heredoc Python 語法检查通過，沒有模型、完整測試或全量執行。Commit 主旨 `docs: record lab console and focused NAFNet diagnosis`。Phase 03 保留 In progress；NAFNet 內容異常、lab 執行 SHA 與正式全量結果仍未驗證。

### 2026-10-04 — Phase 03：記錄使用者授權跳過 NAFNet

- 使用者明確要求：「我插入一下，跳過NAFNet不管，這邊就備註有問題以及我授權跳過，繼續」。本次授權覆蓋原本全部 checkpoint 的 NAFNet 例外；後續跳過已知 `NAFNet-GoPro-width64.pth`（同檔名含子目錄），取消尚未執行的精度對照/根因定位，其餘模型繼續。不是模型自動淘汰或宣稱 NAFNet 修復。
- 修改前唯讀确认同一 WSL repo、Linux Git/現有 Python/pip、fix@`6a3ef25` 與乾淨工作區；重新讀 AGENTS、GOALS、PLANS、build-log、phase-03 與相關 context。沒有其他 AGENTS/code_review，指定本機資料/checkpoint 仍缺，未跑 GPU。
- 先更新 GOALS 的使用者授權例外、PLANS 的後續順序、phase-03 的四模型全量驗收與 context 的取消註記，再進行入口修改。原始 `20261004T124643.486598Z` 的圖片/三 CSV 及歷史失敗紀錄保留原樣，未刪模型權重。
- 原短試跑扣除 NAFNet 循環後約 `1851.8s/4 張`，約 7.72 分/張；僅為同條件四模型粗略參考，未啟動全量或承諾總耗時。全量實驗仍由使用者在 lab 執行，Phase 03 保持 In progress。
- 首次文件 patch 的 context 比對行不完整而被拒絕；隨後唯讀確認工作區仍乾淨，補正為完整原行後重套，沒有半套檔案修改或模型執行失敗。
- 本步僅更新五份既存 deblur 文件，差異閱讀及 `git diff --check` 通過；不跑模型或程式測試。Commit 主旨 `docs: authorize skipping NAFNet and continue lab plan`。入口排除與其餘代表圖目視留待下一步。
