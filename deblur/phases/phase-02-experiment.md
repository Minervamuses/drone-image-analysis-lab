# Phase 02 — 覆寫 lab/run.sh 完成一次性實驗

## 目標、範圍與非目標

直接覆寫 lab/run.sh，以 [GOALS](../GOALS.md) 的資料與分數約定實作一次實驗。唯一正式程式修改是此檔，另更新 ../build-log.md。不改 SR 入口、套件、tests 或舊 report/runner；不增加持久 Python 模組。

## 前置與未知

第一階段已 Complete，分支為 fix，工作區符合紀錄。沿用現有 .venv 與 Python 3.12。資料、權重與 LPIPS cache 在 lab，不能要求本機具備，也不把本機的無模型檢查當作真正推論證據。

## 最小實作

1. Bash 只定位 repo、選既有 Python，執行檔內 Python。新入口預設 `bash lab/run.sh` 跑全部；只需 `--limit N` 供第三階段短試跑，與 `--help`。資料、模型、輸出路徑與固定參數集中在腳本頂部，不建設定框架。--limit 在固定洗牌後取前 N 張，再做相同循環分組。
2. 先處理所有選定圖片並保存模型輸入。沿用 `drone_sr.image_io.read_image` 解碼原圖，尊重 EXIF/MPO；實作四種單一處理及輸出對應。GT 清晰度只算一次。分配與 CSV 欄位完全按 GOALS。
3. 逐個 checkpoint 載入 `drone_sr.inference.load_model(..., role="deblur")`，用 `upscale` 逐張處理，保存原尺寸 PNG。每次只保留一個模型，不新增並行。檢查 CUDA、必要套件及既有 LPIPS cache；缺少就明確告知，不自動安裝／下載／轉 CPU 長跑。
4. 用既有 `metrics.psnr/ssim`、`perceptual.PerceptualMetric` 計算輸出相對原圖的分數，並用 `blur_metrics.measure_tensor` 算四項原始值。LPIPS 不另做模型選擇。先清除已不需要的模型／tensor 再算重型度量；不修改算法。
5. 寫出 GOALS 指定的三份 CSV 與圖片。單圖／單模型錯誤記錄後處理其餘項目；模型載入失敗也在 summary 有一列，不靜默消失。四項無參考指標不寫敘述、不產生分析。報表不因一個失敗模型把其他結果清空。

按兩個修改步驟提交：完成上列 1–2（入口與前處理）即做下列語法／help 檢查，更新 log 為 In progress，commit `feat: prepare evenly assigned deblur inputs`；再完成 3–5（模型、指標與 CSV），檢查並作第二個 commit。必要修正也各自 commit，不等到 lab 正式實驗才一次提交。

## 最小驗證

僅需：
- `bash -n lab/run.sh`。
- 用既有 Python 的 `ast.parse` 或 `compile(..., "exec")` 解析腳本內 heredoc 的 Python；依實際 delimiter 擷取，不执行模型或下載。這個檢查不產生 .pyc。
- `git diff --check`，閱讀這次 diff，確認目標是原圖、N 張只分一次組、各模型讀相同輸入，三份 CSV 符合 GOALS。
- `bash lab/run.sh --help` 在未有資料／權重時仍可顯示使用方式，不提前載入模型。

不新增測試檔、不跑全套、不造假模型、不跑 GPU；本機資料／checkpoint 缺席明列未驗證。此處是程式交付門檻，不宣稱實驗成功。

## 驗收與失敗處理

- lab/run.sh 可被上述語法／help 檢查，且未修改其他正式程式或依賴。
- 有三份 CSV 的明確寫出路徑、數據欄位及失敗列；未引入多強度／每圖四版本的矩陣。
- 缺失外部資料不阻擋本機交付；語法或已確認計算／配對錯誤則修復後才交接。
- 如果完成本次要求確實需要改共用檔或依賴，先依 PLANS 說明必要性與最小範圍；不能自行增設 adapter。
- 兩次聚焦修正失敗就停止，記錄證據與最小下一步。

## 證據、commit 與交接

第一個 commit 完成後繼續本階段；最後把實際檢查、未驗證項目、變更範圍寫入 ../build-log.md，提交第二個修改步驟：

```bash
git add -- lab/run.sh deblur/build-log.md
git diff --cached --check
git commit -m "feat: run one minimal deblur experiment with raw metric tables"
```

第二階段可標 Complete，第三階段等待 lab。回報使用者在 lab 執行的現有命令，不代為 push 或修改 server。
