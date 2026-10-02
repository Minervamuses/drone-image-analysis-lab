# Phase 01 — 本機環境、三模式與有序推論

## 來源與目標

讀 [GOALS.md](../GOALS.md)、[PLANS.md](../PLANS.md)、`src/drone_sr/__main__.py`、`inference.py`、`tiling.py`、`image_io.py` 及相關 tests。
成果：一般 CLI 正確執行兩種單階段與兩種有序 combine，支援候選架構的載入路徑，維持原圖保護；以本機無預訓練權重檢查證實。

## 範圍、依賴與非目標

無前置 phase。依 PLANS 自行建立符合 Python 3.12 的 Linux `.venv` 並安裝／核對現有固定依賴；使用者已授權，不等待安裝批准。
主要改 `__main__.py`、`inference.py`、相關依賴檔與 README；`tiling.py` 僅為已證明必要的 x1 相容修改。保留 `image_io.py` 與 tests 既有使用者差異。
加入官方 extra_arches 的必要依賴與一次註冊，不為五個候選建立五套執行器。真權重、GPU、畫質及任意模型分塊效果不是本 phase 驗收；本機不下載權重。
不做 evaluation 調度／四指標，不調整既有 SR tile／精度，不建立 generic pipeline module。

## Preflight 與實作方向

1. 確認相關工作樹、呼叫者、README 安裝流程及套件版本；無權重的聚焦測試建立基線。必要安裝先告知預估成本後自行完成；不得混用 Windows Python。
2. argparse 用同一有序清單收集 `--sr`／`--deblur`；無旗標或重複旗標報錯。非交換 synthetic 操作驗證順序。模型參數與別名遵循 GOALS，只載啟用階段。
3. 載入前註冊 `spandrel_extra_arches.install()`；讀取目前安裝版本確認 API，以無權重 registry 檢查涵蓋五個候選。僅使用官方 Spandrel descriptor 介面。
4. load_model 接收階段角色；SR 要 ImageModelDescriptor、RGB、scale>1，deblur 要 ImageModelDescriptor、RGB、Restoration、scale=1。保留直接呼叫者的 SR 預設角色；預設路徑遷移到 `models/sr/model.pth`，不移動素材。
5. 在既有 inference 模組放最小有序串接，供一般入口與 evaluation 呼叫。階段間直接傳 tensor，最後才 write_png。逐階段檢查 shape／有限值並標記錯誤階段；finite 是新增的階段邊界檢查，原本只有末端寫檔檢查。
6. 保持 FP32。查 descriptor 的尺寸／tiling 資訊，尊重其 padding／裁回：來源顯示 FFTformer 32 倍數、Uformer 正方形且 128 倍數、Restormer／MPRNet 8 倍數；以安裝版本為準。不要繞過 descriptor 或因 x1 synthetic 通過便宣稱五模型無接縫。
7. x1 direct／tile 的座標、維度與 device 銜接以 tiny descriptors 檢查；SR 原設定不變。不要一次把所有 deblur 權重載入 GPU；實際權重不支援時必須明確失敗，不默默切 CPU。
8. 第二階段失敗保持整張失敗，保留既有舊成功檔但不計新成功，繼續下一張。一般 CLI 比較四模式的文件指定各自輸出目錄。
9. 同步一般 CLI help／README 必要範例及模型路徑；Colab 鎖定舊版本部分不重做，Docker 用法若受 CLI 影響只做直接修正並標未驗證。

## 必要驗證

repo 根目錄、新建或已準備的 Linux .venv：
```bash
.venv/bin/python -m pip check
PYTHONPATH=src:tests .venv/bin/python -m unittest test_cli test_inference test_tiling
PYTHONPATH=src:tests .venv/bin/python -m unittest test_image_io
git diff --check
```

沿用既有 tests，加入最少的旗標／非交換順序、角色／通道／倍率、x1 direct／tile、階段失敗案例；registry 檢查置於 test_inference，不產生預訓練模型下載。
環境與模型套件 import 檢查不代表真權重已可推論。必要測試失敗先按 PLANS 修復，不能跳進 Phase 02。
完成每批改動的檢查、build-log 記錄和 commit 後才做下一批；不另外安排通用重構。

## 驗收条件

- [ ] 可用 Linux 測試環境、pip check 與聚焦案例通過；未下載預訓練權重。
- [ ] 無／重複 flags 在載入前報錯；四路順序正確，單階段不要求另一類權重。
- [ ] 官方核心＋extra registry 包含五候選，實際 checkpoint 狀態明示待 lab。
- [ ] RGB／角色／倍率錯誤拒絕，階段有限值／尺寸錯誤可定位。
- [ ] x1 與 SR 座標／尺寸、安全寫檔、逐圖續行及既有 I/O 回歸通過。
- [ ] 相關 help／說明一致，自己的變更已 commit、使用者既有修改保留。

## 恢復、證據與交接

只修本次範圍，不清除既有输出／素材。build-log 記實際環境、命令、結果、commit 與未驗證限制；僅重要 descriptor／device 發現寫 context。
必要驗收完成並提交後自動進 Phase 02；不等待使用者下載 deblur、不提前跑真模型。
