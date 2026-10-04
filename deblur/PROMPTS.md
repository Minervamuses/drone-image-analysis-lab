# deblur — 啟動與續作提示

## 啟動／續作整份計劃

複製以下提示：

> 執行 deblur/ 計劃。先讀適用的 AGENTS.md、deblur/GOALS.md、deblur/PLANS.md、deblur/build-log.md，再讀依 PLANS 依賴順序找到的第一個未完成且前置已完成的 phases 文件，以及已存在的相關 context/code_review。從 build-log 恢復狀態，不依賴聊天記憶。
>
> 每個步驟先唯讀確認 repo、WSL 工具鏈、Git 狀態與當前檔案，再只實作該步驟；依 phase 做最小必要檢查，修正失敗，更新 build-log 並立即 commit。Git 命令全部在 WSL；分支／回退／commit 已依 PLANS 授權，不重複要求批准。
>
> 依授權自主繼續，不額外跑嚴格測試、不擴張框架。發現會影響下一步的事實才寫 context；推翻後續計劃時先修 PLANS 與未開始的 phase，再繼續。保留失敗證據。直到整體完成或 PLANS 的停止條件；本機缺 lab 資料與權重時交付已完成的程式，記錄 lab 待跑，不下載替代資料。

## 執行單一階段

> 讀取上述 deblur/ 文件，從 deblur/PLANS.md 選擇我指定的單一階段。確認依賴、唯讀 preflight、只做該階段與必要檢查，記錄證據並按授權 commit，完成後停止。

## 驗證／修正計劃

> 依 deblur/GOALS.md 對照目前實際檔案、diff 和 deblur/build-log.md，檢查該階段是否符合本次一次性實驗。只用 phase 指定的必要檢查，不加完整套件或新框架。若新證據推翻計劃，只修 deblur/PLANS.md 與尚未開始的 phase，保留既有證據；此提示本身不授權執行修正後的程式工作。

## 收尾檢視

> 對照 deblur/GOALS.md 與 lab 實際產物，確認逐圖結果、每模型全體平均、純資料 sharpness.csv 和 Git 紀錄。區分程式交付與實驗完成；沒有 lab 輸出證據不標第三階段 Complete。回報限制後停止。
