
# 糖尿病預測模型訓練結果摘要

## 🏆 最佳模型
- 模型名稱：LightGBM
- AUC 分數：0.9793
- 準確率：0.9082
- 精確率：0.4790
- 召回率：0.9188
- F1分數：0.6297

## 📊 所有模型性能
                    accuracy  precision  recall      f1     auc
LogisticRegression    0.8791     0.4055  0.9065  0.5604  0.9682
RandomForest          0.9708     0.9559  0.6882  0.8003  0.9652
XGBoost               0.9070     0.4758  0.9200  0.6272  0.9791
LightGBM              0.9082     0.4790  0.9188  0.6297  0.9793
SimpleEnsemble        0.9368     0.5875  0.8612  0.6985  0.9782

## 🎯 選中的關鍵特徵 (共61個)
- age
- hypertension
- heart_disease
- bmi
- hbA1c_level
- blood_glucose_level
- HbA1c_Diabetic
- HbA1c_Prediabetic
- Glucose_Diabetic
- Glucose_Prediabetic
- BMI_Overweight
- BMI_Squared
- Age_High_Risk
- Age_Squared
- CVD_Risk
...

## 📁 檔案說明
- 模型檔案：*_model.pkl
- 預處理器：preprocessor.pkl
- 特徵選擇器：feature_selector.pkl
- 性能比較圖：plots/model_comparison.png
- ROC曲線比較：plots/all_roc_curves.png
- 雷達圖：plots/performance_radar.png
- 特徵選擇圖：plots/feature_selection.png
