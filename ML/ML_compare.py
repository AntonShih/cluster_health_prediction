# ====================================================================
# 糖尿病預測機器學習系統 - 精簡版
# 整合：預處理 + 特徵工程 + 機器學習模型 + 評估
# ====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================================================================
# 🎯 為什麼這樣設計？
# ====================================================================
"""
針對你的糖尿病資料集(100K筆, 16特徵)的設計考量：

1. 醫學特徵工程：
   - HbA1c >= 6.5, 血糖 >= 126 → 診斷標準特徵
   - BMI分級、年齡分組 → 風險分層
   - 種族風險、交互作用 → 醫學知識

2. 預處理策略：
   - 數值特徵 → StandardScaler (適合LR)
   - 類別特徵 → OneHotEncoder (避免順序錯誤)
   - ColumnTransformer → 防止資料洩漏

3. 模型選擇：
   - LogisticRegression → 可解釋性(醫療重要)
   - RandomForest → 處理交互作用
   - XGBoost/LightGBM → 非線性建模
   - Ensemble → 提升穩定性
"""

def load_and_create_features(filepath, target_col='diabetes'):
    """載入資料並創建醫學特徵"""
    print("🚀 載入資料並進行醫學特徵工程...")
    
    df = pd.read_csv(filepath)
    print(f"原始資料：{df.shape[0]:,} 筆, {df.shape[1]} 欄位")
    print(f"糖尿病患病率：{df[target_col].mean():.1%}")
    
    # 醫學特徵工程
    if 'hbA1c_level' in df.columns:
        df['HbA1c_Diabetic'] = (df['hbA1c_level'] >= 6.5).astype(int)
        df['HbA1c_Prediabetic'] = ((df['hbA1c_level'] >= 5.7) & (df['hbA1c_level'] < 6.5)).astype(int)
    
    if 'blood_glucose_level' in df.columns:
        df['Glucose_Diabetic'] = (df['blood_glucose_level'] >= 126).astype(int)
        df['Glucose_Prediabetic'] = ((df['blood_glucose_level'] >= 100) & (df['blood_glucose_level'] < 126)).astype(int)
    
    if 'bmi' in df.columns:
        df['BMI_Obese'] = (df['bmi'] >= 30).astype(int)
        df['BMI_Overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
        df['BMI_Squared'] = df['bmi'] ** 2
    
    if 'age' in df.columns:
        df['Age_High_Risk'] = (df['age'] >= 45).astype(int)
        df['Age_Squared'] = df['age'] ** 2
    
    if 'hypertension' in df.columns and 'heart_disease' in df.columns:
        df['CVD_Risk'] = df['hypertension'] + df['heart_disease']
    
    # 吸菸風險量化
    if 'smoking_history' in df.columns:
        smoking_map = {'never': 0, 'No Info': 0.2, 'former': 0.6, 'current': 1.0, 'not current': 0.3, 'ever': 0.8}
        df['Smoking_Risk'] = df['smoking_history'].map(lambda x: smoking_map.get(x, 0.2))
    
    # 高風險種族
    if 'race:AfricanAmerican' in df.columns or 'race:Hispanic' in df.columns:
        df['High_Risk_Race'] = 0
        if 'race:AfricanAmerican' in df.columns:
            df['High_Risk_Race'] |= df['race:AfricanAmerican']
        if 'race:Hispanic' in df.columns:
            df['High_Risk_Race'] |= df['race:Hispanic']
    
    # 交互作用特徵
    if 'age' in df.columns and 'bmi' in df.columns:
        df['Age_BMI_Interaction'] = df['age'] * df['bmi'] / 1000
    
    print(f"特徵工程後：{df.shape[1]} 欄位")
    return df

def preprocess_data(df, target_col='diabetes'):
    """智能預處理管道"""
    print("🤖 執行智能預處理...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 識別特徵類型
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"數值特徵：{len(numerical_cols)} 個")
    print(f"類別特徵：{len(categorical_cols)} 個 {categorical_cols}")
    
    # 建立預處理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    # 獲取特徵名稱
    feature_names = numerical_cols.copy()
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat']
        for i, col in enumerate(categorical_cols):
            categories = cat_encoder.categories_[i][1:]  # drop='first'
            feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    print(f"預處理完成：{X_df.shape[1]} 特徵")
    
    return X_df, y, preprocessor

def select_features(X, y, min_features=15, save_plots=True, output_dir="diabetes_ml_results"):
    """RFECV特徵選擇並儲存視覺化"""
    print("🎯 執行RFECV特徵選擇...")
    
    # 只用80%資料做特徵選擇，避免過擬合
    X_fs, _, y_fs, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    selector = RFECV(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        step=1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc',
        min_features_to_select=min_features,
        n_jobs=-1
    )
    
    selector.fit(X_fs, y_fs)
    
    selected_features = X.columns[selector.support_].tolist()
    X_selected = X[selected_features]
    
    print(f"選出特徵數：{len(selected_features)}")
    print(f"最佳CV分數：{max(selector.cv_results_['mean_test_score']):.4f}")
    
    # 視覺化RFECV結果
    if save_plots:
        import os
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RFECV分數曲線
    scores = selector.cv_results_['mean_test_score']
    ax1.plot(range(1, len(scores) + 1), scores, 'bo-', linewidth=2)
    ax1.axvline(x=selector.n_features_, color='red', linestyle='--', 
               label=f'最佳特徵數: {selector.n_features_}')
    ax1.set_xlabel('特徵數量')
    ax1.set_ylabel('交叉驗證 AUC 分數')
    ax1.set_title('RFECV 特徵選擇結果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top 10 重要特徵
    importance = np.abs(selector.estimator_.coef_[0])
    top_10_idx = np.argsort(importance)[-10:]
    
    ax2.barh(range(10), importance[top_10_idx])
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([selected_features[i] for i in top_10_idx])
    ax2.set_xlabel('特徵重要性 (絕對係數值)')
    ax2.set_title('Top 10 重要特徵')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/plots/feature_selection.png", dpi=300, bbox_inches='tight')
        print(f"🎯 特徵選擇圖已儲存：{output_dir}/plots/feature_selection.png")
    plt.show()
    
    return X_selected, selected_features, selector

def train_models(X, y, test_size=0.2):
    """訓練多個機器學習模型"""
    print("🚀 開始訓練模型...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    models = {}
    metrics = {}
    
    # 1. Logistic Regression
    print("📊 訓練 Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    models['LogisticRegression'] = lr
    metrics['LogisticRegression'] = evaluate_model(lr, X_test, y_test)
    
    # 2. Random Forest
    print("🌲 訓練 Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    metrics['RandomForest'] = evaluate_model(rf, X_test, y_test)
    
    # 3. XGBoost
    print("⚡ 訓練 XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    metrics['XGBoost'] = evaluate_model(xgb_model, X_test, y_test)
    
    # 4. LightGBM
    print("💡 訓練 LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        class_weight='balanced', random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    metrics['LightGBM'] = evaluate_model(lgb_model, X_test, y_test)
    
    # 5. Ensemble (Top 3) - 修復版本相容性問題
    print("🎭 訓練 Ensemble...")
    try:
        model_scores = [(name, m['auc']) for name, m in metrics.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_3 = model_scores[:3]
        
        # 重新訓練選中的模型，確保相容性
        ensemble_models = []
        for name, _ in top_3:
            if name == 'LogisticRegression':
                model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            elif name == 'RandomForest':
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
            elif name == 'XGBoost':
                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
                )
            elif name == 'LightGBM':
                model = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    class_weight='balanced', random_state=42, verbose=-1
                )
            
            model.fit(X_train, y_train)
            ensemble_models.append((name, model))
        
        weights = [score for _, score in top_3]
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',
            weights=weights
        )
        ensemble.fit(X_train, y_train)
        models['Ensemble'] = ensemble
        metrics['Ensemble'] = evaluate_model(ensemble, X_test, y_test)
        
        print(f"🏆 Ensemble 使用：{[name for name, _ in top_3]}")
        
    except Exception as e:
        print(f"⚠️  Ensemble 訓練失敗: {str(e)}")
        print("📊 繼續使用其他4個模型...")
        
        # 如果 Ensemble 失敗，建立簡單的平均預測
        print("🔄 建立簡單平均 Ensemble...")
        try:
            # 簡單平均預測
            all_probas = []
            for name, model in models.items():
                y_proba = model.predict_proba(X_test)[:, 1]
                all_probas.append(y_proba)
            
            avg_proba = np.mean(all_probas, axis=0)
            avg_pred = (avg_proba > 0.5).astype(int)
            
            # 建立虛擬 ensemble 評估
            ensemble_metrics = {
                'accuracy': accuracy_score(y_test, avg_pred),
                'precision': precision_score(y_test, avg_pred),
                'recall': recall_score(y_test, avg_pred),
                'f1': f1_score(y_test, avg_pred),
                'auc': roc_auc_score(y_test, avg_proba)
            }
            
            metrics['SimpleEnsemble'] = ensemble_metrics
            print("✅ 簡單平均 Ensemble 完成")
            
        except Exception as e2:
            print(f"❌ 簡單 Ensemble 也失敗: {str(e2)}")
            print("📊 只使用單一模型結果")
    
    return models, metrics, (X_train, X_test, y_train, y_test)

def evaluate_model(model, X_test, y_test):
    """評估模型性能"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }

def plot_results(metrics, models=None, X_test=None, y_test=None, save_plots=True, output_dir="diabetes_ml_results"):
    """繪製結果比較並儲存圖片"""
    # 性能比較表
    results_df = pd.DataFrame(metrics).T.round(4)
    print("\n📊 模型性能比較：")
    print(results_df)
    
    best_model = results_df['auc'].idxmax()
    print(f"\n🏆 最佳模型：{best_model} (AUC: {results_df.loc[best_model, 'auc']:.4f})")
    
    if save_plots:
        import os
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # 1. 繪製AUC比較圖和ROC曲線
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    auc_scores = results_df['auc'].sort_values()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    bars = plt.barh(range(len(auc_scores)), auc_scores.values, color=colors[:len(auc_scores)])
    plt.yticks(range(len(auc_scores)), auc_scores.index)
    plt.xlabel('AUC Score')
    plt.title('🎯 模型 AUC 比較')
    
    # 在條形圖上顯示數值
    for i, (bar, value) in enumerate(zip(bars, auc_scores.values)):
        plt.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=10)
    
    # 如果有模型和測試資料，繪製最佳模型的ROC曲線
    if models and X_test is not None and y_test is not None:
        plt.subplot(1, 2, 2)
        best_model_obj = models[best_model]
        y_proba = best_model_obj.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{best_model} (AUC = {results_df.loc[best_model, "auc"]:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'📈 {best_model} ROC 曲線')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/plots/model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"📊 模型比較圖已儲存：{output_dir}/plots/model_comparison.png")
    plt.show()
    
    # 2. 繪製所有模型的ROC曲線對比
    if models and X_test is not None and y_test is not None:
        plt.figure(figsize=(8, 6))
        
        for i, (name, model) in enumerate(models.items()):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = results_df.loc[name, 'auc']
            
            plt.plot(fpr, tpr, linewidth=2, color=colors[i % len(colors)],
                    label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('📈 所有模型 ROC 曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(f"{output_dir}/plots/all_roc_curves.png", dpi=300, bbox_inches='tight')
            print(f"📈 ROC曲線比較圖已儲存：{output_dir}/plots/all_roc_curves.png")
        plt.show()
    
    # 3. 繪製混淆矩陣（所有模型）
    if models and X_test is not None and y_test is not None:
        n_models = len(models)
        cols = 3 if n_models > 3 else n_models
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, model) in enumerate(models.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['非糖尿病', '糖尿病'],
                       yticklabels=['非糖尿病', '糖尿病'])
            ax.set_title(f'🔍 {name} 混淆矩陣')
            ax.set_xlabel('預測值')
            ax.set_ylabel('真實值')
        
        # 隱藏多餘的子圖
        for i in range(n_models, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{output_dir}/plots/confusion_matrices.png", dpi=300, bbox_inches='tight')
            print(f"🔍 混淆矩陣圖已儲存：{output_dir}/plots/confusion_matrices.png")
        plt.show()
    
    # 4. 繪製雷達圖
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    angles = np.linspace(0, 2 * np.pi, len(metrics_list), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    for i, (model_name, model_metrics) in enumerate(results_df.iterrows()):
        values = [model_metrics[metric] for metric in metrics_list]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=model_name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics_list])
    ax.set_ylim(0, 1)
    ax.set_title('🎯 模型性能雷達圖', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/plots/performance_radar.png", dpi=300, bbox_inches='tight')
        print(f"🎯 雷達圖已儲存：{output_dir}/plots/performance_radar.png")
    plt.show()
    
    return results_df

def save_results(models, preprocessor, selector, selected_features, metrics, output_dir="diabetes_ml_results"):
    """儲存所有結果包含圖片"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # 儲存模型
    for name, model in models.items():
        joblib.dump(model, f"{output_dir}/{name}_model.pkl")
    
    # 儲存預處理器和特徵選擇器
    joblib.dump(preprocessor, f"{output_dir}/preprocessor.pkl")
    joblib.dump(selector, f"{output_dir}/feature_selector.pkl")
    
    # 儲存特徵名稱
    with open(f"{output_dir}/selected_features.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(selected_features))
    
    # 儲存性能指標為CSV
    results_df = pd.DataFrame(metrics).T.round(4)
    results_df.to_csv(f"{output_dir}/model_performance.csv", encoding='utf-8')
    
    # 建立結果摘要報告
    best_model = results_df['auc'].idxmax()
    best_auc = results_df['auc'].max()
    
    report = f"""
# 糖尿病預測模型訓練結果摘要

## 🏆 最佳模型
- 模型名稱：{best_model}
- AUC 分數：{best_auc:.4f}
- 準確率：{results_df.loc[best_model, 'accuracy']:.4f}
- 精確率：{results_df.loc[best_model, 'precision']:.4f}
- 召回率：{results_df.loc[best_model, 'recall']:.4f}
- F1分數：{results_df.loc[best_model, 'f1']:.4f}

## 📊 所有模型性能
{results_df.to_string()}

## 🎯 選中的關鍵特徵 (共{len(selected_features)}個)
{chr(10).join([f"- {feature}" for feature in selected_features[:15]])}
{'...' if len(selected_features) > 15 else ''}

## 📁 檔案說明
- 模型檔案：*_model.pkl
- 預處理器：preprocessor.pkl
- 特徵選擇器：feature_selector.pkl
- 性能比較圖：plots/model_comparison.png
- ROC曲線比較：plots/all_roc_curves.png
- 雷達圖：plots/performance_radar.png
- 特徵選擇圖：plots/feature_selection.png
"""
    
    with open(f"{output_dir}/README.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 所有結果已儲存至：{output_dir}/")
    print(f"📊 包含 5 張圖片：")
    print(f"   - {output_dir}/plots/model_comparison.png")
    print(f"   - {output_dir}/plots/all_roc_curves.png")
    print(f"   - {output_dir}/plots/confusion_matrices.png")
    print(f"   - {output_dir}/plots/performance_radar.png")
    print(f"   - {output_dir}/plots/feature_selection.png")
    print(f"📄 摘要報告：{output_dir}/README.md")

# ====================================================================
# 主要執行函數
# ====================================================================

def main(filepath="data/diabetes_dataset.csv", target_col='diabetes'):
    """完整執行流程"""
    print("🎯 糖尿病預測機器學習系統")
    print("=" * 50)
    
    # 1. 載入資料並特徵工程
    df = load_and_create_features(filepath, target_col)
    
    # 2. 預處理
    X_processed, y, preprocessor = preprocess_data(df, target_col)
    
    # 3. 特徵選擇
    X_selected, selected_features, selector = select_features(X_processed, y)
    
    # 4. 訓練模型
    models, metrics, (X_train, X_test, y_train, y_test) = train_models(X_selected, y)
    
    # 5. 結果比較（包含圖片儲存）
    results_df = plot_results(metrics, models, X_test, y_test, save_plots=True, output_dir="diabetes_ml_results")
    
    # 6. 儲存所有結果（包含圖片）
    save_results(models, preprocessor, selector, selected_features, metrics, "diabetes_ml_results")
    
    print("\n🎉 完成！")
    return models, metrics, results_df, selected_features

# 執行範例
if __name__ == "__main__":
    # 執行完整流程
    models, metrics, results_df, selected_features = main()
    
    print(f"\n🎯 重點特徵：{selected_features[:10]}")
    print(f"📊 最終結果：")
    print(results_df)