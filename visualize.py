import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
sns.set_palette('Set2')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

# 讀取評估結果
overall_df = pd.read_csv('result/overall_metrics.csv')
language_df = pd.read_csv('result/language_metrics.csv')
subject_df = pd.read_csv('result/subject_metrics.csv')

# 1. 比較不同模型和格式的準確率
def plot_accuracy_comparison():
    plt.figure(figsize=(14, 8))
    
    bar_width = 0.35
    x = np.arange(len(overall_df))
    
    plt.bar(x - bar_width/2, overall_df['Original_Accuracy'], width=bar_width, 
            label='Original Order', color='steelblue')
    plt.bar(x + bar_width/2, overall_df['Shuffled_Accuracy'], width=bar_width, 
            label='Shuffled Order', color='lightcoral')
    
    plt.xlabel('Model & Format Combination')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: Original vs. Shuffled Order')
    plt.xticks(x, overall_df.apply(lambda row: f"{row['Model']}\n{row['Format']}", axis=1), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/accuracy_comparison.png', dpi=300)
    plt.close()

# 2. 比較不同模型和格式的波動率
def plot_fluctuation_rate():
    plt.figure(figsize=(14, 8))
    
    # 按模型分組並按波動率排序
    pivot_df = overall_df.pivot(index='Format', columns='Model', values='Fluctuation_Rate')
    pivot_df.plot(kind='bar', rot=0)
    
    plt.xlabel('Format Type')
    plt.ylabel('Fluctuation Rate')
    plt.title('Order Sensitivity (Fluctuation Rate) Across Input-Output Formats')
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figures/fluctuation_rate.png', dpi=300)
    plt.close()

# 3. 跨語言的順序敏感性
def plot_language_comparison():
    plt.figure(figsize=(14, 8))
    
    # 只看 base_format 格式下的語言比較
    lang_df = language_df[language_df['Format'] == 'base_format']
    
    # 創建分組條形圖
    bar_width = 0.35
    models = lang_df['Model'].unique()
    languages = lang_df['Language'].unique()
    x = np.arange(len(languages))
    
    for i, model in enumerate(models):
        model_data = lang_df[lang_df['Model'] == model]
        plt.bar(x + (i - 0.5) * bar_width, model_data['Fluctuation_Rate'], 
                width=bar_width, label=model)
    
    plt.xlabel('Language')
    plt.ylabel('Fluctuation Rate')
    plt.title('Order Sensitivity Across Languages')
    plt.xticks(x, languages)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/language_comparison.png', dpi=300)
    plt.close()

# 4. 跨學科的順序敏感性熱圖
def plot_subject_heatmap():
    # 只比較基本格式下的學科差異 
    subject_pivot = subject_df[subject_df['Format'] == 'base_format'].pivot_table(
        index='Subject', columns='Model', values='Fluctuation_Rate')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(subject_pivot, annot=True, cmap='YlOrRd', fmt='.3f', 
                linewidths=0.5, cbar_kws={'label': 'Fluctuation Rate'})
    plt.title('Order Sensitivity Across Subjects')
    plt.tight_layout()
    plt.savefig('figures/subject_heatmap.png', dpi=300)
    plt.close()

# 5. 信心與順序敏感性的關係分析 (使用 CKLD 作為信心指標)
def plot_confidence_vs_sensitivity():
    plt.figure(figsize=(10, 8))
    
    for model in overall_df['Model'].unique():
        model_data = overall_df[overall_df['Model'] == model]
        plt.scatter(model_data['Original_CKLD'], model_data['Fluctuation_Rate'], 
                   label=model, s=100, alpha=0.7)
    
    # 添加趨勢線（添加錯誤處理）
    x = overall_df['Original_CKLD']
    y = overall_df['Fluctuation_Rate']
    
    # 檢查數據有效性
    valid_indices = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    
    # 只有當有足夠的有效數據點時才計算趨勢線
    if len(x_valid) > 1:
        try:
            z = np.polyfit(x_valid, y_valid, 1)
            p = np.poly1d(z)
            plt.plot(x_valid, p(x_valid), "r--", alpha=0.5)
            
            # 計算相關係數
            corr = np.corrcoef(x_valid, y_valid)[0, 1]
            plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        except np.linalg.LinAlgError:
            print("警告：無法計算趨勢線，數據可能不足或存在問題")
    else:
        print("警告：有效數據點不足，無法顯示趨勢線")
    
    plt.xlabel('Model Confidence (CKLD)')
    plt.ylabel('Order Sensitivity (Fluctuation Rate)')
    plt.title('Relationship Between Model Confidence and Order Sensitivity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figures/confidence_vs_sensitivity.png', dpi=300)
    plt.close()

# 6. 不同格式下的偏差指標比較
def plot_bias_metrics_by_format():
    # 融合模型數據，計算平均值
    format_metrics = overall_df.groupby('Format').agg({
        'Original_RSD': 'mean',
        'Shuffled_RSD': 'mean',
        'Original_RStd': 'mean',
        'Shuffled_RStd': 'mean',
        'Fluctuation_Rate': 'mean',
        'Original_CKLD': 'mean',
        'Shuffled_CKLD': 'mean'
    }).reset_index()
    
    # 創建雷達圖
    categories = ['RSD', 'RStd', 'Fluctuation Rate', 'CKLD']
    N = len(categories)
    
    # 計算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 閉合多邊形
    
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)
    
    # 為每個格式畫一條線
    for i, format_name in enumerate(format_metrics['Format']):
        values = [
            format_metrics.loc[i, 'Original_RSD'] / 100,  # 歸一化
            format_metrics.loc[i, 'Original_RStd'],
            format_metrics.loc[i, 'Fluctuation_Rate'],
            format_metrics.loc[i, 'Original_CKLD'] / 2  # 歸一化
        ]
        values += values[:1]  # 閉合多邊形
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=format_name)
        ax.fill(angles, values, alpha=0.1)
    
    # 設置雷達圖參數
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Bias Metrics Comparison Across Formats', size=15, y=1.1)
    plt.tight_layout()
    plt.savefig('figures/bias_metrics_radar.png', dpi=300)
    plt.close()

# 執行所有圖表生成
def main():
    print("Generating visualizations...")
    
    plot_accuracy_comparison()
    plot_fluctuation_rate()
    plot_language_comparison()
    plot_subject_heatmap()
    plot_confidence_vs_sensitivity()
    plot_bias_metrics_by_format()
    
    print("Visualization complete. Results saved in 'figures' directory.")

if __name__ == "__main__":
    main()