import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
# Thêm ngay sau phần import
pd.set_option('mode.use_inf_as_na', True)
import gc

# Và sau mỗi hàm nặng, thêm:
gc.collect()

# Cấu hình
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")
PROCESSED_DIR = "processed_features"
SAMPLE_SIZE = None
OUTPUT_DIR = "visualizations"

# Tạo thư mục output
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_data(sample_size=None):
    """Đọc tất cả dữ liệu"""
    print(f"📂 Đang đọc dữ liệu (sample_size={sample_size})...")
    
    metadata_path = os.path.join(PROCESSED_DIR, "metadata.csv")
    metadata = pd.read_csv(metadata_path, nrows=sample_size)
    
    feature_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_features.csv')]
    all_features = {}
    
    for file in sorted(feature_files):
        feature_name = file.replace('_features.csv', '')
        file_path = os.path.join(PROCESSED_DIR, file)
        all_features[feature_name] = pd.read_csv(file_path, nrows=sample_size)
        print(f"   ✓ {feature_name}: {all_features[feature_name].shape}")
    
    return metadata, all_features

def plot_class_distribution(metadata):
    """1. Phân tích phân bố nhãn chi tiết"""
    print("\n📊 1. Phân tích phân bố nhãn...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    label_counts = metadata['label'].value_counts().sort_index()
    label_mapping = {-1: 'Unknown', 0: 'Benign', 1: 'Malware'}
    colors = {-1: '#95E1D3', 0: '#4ECDC4', 1: '#FF6B6B'}
    
    # 1. Pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    labels_pie = [f"{label_mapping.get(k, f'Label {k}')}\n{v:,}" 
                  for k, v in zip(label_counts.index, label_counts.values)]
    actual_colors = [colors.get(k, '#CCCCCC') for k in label_counts.index]
    ax1.pie(label_counts.values, labels=labels_pie, autopct='%1.1f%%', 
            startangle=90, colors=actual_colors, textprops={'fontsize': 10})
    ax1.set_title('Phân bố tổng thể', fontsize=12, fontweight='bold')
    
    # 2. Bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    actual_labels = [label_mapping.get(k, f'Label {k}') for k in label_counts.index]
    bars = ax2.bar(actual_labels, label_counts.values, color=actual_colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Số lượng mẫu', fontsize=11, fontweight='bold')
    ax2.set_title('Số lượng theo nhãn', fontsize=12, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Imbalance ratio
    ax3 = fig.add_subplot(gs[0, 2])
    if 0 in label_counts.index and 1 in label_counts.index:
        imbalance_ratio = max(label_counts[0], label_counts[1]) / min(label_counts[0], label_counts[1])
        ax3.barh(['Imbalance\nRatio'], [imbalance_ratio], color='#FF6B6B', edgecolor='black', linewidth=1.5)
        ax3.text(imbalance_ratio/2, 0, f'{imbalance_ratio:.2f}:1', 
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax3.set_xlabel('Ratio', fontsize=11, fontweight='bold')
        ax3.set_title('Tỷ lệ mất cân bằng', fontsize=12, fontweight='bold')
        
        threshold = 1.5
        ax3.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Ngưỡng tốt ({threshold})')
        ax3.legend(fontsize=9)
    
    # 4. Distribution by appeared field (train/test)
    if 'appeared' in metadata.columns:
        ax4 = fig.add_subplot(gs[1, :])
        appeared_label = metadata.groupby(['appeared', 'label']).size().unstack(fill_value=0)
        appeared_label.plot(kind='bar', ax=ax4, color=[colors.get(k, '#CCCCCC') for k in appeared_label.columns],
                           edgecolor='black', linewidth=1.5, width=0.7)
        ax4.set_xlabel('Dataset Split', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Số lượng mẫu', fontsize=11, fontweight='bold')
        ax4.set_title('Phân bố nhãn theo Train/Test Split', fontsize=12, fontweight='bold')
        ax4.legend(title='Label', labels=[label_mapping.get(k, f'Label {k}') for k in appeared_label.columns])
        ax4.tick_params(axis='x', rotation=0)
        
        # Add value labels
        for container in ax4.containers:
            ax4.bar_label(container, fmt='%d', fontsize=9)
    
    # 5. Statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    stats_data = []
    for label, count in label_counts.items():
        label_name = label_mapping.get(label, f'Label {label}')
        pct = count / len(metadata) * 100
        stats_data.append([label_name, f'{count:,}', f'{pct:.2f}%'])
    
    stats_data.append(['TOTAL', f'{len(metadata):,}', '100.00%'])
    
    table = ax5.table(cellText=stats_data,
                     colLabels=['Nhãn', 'Số lượng', 'Tỷ lệ (%)'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(stats_data) + 1):
        if i == 0:  # Header
            for j in range(3):
                table[(i, j)].set_facecolor('#4ECDC4')
                table[(i, j)].set_text_props(weight='bold', color='white')
        elif i == len(stats_data):  # Total row
            for j in range(3):
                table[(i, j)].set_facecolor('#FFE66D')
                table[(i, j)].set_text_props(weight='bold')
    
    plt.suptitle('Phân tích phân bố lớp dữ liệu (Class Distribution Analysis)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    save_path = os.path.join(OUTPUT_DIR, '01_class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Đã lưu: {save_path}")
    plt.close()

def plot_feature_dimensions(all_features):
    """2. Phân tích số chiều và tỷ lệ"""
    print("\n📊 2. Phân tích số chiều đặc trưng...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    dims = {name: df.shape[1] for name, df in all_features.items()}
    total_dims = sum(dims.values())
    total_samples = list(all_features.values())[0].shape[0]
    
    sorted_dims = sorted(dims.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_dims]
    values = [x[1] for x in sorted_dims]
    
    # 1. Bar chart with percentages
    ax1 = fig.add_subplot(gs[0, :])
    colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = ax1.bar(names, values, color=colors_gradient, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Số chiều', fontsize=12, fontweight='bold')
    ax1.set_title(f'Số chiều các nhóm đặc trưng (Tổng: {total_dims:,} chiều)', 
                 fontsize=13, fontweight='bold', pad=15)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    
    for bar in bars:
        height = bar.get_height()
        pct = height/total_dims*100
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Pie chart with percentages
    ax2 = fig.add_subplot(gs[1, 0])
    explode = [0.05 if v/total_dims > 0.15 else 0 for v in values]
    wedges, texts, autotexts = ax2.pie(values, labels=names, autopct='%1.1f%%',
                                        startangle=90, colors=colors_gradient,
                                        explode=explode, textprops={'fontsize': 9})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title('Tỷ lệ % các nhóm', fontsize=12, fontweight='bold')
    
    # 3. Cumulative percentage
    ax3 = fig.add_subplot(gs[1, 1])
    cumsum = np.cumsum(values)
    cumsum_pct = cumsum / total_dims * 100
    
    ax3.plot(range(len(names)), cumsum_pct, marker='o', linewidth=2.5, 
            markersize=8, color='#FF6B6B')
    ax3.fill_between(range(len(names)), cumsum_pct, alpha=0.3, color='#FF6B6B')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Tỷ lệ tích lũy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Phân bố tích lũy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% threshold')
    ax3.legend(fontsize=9)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(range(len(names)), cumsum_pct)):
        ax3.text(x, y+2, f'{y:.1f}%', ha='center', fontsize=8)
    
    plt.suptitle('Phân tích chiều dữ liệu (Dimensionality Analysis)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    save_path = os.path.join(OUTPUT_DIR, '02_feature_dimensions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Đã lưu: {save_path}")
    plt.close()

def plot_sparsity_analysis(all_features):
    """3. Phân tích độ thưa chi tiết"""
    print("\n📊 3. Phân tích độ thưa dữ liệu...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    sparsity_stats = []
    for name, df in all_features.items():
        print(f"   ⏳ Đang tính sparsity cho {name}...")
        
        # Tính toán hiệu quả hơn cho dataset lớn
        total_elements = df.shape[0] * df.shape[1]
        
        # Sử dụng numpy để tránh tràn RAM
        zero_elements = int(np.sum(df.values == 0))
        nan_elements = int(np.sum(np.isnan(df.values)))
        
        sparsity_stats.append({
            'Nhóm': name,
            'Zero (%)': zero_elements / total_elements * 100,
            'NaN (%)': nan_elements / total_elements * 100,
            'Non-zero (%)': (total_elements - zero_elements - nan_elements) / total_elements * 100,
            'Total Elements': total_elements
        })
    
    df_sparsity = pd.DataFrame(sparsity_stats).sort_values('Zero (%)', ascending=False)
    
    # 1. Stacked bar chart
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(df_sparsity))
    width = 0.6
    
    p1 = ax1.bar(x, df_sparsity['Zero (%)'], width, label='Zero values',
                color='#FF6B6B', edgecolor='black', linewidth=1)
    p2 = ax1.bar(x, df_sparsity['NaN (%)'], width, bottom=df_sparsity['Zero (%)'],
                label='NaN values', color='#FFE66D', edgecolor='black', linewidth=1)
    p3 = ax1.bar(x, df_sparsity['Non-zero (%)'], width,
                bottom=df_sparsity['Zero (%)'] + df_sparsity['NaN (%)'],
                label='Non-zero values', color='#4ECDC4', edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Nhóm đặc trưng', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Tỷ lệ (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Phân tích độ thưa theo nhóm (Stacked)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_sparsity['Nhóm'], rotation=45, ha='right', fontsize=10)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim([0, 105])
    
    # 2. Zero percentage only
    ax2 = fig.add_subplot(gs[1, 0])
    colors_zero = ['#FF6B6B' if v > 80 else '#FFE66D' if v > 50 else '#4ECDC4' 
                   for v in df_sparsity['Zero (%)']]
    bars = ax2.barh(df_sparsity['Nhóm'], df_sparsity['Zero (%)'], 
                    color=colors_zero, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Zero values (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Tỷ lệ giá trị Zero', fontsize=12, fontweight='bold')
    ax2.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='50%')
    ax2.axvline(x=80, color='red', linestyle='--', linewidth=2, label='80%')
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, df_sparsity['Zero (%)'])):
        ax2.text(val + 2, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    # 3. Scatter plot: Zero vs Total Elements
    ax3 = fig.add_subplot(gs[1, 1])
    scatter = ax3.scatter(df_sparsity['Total Elements'], df_sparsity['Zero (%)'],
                         s=200, c=df_sparsity['Zero (%)'], cmap='RdYlGn_r',
                         edgecolor='black', linewidth=2, alpha=0.7)
    
    for idx, row in df_sparsity.iterrows():
        ax3.annotate(row['Nhóm'], 
                    (row['Total Elements'], row['Zero (%)']),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=8, fontweight='bold')
    
    ax3.set_xlabel('Tổng số phần tử', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Zero values (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Mối quan hệ: Kích thước vs Độ thưa', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='Zero %')
    ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap visualization
    ax4 = fig.add_subplot(gs[2, :])
    heatmap_data = df_sparsity[['Nhóm', 'Zero (%)', 'NaN (%)', 'Non-zero (%)']].set_index('Nhóm').T
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Percentage'}, linewidths=1, linecolor='black',
                ax=ax4, vmin=0, vmax=100)
    ax4.set_title('Ma trận độ thưa (Sparsity Heatmap)', fontsize=12, fontweight='bold', pad=10)
    ax4.set_ylabel('Loại giá trị', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Nhóm đặc trưng', fontsize=11, fontweight='bold')
    
    plt.suptitle('Phân tích độ thưa toàn diện (Comprehensive Sparsity Analysis)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    save_path = os.path.join(OUTPUT_DIR, '03_sparsity_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Đã lưu: {save_path}")
    plt.close()
    
    return df_sparsity

def plot_variance_analysis(all_features):
    """4. Phân tích phương sai - PHIÊN BẢN TIẾT KIỆM RAM"""
    print("\n📊 4. Phân tích phương sai (phiên bản tiết kiệm bộ nhớ)...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    variance_stats = []
    
    for name, df in all_features.items():
        print(f"   ⏳ Đang tính variance cho {name} ({df.shape[1]} cột)...")
        
        # Chỉ tính variance theo từng cột, không load toàn bộ vào bộ nhớ cùng lúc
        variances = []
        for col in df.columns:
            col_data = df[col]
            # Bỏ qua cột toàn NaN hoặc constant
            if col_data.nunique() <= 1:
                variances.append(0.0)
            else:
                # Dùng ddof=1 (sample variance) như pandas mặc định
                var_val = col_data.var(ddof=1, skipna=True)
                variances.append(0.0 if np.isnan(var_val) else var_val)
        
        variances = np.array(variances)
        
        variance_stats.append({
            'Nhóm': name,
            'Mean Var': float(variances.mean()),
            'Median Var': float(np.median(variances)),
            'Max Var': float(variances.max()),
            'Zero Var': int((variances == 0).sum()),
            'Low Var (<0.01)': int((variances < 0.01).sum()),
            'Medium Var (<0.1)': int(((variances >= 0.01) & (variances < 0.1)).sum()),
            'High Var (>=0.1)': int((variances >= 0.1).sum()),
            'Total': len(variances)
        })
    
    df_var = pd.DataFrame(variance_stats)
    
    # 1. Stacked bar: variance categories
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(df_var))
    width = 0.6
    
    p1 = ax1.bar(x, df_var['Zero Var'], width, label='Zero Var',
                color='#FF6B6B', edgecolor='black', linewidth=1)
    p2 = ax1.bar(x, df_var['Low Var (<0.01)'], width, bottom=df_var['Zero Var'],
                label='Low Var (<0.01)', color='#FFE66D', edgecolor='black', linewidth=1)
    p3 = ax1.bar(x, df_var['Medium Var (<0.1)'], width,
                bottom=df_var['Zero Var'] + df_var['Low Var (<0.01)'],
                label='Medium Var (<0.1)', color='#95E1D3', edgecolor='black', linewidth=1)
    p4 = ax1.bar(x, df_var['High Var (>=0.1)'], width,
                bottom=df_var['Zero Var'] + df_var['Low Var (<0.01)'] + df_var['Medium Var (<0.1)'],
                label='High Var (>=0.1)', color='#4ECDC4', edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Nhóm đặc trưng', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Số lượng features', fontsize=12, fontweight='bold')
    ax1.set_title('Phân loại features theo mức độ phương sai', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_var['Nhóm'], rotation=45, ha='right', fontsize=10)
    ax1.legend(loc='upper right', fontsize=10)
    
    # 2. Percentage of low variance features
    ax2 = fig.add_subplot(gs[1, 0])
    low_var_pct = (df_var['Zero Var'] + df_var['Low Var (<0.01)']) / df_var['Total'] * 100
    colors_pct = ['#FF6B6B' if v > 50 else '#FFE66D' if v > 20 else '#4ECDC4' 
                  for v in low_var_pct]
    bars = ax2.barh(df_var['Nhóm'], low_var_pct, color=colors_pct, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Tỷ lệ features variance thấp (%)', fontsize=11, fontweight='bold')
    ax2.set_title('% Features cần loại bỏ', fontsize=12, fontweight='bold')
    ax2.axvline(x=20, color='orange', linestyle='--', linewidth=2, label='20%')
    ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50%')
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()
    
    for bar, val in zip(bars, low_var_pct):
        ax2.text(val + 2, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    # 3. Mean variance comparison
    ax3 = fig.add_subplot(gs[1, 1])
    df_var_sorted = df_var.sort_values('Mean Var', ascending=True)
    bars = ax3.barh(df_var_sorted['Nhóm'], df_var_sorted['Mean Var'],
                    color=plt.cm.viridis(np.linspace(0, 1, len(df_var_sorted))),
                    edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Phương sai trung bình', fontsize=11, fontweight='bold')
    ax3.set_title('Mức độ phân tán dữ liệu', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    
    for bar, val in zip(bars, df_var_sorted['Mean Var']):
        ax3.text(val, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', ha='left', fontsize=8, fontweight='bold')
    
    plt.suptitle('Phân tích phương sai toàn diện (Variance Analysis)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    save_path = os.path.join(OUTPUT_DIR, '04_variance_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Đã lưu: {save_path}")
    plt.close()
    
    return df_var

def plot_sample_distributions(all_features, metadata, n_samples=6):
    """5. Phân tích phân bố mẫu của từng nhóm"""
    print("\n📊 5. Phân tích phân bố mẫu...")
    
    # Chọn ngẫu nhiên một vài features từ mỗi nhóm
    feature_groups = list(all_features.keys())[:n_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, group_name in enumerate(feature_groups):
        if idx >= len(axes):
            break
            
        df = all_features[group_name]
        
        # Lấy cột đầu tiên có variance > 0
        valid_cols = [col for col in df.columns if df[col].var() > 0]
        if valid_cols:
            sample_col = valid_cols[0]
            data = df[sample_col].dropna()
            
            # Tách theo label nếu có
            if 'label' in metadata.columns:
                benign_data = data[metadata['label'] == 0]
                malware_data = data[metadata['label'] == 1]
                
                axes[idx].hist([benign_data, malware_data], bins=50, 
                              label=['Benign', 'Malware'],
                              color=['#4ECDC4', '#FF6B6B'], 
                              alpha=0.7, edgecolor='black', linewidth=1)
                axes[idx].legend(fontsize=9)
            else:
                axes[idx].hist(data, bins=50, color='#4ECDC4', 
                              alpha=0.7, edgecolor='black', linewidth=1)
            
            axes[idx].set_title(f'{group_name}\n({sample_col})', 
                              fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Giá trị', fontsize=9)
            axes[idx].set_ylabel('Tần suất', fontsize=9)
            axes[idx].grid(True, alpha=0.3)
    
    # Ẩn các subplot không dùng
    for idx in range(len(feature_groups), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Phân bố mẫu theo nhóm đặc trưng (Sample Distributions)', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, '05_sample_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Đã lưu: {save_path}")
    plt.close()

def plot_correlation_heatmap(all_features, n_features=20):
    """6. Phân tích ma trận tương quan"""
    print("\n📊 6. Phân tích ma trận tương quan...")
    
    # Chọn nhóm có số chiều vừa phải để visualize
    selected_groups = []
    for name, df in all_features.items():
        if 10 <= df.shape[1] <= 100:
            selected_groups.append((name, df))
        if len(selected_groups) >= 3:
            break
    
    if not selected_groups:
        selected_groups = [(name, df) for name, df in list(all_features.items())[:3]]
    
    fig, axes = plt.subplots(1, len(selected_groups), figsize=(18, 5))
    if len(selected_groups) == 1:
        axes = [axes]
    
    for idx, (name, df) in enumerate(selected_groups):
        # Lấy n features đầu tiên
        sample_df = df.iloc[:, :min(n_features, df.shape[1])]
        
        # Tính correlation
        corr = sample_df.corr()
        
        # Vẽ heatmap
        sns.heatmap(corr, cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[idx],
                   vmin=-1, vmax=1)
        axes[idx].set_title(f'{name}\n(Top {sample_df.shape[1]} features)', 
                          fontsize=11, fontweight='bold')
        axes[idx].tick_params(labelsize=7)
    
    plt.suptitle('Ma trận tương quan các nhóm đặc trưng (Correlation Heatmaps)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, '06_correlation_heatmaps.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Đã lưu: {save_path}")
    plt.close()

def plot_pca_visualization(all_features, metadata, n_components=3):
    """7. Phân tích PCA và visualization - ĐÃ SỬA LỖI SAMPLE"""
    print("\n📊 7. Phân tích PCA và visualization...")
    
    print("   ⏳ Đang kết hợp features và đồng bộ sample với metadata...")
    
    max_samples_for_pca = 50000
    
    # === KHÓA CHẶN: Luôn lấy cùng một tập mẫu cho cả features và metadata ===
    if metadata.shape[0] > max_samples_for_pca:
        metadata_sampled = metadata.sample(n=max_samples_for_pca, random_state=42).sort_index()
        sample_indices = metadata_sampled.index
    else:
        metadata_sampled = metadata.copy()
        sample_indices = metadata.index
    
    # Lấy đúng các dòng tương ứng từ tất cả các nhóm features
    sample_data = []
    for name, df in all_features.items():
        if df.index.is_monotonic_increasing:
            sampled_df = df.loc[sample_indices]
        else:
            sampled_df = df.iloc[sample_indices] if df.shape[0] == len(sample_indices) else df.sample(n=max_samples_for_pca, random_state=42)
        sample_data.append(sampled_df)
        print(f"      ✓ {name}: {sampled_df.shape}")
    
    all_data = pd.concat(sample_data, axis=1)
    
    # Đảm bảo metadata cũng đúng thứ tự
    metadata_sampled = metadata.loc[all_data.index.intersection(metadata.index)]
    if len(metadata_sampled) < len(all_data):
        print("   ⚠️ Cảnh báo: Một số chỉ số metadata không khớp, sẽ dùng label còn lại")
        metadata_sampled = metadata.reindex(all_data.index).iloc[:len(all_data)]
    
    print(f"   → Dữ liệu PCA cuối cùng: {all_data.shape}, metadata: {len(metadata_sampled)}")
    
    # Xử lý missing values
    print("   ⏳ Đang xử lý missing values...")
    all_data = all_data.fillna(0)
    
    # Chuẩn hóa
    print("   ⏳ Đang chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(all_data)
    except MemoryError:
        print("   ⚠️ RAM không đủ, giảm xuống 20k mẫu...")
        all_data = all_data.sample(n=20000, random_state=42)
        metadata_sampled = metadata_sampled.sample(n=20000, random_state=42)
        X_scaled = scaler.fit_transform(all_data)
    
    # Fit PCA
    print("   ⏳ Đang thực hiện PCA...")
    pca = PCA(n_components=min(50, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    try:
        X_pca = pca.fit_transform(X_scaled)
    except MemoryError:
        print("      ⚠️  RAM không đủ cho PCA, giảm xuống 20 components...")
        pca = PCA(n_components=20)
        X_pca = pca.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # 1. Explained variance ratio
    ax1 = fig.add_subplot(gs[0, :])
    explained_var = pca.explained_variance_ratio_[:20]
    cumsum_var = np.cumsum(explained_var)
    
    x = np.arange(len(explained_var))
    ax1.bar(x, explained_var * 100, color='#4ECDC4', 
           edgecolor='black', linewidth=1.5, label='Individual')
    ax1.plot(x, cumsum_var * 100, color='#FF6B6B', marker='o', 
            linewidth=2.5, markersize=6, label='Cumulative')
    
    ax1.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Explained Variance (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Phương sai giải thích bởi PCA (Top 20 components)', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (ind, cum) in enumerate(zip(explained_var[:5], cumsum_var[:5])):
        ax1.text(i, ind * 100 + 1, f'{ind*100:.1f}%', 
                ha='center', fontsize=8, fontweight='bold')
        ax1.text(i, cum * 100 + 2, f'{cum*100:.1f}%', 
                ha='center', fontsize=8, color='red', fontweight='bold')
    
    # 2. 2D PCA scatter (PC1 vs PC2)
    ax2 = fig.add_subplot(gs[1, 0])
    if 'label' in metadata_sampled.columns:
        for label, color, name in [(0, '#4ECDC4', 'Benign'), (1, '#FF6B6B', 'Malware')]:
            mask = metadata_sampled['label'] == label
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=color, label=name, alpha=0.6, s=20, edgecolor='black', linewidth=0.5)
        ax2.legend(fontsize=9)
    else:
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='#4ECDC4', 
                   alpha=0.6, s=20, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                  fontsize=10, fontweight='bold')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                  fontsize=10, fontweight='bold')
    ax2.set_title('PCA: PC1 vs PC2', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 2D PCA scatter (PC2 vs PC3)
    ax3 = fig.add_subplot(gs[1, 1])
    if 'label' in metadata_sampled.columns:
        for label, color, name in [(0, '#4ECDC4', 'Benign'), (1, '#FF6B6B', 'Malware')]:
            mask = metadata_sampled['label'] == label
            ax3.scatter(X_pca[mask, 1], X_pca[mask, 2], 
                       c=color, label=name, alpha=0.6, s=20, edgecolor='black', linewidth=0.5)
        ax3.legend(fontsize=9)
    else:
        ax3.scatter(X_pca[:, 1], X_pca[:, 2], c='#4ECDC4', 
                   alpha=0.6, s=20, edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                  fontsize=10, fontweight='bold')
    ax3.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', 
                  fontsize=10, fontweight='bold')
    ax3.set_title('PCA: PC2 vs PC3', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 3D PCA scatter
    ax4 = fig.add_subplot(gs[1, 2], projection='3d')
    if 'label' in metadata_sampled.columns:
        for label, color, name in [(0, '#4ECDC4', 'Benign'), (1, '#FF6B6B', 'Malware')]:
            mask = metadata_sampled['label'] == label
            ax4.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                       c=color, label=name, alpha=0.5, s=10)
        ax4.legend(fontsize=8)
    else:
        ax4.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                   c='#4ECDC4', alpha=0.5, s=10)
    
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=8)
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=8)
    ax4.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=8)
    ax4.set_title('PCA 3D Visualization', fontsize=10, fontweight='bold')
    
    # 5. Scree plot
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(range(1, len(explained_var)+1), explained_var * 100, 
            marker='o', linewidth=2, markersize=6, color='#4ECDC4')
    ax5.set_xlabel('Component Number', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Explained Variance (%)', fontsize=10, fontweight='bold')
    ax5.set_title('Scree Plot', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative variance threshold
    ax6 = fig.add_subplot(gs[2, 1])
    n_components_80 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.80) + 1
    n_components_90 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1
    n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
    
    thresholds = [80, 90, 95]
    n_components_list = [n_components_80, n_components_90, n_components_95]
    colors_thresh = ['#4ECDC4', '#FFE66D', '#FF6B6B']
    
    bars = ax6.bar([str(t) + '%' for t in thresholds], n_components_list, 
                   color=colors_thresh, edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('Số components cần thiết', fontsize=10, fontweight='bold')
    ax6.set_title('Components để đạt ngưỡng variance', fontsize=11, fontweight='bold')
    
    for bar, val in zip(bars, n_components_list):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 5, 
                f'{val}', ha='center', fontsize=10, fontweight='bold')
    
    # 7. Recommendation table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    recommendations = [
        ['Mục tiêu', 'Số PC', 'Giảm chiều'],
        ['80% variance', str(n_components_80), f'{(1-n_components_80/all_data.shape[1])*100:.1f}%'],
        ['90% variance', str(n_components_90), f'{(1-n_components_90/all_data.shape[1])*100:.1f}%'],
        ['95% variance', str(n_components_95), f'{(1-n_components_95/all_data.shape[1])*100:.1f}%'],
        ['Chiều gốc', str(all_data.shape[1]), '0%']
    ]
    
    table = ax7.table(cellText=recommendations, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(recommendations)):
        if i == 0:
            for j in range(3):
                table[(i, j)].set_facecolor('#4ECDC4')
                table[(i, j)].set_text_props(weight='bold', color='white')
        elif i == len(recommendations) - 1:
            for j in range(3):
                table[(i, j)].set_facecolor('#FFE66D')
                table[(i, j)].set_text_props(weight='bold')
    
    plt.suptitle('Phân tích PCA và khả năng giảm chiều (PCA & Dimensionality Reduction)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    save_path = os.path.join(OUTPUT_DIR, '07_pca_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Đã lưu: {save_path}")
    plt.close()
    
    return pca, X_pca

def generate_comprehensive_report(metadata, all_features, df_sparsity, df_var):
    """8. Tạo báo cáo tổng hợp"""
    print("\n📊 8. Tạo báo cáo tổng hợp...")
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    
    total_dims = sum(df.shape[1] for df in all_features.values())
    total_samples = len(metadata)
    
    # 1. Dataset Overview
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    overview_data = [
        ['Tổng số mẫu', f'{total_samples:,}'],
        ['Tổng số chiều', f'{total_dims:,}'],
        ['Tỷ lệ mẫu/chiều', f'{total_samples/total_dims:.4f}'],
        ['Số nhóm đặc trưng', f'{len(all_features)}'],
        ['Độ thưa trung bình', f'{df_sparsity["Zero (%)"].mean():.2f}%'],
        ['Features variance thấp', f'{df_var["Low Var (<0.01)"].sum()}/{total_dims}']
    ]
    
    table1 = ax1.table(cellText=overview_data, cellLoc='left', loc='center',
                      colWidths=[0.4, 0.6])
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(1, 3)
    
    for i in range(len(overview_data)):
        table1[(i, 0)].set_facecolor('#4ECDC4')
        table1[(i, 0)].set_text_props(weight='bold', color='white')
        table1[(i, 1)].set_facecolor('#F8F9FA')
        table1[(i, 1)].set_text_props(weight='bold', fontsize=13)
    
    ax1.set_title('TỔNG QUAN DỮ LIỆU (DATASET OVERVIEW)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 2. Class Balance
    ax2 = fig.add_subplot(gs[1, 0])
    label_counts = metadata['label'].value_counts().sort_index()
    label_mapping = {-1: 'Unknown', 0: 'Benign', 1: 'Malware'}
    colors = {-1: '#95E1D3', 0: '#4ECDC4', 1: '#FF6B6B'}
    
    labels = [label_mapping.get(k, f'Label {k}') for k in label_counts.index]
    actual_colors = [colors.get(k, '#CCCCCC') for k in label_counts.index]
    
    bars = ax2.bar(labels, label_counts.values, color=actual_colors, 
                  edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Số lượng', fontsize=11, fontweight='bold')
    ax2.set_title('Cân bằng lớp', fontsize=12, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # 3. Feature dimensions
    ax3 = fig.add_subplot(gs[1, 1])
    dims = {name: df.shape[1] for name, df in all_features.items()}
    sorted_dims = sorted(dims.items(), key=lambda x: x[1], reverse=True)[:8]
    
    names = [x[0] for x in sorted_dims]
    values = [x[1] for x in sorted_dims]
    colors_grad = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    bars = ax3.barh(names, values, color=colors_grad, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Số chiều', fontsize=11, fontweight='bold')
    ax3.set_title('Top nhóm đặc trưng', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    
    for bar, val in zip(bars, values):
        ax3.text(val, bar.get_y() + bar.get_height()/2, 
                f'{val}', va='center', ha='left', fontsize=9, fontweight='bold')
    
    # 4. Sparsity comparison
    ax4 = fig.add_subplot(gs[2, 0])
    top_sparse = df_sparsity.nlargest(8, 'Zero (%)')
    
    bars = ax4.barh(top_sparse['Nhóm'], top_sparse['Zero (%)'], 
                    color='#FF6B6B', edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Zero values (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Top nhóm thưa nhất', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    
    for bar, val in zip(bars, top_sparse['Zero (%)']):
        ax4.text(val + 2, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    # 5. Low variance features
    ax5 = fig.add_subplot(gs[2, 1])
    top_low_var = df_var.nlargest(8, 'Low Var (<0.01)')
    
    bars = ax5.barh(top_low_var['Nhóm'], top_low_var['Low Var (<0.01)'], 
                    color='#FFE66D', edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Số features', fontsize=11, fontweight='bold')
    ax5.set_title('Features variance thấp', fontsize=12, fontweight='bold')
    ax5.invert_yaxis()
    
    for bar, val in zip(bars, top_low_var['Low Var (<0.01)']):
        ax5.text(val, bar.get_y() + bar.get_height()/2, 
                f'{int(val)}', va='center', ha='left', fontsize=9, fontweight='bold')
    
    # 6. Recommendations
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    recommendations_text = "KHUYẾN NGHỊ XỬ LÝ DỮ LIỆU:\n\n"
    
    # Class imbalance
    if 0 in label_counts.index and 1 in label_counts.index:
        imbalance_ratio = max(label_counts[0], label_counts[1]) / min(label_counts[0], label_counts[1])
        if imbalance_ratio > 2:
            recommendations_text += f"1. ⚠️ XỬ LÝ MẤT CÂN BẰNG (ratio {imbalance_ratio:.1f}:1)\n"
            recommendations_text += "   → Áp dụng: SMOTE, ADASYN, class_weight='balanced'\n\n"
    
    # Dimensionality
    avg_sparsity = df_sparsity['Zero (%)'].mean()
    if total_dims > 1000:
        recommendations_text += f"2. ⚠️ GIẢM CHIỀU (hiện tại: {total_dims:,} chiều)\n"
        recommendations_text += "   → PCA: Giảm xuống 100-300 chiều\n"
        recommendations_text += "   → SelectKBest hoặc VarianceThreshold\n\n"
    
    # Low variance
    total_low_var = df_var['Low Var (<0.01)'].sum()
    if total_low_var > 50:
        recommendations_text += f"3. ⚠️ LOẠI BỎ {total_low_var} features variance thấp\n"
        recommendations_text += "   → Dùng: VarianceThreshold(threshold=0.01)\n\n"
    
    # Model selection
    recommendations_text += "4. ✅ MÔ HÌNH KHUYẾN NGHỊ:\n"
    if avg_sparsity > 70:
        recommendations_text += "   → XGBoost, LightGBM (tốt với sparse data)\n"
        recommendations_text += "   → Random Forest (robust)\n"
    else:
        recommendations_text += "   → Ensemble: XGBoost, LightGBM, CatBoost\n"
        recommendations_text += "   → Neural Network (sau khi giảm chiều)\n"
    
    ax6.text(0.05, 0.95, recommendations_text,
            transform=ax6.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#4ECDC4', linewidth=2),
            family='monospace')
    
    plt.suptitle('BÁO CÁO TỔNG HỢP PHÂN TÍCH DỮ LIỆU EMBER 2018', 
                fontsize=15, fontweight='bold', y=0.98)
    
    save_path = os.path.join(OUTPUT_DIR, '08_comprehensive_report.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Đã lưu: {save_path}")
    plt.close()

def save_text_report(metadata, all_features, df_sparsity, df_var):
    """9. Lưu báo cáo dạng text"""
    print("\n📄 9 Tạo báo cáo text...")
    
    report_path = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BÁO CÁO PHÂN TÍCH DỮ LIỆU EMBER 2018\n")
        f.write("="*80 + "\n\n")
        
        # 1. Dataset Overview
        f.write("1. TỔNG QUAN DỮ LIỆU\n")
        f.write("-"*80 + "\n")
        total_dims = sum(df.shape[1] for df in all_features.values())
        total_samples = len(metadata)
        
        f.write(f"Tổng số mẫu:              {total_samples:,}\n")
        f.write(f"Tổng số chiều:            {total_dims:,}\n")
        f.write(f"Tỷ lệ mẫu/chiều:          {total_samples/total_dims:.4f}\n")
        f.write(f"Số nhóm đặc trưng:        {len(all_features)}\n")
        f.write(f"Độ thưa trung bình:       {df_sparsity['Zero (%)'].mean():.2f}%\n")
        f.write(f"Features variance thấp:   {df_var['Low Var (<0.01)'].sum()}/{total_dims}\n\n")
        
        # 2. Class Distribution
        f.write("2. PHÂN BỐ LỚP\n")
        f.write("-"*80 + "\n")
        label_counts = metadata['label'].value_counts().sort_index()
        label_mapping = {-1: 'Unknown', 0: 'Benign', 1: 'Malware'}
        
        for label, count in label_counts.items():
            label_name = label_mapping.get(label, f'Label {label}')
            pct = count / len(metadata) * 100
            f.write(f"{label_name:15s}: {count:8,} mẫu ({pct:6.2f}%)\n")
        
        if 0 in label_counts.index and 1 in label_counts.index:
            imbalance_ratio = max(label_counts[0], label_counts[1]) / min(label_counts[0], label_counts[1])
            f.write(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1\n")
            
            if imbalance_ratio > 3:
                f.write("⚠️  CẢNH BÁO: Dữ liệu MẤT CÂN BẰNG NGHIÊM TRỌNG!\n")
            elif imbalance_ratio > 1.5:
                f.write("⚠️  Dữ liệu hơi mất cân bằng\n")
        f.write("\n")
        
        # 3. Feature Dimensions
        f.write("3. CHIỀU DỮ LIỆU\n")
        f.write("-"*80 + "\n")
        dims = {name: df.shape[1] for name, df in all_features.items()}
        
        for name, dim in sorted(dims.items(), key=lambda x: x[1], reverse=True):
            pct = dim/total_dims*100
            f.write(f"{name:30s}: {dim:5,} chiều ({pct:5.1f}%)\n")
        f.write("\n")
        
        # 4. Sparsity Analysis
        f.write("4. PHÂN TÍCH ĐỘ THƯA\n")
        f.write("-"*80 + "\n")
        for idx, row in df_sparsity.iterrows():
            f.write(f"{row['Nhóm']:30s}: Zero={row['Zero (%)']:5.1f}%, "
                   f"NaN={row['NaN (%)']:5.1f}%, "
                   f"Total Sparse={row['Zero (%)'] + row['NaN (%)']:5.1f}%\n")
        
        avg_sparsity = df_sparsity['Zero (%)'].mean()
        f.write(f"\nĐộ thưa trung bình: {avg_sparsity:.2f}%\n")
        
        if avg_sparsity > 80:
            f.write("⚠️  Dữ liệu RẤT THƯA - Nên dùng Tree-based models\n")
        elif avg_sparsity > 50:
            f.write("⚠️  Dữ liệu khá thưa - Phù hợp: XGBoost, LightGBM\n")
        f.write("\n")
        
        # 5. Variance Analysis
        f.write("5. PHÂN TÍCH PHƯƠNG SAI\n")
        f.write("-"*80 + "\n")
        for idx, row in df_var.iterrows():
            f.write(f"{row['Nhóm']:30s}: Zero-Var={row['Zero Var']:4d}, "
                   f"Low-Var={row['Low Var (<0.01)']:4d}, "
                   f"Total={row['Total']:4d}\n")
        
        total_low_var = df_var['Low Var (<0.01)'].sum()
        f.write(f"\nTổng features variance thấp: {total_low_var}/{total_dims} "
               f"({total_low_var/total_dims*100:.1f}%)\n\n")
        
        
        # 7. Recommendations
        f.write("7. KHUYẾN NGHỊ XỬ LÝ\n")
        f.write("="*80 + "\n\n")
        
        # Class imbalance
        if 0 in label_counts.index and 1 in label_counts.index:
            imbalance_ratio = max(label_counts[0], label_counts[1]) / min(label_counts[0], label_counts[1])
            if imbalance_ratio > 2:
                f.write(f"✦ XỬ LÝ MẤT CÂN BẰNG LỚP (ratio {imbalance_ratio:.1f}:1)\n")
                f.write("  Phương pháp:\n")
                f.write("  - SMOTE (Synthetic Minority Over-sampling)\n")
                f.write("  - ADASYN (Adaptive Synthetic Sampling)\n")
                f.write("  - class_weight='balanced' trong model\n")
                f.write("  - Stratified K-Fold cross-validation\n\n")
        
        # Dimensionality reduction
        if total_dims > 1000:
            f.write(f"✦ GIẢM CHIỀU (hiện tại: {total_dims:,} chiều)\n")
            f.write("  Phương pháp ưu tiên:\n")
            f.write("  - PCA: Giảm xuống 100-300 chiều (giữ 90-95% variance)\n")
            f.write("  - SelectKBest: Chọn top K features quan trọng\n")
            f.write("  - VarianceThreshold: Loại bỏ features variance thấp\n")
            if avg_sparsity > 70:
                f.write("  - TruncatedSVD: Phù hợp với sparse data\n")
            f.write(f"  - Feature Selection: Có thể giảm ~{total_low_var} chiều\n\n")
        
        # Low variance features
        if total_low_var > 50:
            f.write(f"✦ LOẠI BỎ FEATURES VARIANCE THẤP ({total_low_var} features)\n")
            f.write("  Code mẫu:\n")
            f.write("  from sklearn.feature_selection import VarianceThreshold\n")
            f.write("  selector = VarianceThreshold(threshold=0.01)\n")
            f.write("  X_filtered = selector.fit_transform(X)\n\n")
        
        # Model recommendations
        f.write("✦ MÔ HÌNH KHUYẾN NGHỊ\n")
        f.write("  Dựa trên đặc điểm dữ liệu:\n")
        
        if avg_sparsity > 70:
            f.write("  1. XGBoost (xử lý tốt sparse data, nhanh)\n")
            f.write("  2. LightGBM (hiệu quả với high-dimensional sparse data)\n")
            f.write("  3. Random Forest (robust, không cần scaling)\n")
            f.write("  4. CatBoost (tốt với categorical features)\n")
        else:
            f.write("  1. Ensemble: XGBoost, LightGBM (hiệu quả cao)\n")
            f.write("  2. Random Forest (dễ tune, ít overfit)\n")
            f.write("  3. Neural Network (sau khi giảm chiều với PCA)\n")
            f.write("  4. SVM với kernel RBF (cho dataset nhỏ hơn)\n")
        
        f.write("\n  Lưu ý:\n")
        f.write("  - Sử dụng Stratified K-Fold để đảm bảo class balance\n")
        f.write("  - Cross-validation với ít nhất 5 folds\n")
        f.write("  - Hyperparameter tuning với GridSearchCV hoặc RandomizedSearchCV\n")
        f.write("  - Đánh giá với nhiều metrics: Accuracy, Precision, Recall, F1, AUC-ROC\n\n")
        
        # Sample size recommendation
        min_recommended = total_dims * 10
        good_recommended = total_dims * 50
        
        f.write("✦ KHUYẾN NGHỊ VỀ KÍCH THƯỚC MẪU\n")
        f.write(f"  Số mẫu hiện tại:   {total_samples:,}\n")
        f.write(f"  Tối thiểu (10x):   {min_recommended:,} mẫu\n")
        f.write(f"  Tốt (50x):         {good_recommended:,} mẫu\n")
        
        if total_samples < min_recommended:
            shortage = min_recommended - total_samples
            f.write(f"\n  ⚠️  CẢNH BÁO: Thiếu {shortage:,} mẫu!\n")
            f.write("  → BẮT BUỘC phải giảm chiều hoặc tăng dữ liệu\n")
        elif total_samples < good_recommended:
            f.write("\n  ⚠️  Nên giảm chiều để tránh overfitting\n")
        else:
            f.write("\n  ✅ Kích thước mẫu TỐT\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("HẾT BÁO CÁO\n")
        f.write("="*80 + "\n")
    
    print(f"   ✓ Đã lưu: {report_path}")

def main():
    """Hàm chính"""
    print("\n" + "="*100)
    print("🔬 PHÂN TÍCH TOÀN DIỆN DỮ LIỆU EMBER 2018")
    print("   Trực quan hóa chi tiết và đưa ra khuyến nghị xử lý")
    print("="*100)
    
    # Load dữ liệu
    metadata, all_features = load_all_data(sample_size=SAMPLE_SIZE)
    
    # 1. Phân tích phân bố lớp
    plot_class_distribution(metadata)
    
    # 2. Phân tích số chiều
    plot_feature_dimensions(all_features)
    
    # 3. Phân tích độ thưa
    df_sparsity = plot_sparsity_analysis(all_features)
    
    # 4. Phân tích phương sai
    df_var = plot_variance_analysis(all_features)
    
    # 5. Phân tích phân bố mẫu
    plot_sample_distributions(all_features, metadata, n_samples=6)
    
    # 6. Phân tích correlation
    plot_correlation_heatmap(all_features, n_features=20)
    
    # 7. Phân tích PCA
    pca, X_pca = plot_pca_visualization(all_features, metadata, n_components=3)
    
    # 8. Báo cáo tổng hợp
    generate_comprehensive_report(metadata, all_features, df_sparsity, df_var)
    
    # 9. Lưu báo cáo text
    save_text_report(metadata, all_features, df_sparsity, df_var)
    
    # Lưu summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Samples',
            'Total Features', 
            'Benign Samples',
            'Malware Samples',
            'Unknown Samples',
            'Avg Sparsity (%)',
            'Low Variance Features',
            'Feature Groups'
        ],
        'Value': [
            len(metadata),
            sum(df.shape[1] for df in all_features.values()),
            (metadata['label']==0).sum() if 0 in metadata['label'].values else 0,
            (metadata['label']==1).sum() if 1 in metadata['label'].values else 0,
            (metadata['label']==-1).sum() if -1 in metadata['label'].values else 0,
            round(df_sparsity['Zero (%)'].mean(), 2),
            df_var['Low Var (<0.01)'].sum(),
            len(all_features)
        ]
    })
    
    summary_path = os.path.join(OUTPUT_DIR, 'summary_statistics.csv')
    summary_stats.to_csv(summary_path, index=False)
    print(f"\n   ✓ Đã lưu: {summary_path}")
    
    # In danh sách các file đã tạo
    print("\n" + "="*100)
    print("✅ HOÀN THÀNH! Đã tạo các file trong thư mục '{}':\n".format(OUTPUT_DIR))
    
    visualization_files = [
        "01_class_distribution.png        - Phân tích phân bố lớp chi tiết",
        "02_feature_dimensions.png        - Phân tích số chiều và tỷ lệ",
        "03_sparsity_analysis.png         - Phân tích độ thưa toàn diện",
        "04_variance_analysis.png         - Phân tích phương sai chi tiết",
        "05_sample_distributions.png      - Phân bố mẫu theo nhóm",
        "06_correlation_heatmaps.png      - Ma trận tương quan",
        "07_pca_analysis.png              - PCA và giảm chiều",
        "08_comprehensive_report.png      - Báo cáo tổng hợp",
        "analysis_report.txt              - Báo cáo chi tiết dạng text",
        "summary_statistics.csv           - Thống kê tóm tắt"
    ]
    
    for i, file_desc in enumerate(visualization_files, 1):
        print(f"   {i:2d}. {file_desc}")
    
    print("\n" + "="*100)
    print("💡 GỢI Ý: Xem báo cáo tổng hợp tại '09_comprehensive_report.png'")
    print("          và file text chi tiết tại 'analysis_report.txt'")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()