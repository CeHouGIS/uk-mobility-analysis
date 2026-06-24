# python src/s4_clustering/cluster_profiles_individual.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import ternary

warnings.filterwarnings("ignore")

# --- 1. 配置 ---
# DATA_PATH = "/nas/houce/UK_mobility/individual mobility profile previous/England_individual_1127.csv"
DATA_PATH = "/nas/houce/UK_mobility/data_20251226_new/ML/England_ML.csv"
FIG_SAVE_PATH = "/nas/houce/UK_mobility/data_20251226_new/ML_results"
BASE_FREQ_COLS = ['workplace_frequency','t1_frequency','t3_frequency','t5_frequency','t6_frequency','t8_frequency','t9_frequency']

def load_and_filter_data(path: str, select_home_same: bool) -> pd.DataFrame:
    """加载并筛选初始数据。"""
    print("[INFO] 1. Loading and filtering data...")
    df = pd.read_csv(path)
    print(f"      Initial data shape: {df.shape}")
    df_2020 = df[(df['month'] == 202002) | (df['month'] == 202003)]
    df_2021 = df[df['month'] == 202103]
    df_select = df_2020.merge(df_2021, on='device_aid', how='inner', suffixes=('_2020', '_2021'))
    return df_select

def calculate_regional_ratios(df: pd.DataFrame, year: str) -> pd.DataFrame:
    """计算指定年份的区域（LAD）平均值，并以此计算每个个体的特征比率。"""
    print(f"      Calculating regional averages and ratios for {year}...")
    
    lad_col = f'home_{year}_lad_code'
    feature_cols = [f'home_hours_{year}', f'workplace_days_{year}'] + [f'{col}_{year}' for col in BASE_FEATURE_COLS]
    
    lad_mean = df.groupby(lad_col)[feature_cols].mean().reset_index()
    df = df.merge(lad_mean, on=lad_col, how='left', suffixes=('', '_lad_mean'))
    
    for col in feature_cols:
        mean_col = f'{col}_lad_mean'
        ratio_col = f'{col}_ratio'
        df[ratio_col] = df[col] / (df[mean_col] + 1e-6)
        
    df.drop(columns=[f'{col}_lad_mean' for col in feature_cols], inplace=True)
    return df

def perform_yearly_clustering(df: pd.DataFrame, year: str, k: int) -> KMeans:
    """对指定年份的数据进行独立的聚类，并返回KMeans模型。"""
    print(f"\n[INFO] Performing clustering for {year} with k={k}...")
    
    ratio_cols = [f'home_hours_{year}_ratio', f'workplace_days_{year}_ratio'] + [f'{col}_{year}_ratio' for col in BASE_FEATURE_COLS]
    X = df[ratio_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    
    print(f"      Clustering for {year} complete.")
    return kmeans

def align_cluster_labels(kmeans_ref: KMeans, kmeans_target: KMeans, target_labels: np.ndarray) -> np.ndarray:
    """
    使用匈牙利算法对齐目标聚类标签，使其与参考聚类对齐。
    """
    print("\n[INFO] Aligning cluster labels between 2020 and 2021...")
    
    dist_matrix = cdist(kmeans_ref.cluster_centers_, kmeans_target.cluster_centers_, 'euclidean')
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    label_mapping = {old_label: new_label for new_label, old_label in zip(row_ind, col_ind)}
    aligned_labels = np.array([label_mapping[label] for label in target_labels])
    
    print("      Label alignment complete.")
    print(f"      Label mapping (2021_old -> 2020_aligned): {label_mapping}")
    return aligned_labels

def analyze_and_visualize_yearly_clusters(df: pd.DataFrame, year: str, save_path: str):
    """分析并可视化单年聚类结果。"""
    print(f"\n[INFO] Analyzing and visualizing clusters for {year}...")
    
    cluster_col = f'cluster_{year}'
    
    # --- 修改部分：使用原始特征列进行均值分析 ---
    original_feature_cols = [f'home_hours_{year}', f'workplace_days_{year}'] + [f'{col}_{year}' for col in BASE_FEATURE_COLS]

    # 简化列名以便绘图 (去掉年份后缀)
    readable_cols = ['home', 'workplace', 't1 (Eating)', 't3 (Attraction)', 't5 (Health)', 
                     't6 (Infra)', 't8 (Retail)', 't9 (Transport)']

    # 1. 特征均值分析 (使用原始值)
    cluster_means = df.groupby(cluster_col)[original_feature_cols].mean()
    cluster_means.columns = readable_cols
    print(f"--- Cluster Original Feature Means for {year} ---")
    print(cluster_means)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Original Feature Means by Cluster - {year}')
    plt.savefig(os.path.join(save_path, f'feature_means_original_values_{year}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # --- PCA 可视化部分保持不变 (基于比率特征) ---
    ratio_cols = [f'home_hours_{year}_ratio', f'workplace_days_{year}_ratio'] + [f'{col}_{year}_ratio' for col in BASE_FEATURE_COLS]
    X = df[ratio_cols].values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df[cluster_col], cmap='tab10', s=10, alpha=0.7)
    plt.title(f'PCA of Clusters (on Ratio Features) - {year}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    legend = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend)
    plt.savefig(os.path.join(save_path, f'pca_clusters_{year}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def analyze_transition(df: pd.DataFrame, save_path: str):
    """分析2020到2021的聚类转移情况。"""
    print("\n[INFO] Analyzing cluster transitions from 2020 to 2021...")
    
    transition_matrix = pd.crosstab(df['cluster_2020'], df['cluster_2021'])
    print("--- Transition Matrix (Counts) ---")
    print(transition_matrix)
    
    change_rate = (df['cluster_2020'] != df['cluster_2021']).mean()
    print(f"\n      Overall change rate: {change_rate:.2%}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Cluster Transition Matrix (2020 to 2021)')
    plt.xlabel('2021 Clusters (Aligned)')
    plt.ylabel('2020 Clusters')
    plt.savefig(os.path.join(save_path, 'transition_matrix_counts.png'), dpi=300, bbox_inches='tight')
    plt.show()

    transition_percent = transition_matrix.div(transition_matrix.sum(axis=1), axis=0) * 100
    transition_percent = transition_percent.round(2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_percent, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Cluster Transition Matrix (%) (2020 to 2021)')
    plt.xlabel('2021 Clusters (Aligned)')
    plt.ylabel('2020 Clusters')
    plt.savefig(os.path.join(save_path, 'transition_matrix_percent.png'), dpi=300, bbox_inches='tight')
    plt.show()

def visualize_ternary_distribution(df: pd.DataFrame, year: str, save_path: str):
    """
    使用三元图可视化 Home, Workplace, Amenities 的时间/天数分布。
    """
    print(f"\n[INFO] Generating ternary plot for {year}...")

    # 1. 准备数据
    # 创建设施活动天数的总和
    # df[f'amenities_days_{year}'] = df[amenity_cols].sum(axis=1)

    # 定义用于绘图的列
    # 注意：ternary的坐标轴顺序是 (bottom, right, left)
    # 我们将映射为 (Home, Amenities, Workplace)
    home_col = f'home_hours_{year}'
    amenities_col = f'hours_amenties_{year}_all'
    workplace_col = f'workplace_hours_{year}'


    # FIX: plot the ACTUAL hours composition (each person's home/workplace/amenities
    # share of their total hours). The previous version divided each component by its
    # own LAD mean and then re-closed to sum 1, which forced every individual to the
    # centre of the triangle (~1/3, 1/3, 1/3) and produced a meaningless central blob.
    comp = df[[home_col, amenities_col, workplace_col]].dropna()
    comp = comp[comp.sum(axis=1) > 0]
    data_points = comp.values

    # 2. 数据处理：网格计数
    scale = 100
    counts = dict()

    for (a, b, c) in data_points:
        # 归一化，使其总和为1
        total = a + b + c
        if total == 0: continue
        a, b, c = a/total, b/total, c/total

        i = int(round(a * scale))
        j = int(round(b * scale))
        k = scale - i - j

        if k < 0: continue

        coord = (i, j, k)
        if coord in counts:
            counts[coord] += 1
        else:
            counts[coord] = 1

    # 3. 绘制三元热力图
    fig, tax = ternary.figure(scale=scale)
    fig.set_size_inches(10, 8)

    tax.heatmap(counts, style="triangular", cmap='YlGnBu', colorbar=True)

    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", multiple=10)
    tax.gridlines(color="blue", multiple=1, linewidth=0.5, alpha=0.1)

    fontsize = 12
    tax.ticks(axis='lbr', linewidth=1, multiple=20, offset=0.025, tick_formats="%.1f")

    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.set_title(f"Home, Workplace, Amenities Distribution ({year})", fontsize=18)

    # 标签与数据轴匹配 (bottom, right, left) -> (Home, Amenities, Workplace)
    tax.bottom_axis_label("Home Hours (%)", offset=0.06, fontsize=fontsize)
    tax.right_axis_label("Amenities Hours (%)", offset=0.16, fontsize=fontsize)
    tax.left_axis_label("Workplace Hours (%)", offset=0.16, fontsize=fontsize)

    tax.get_axes().set_facecolor('white')
    
    plt.savefig(os.path.join(save_path, f'ternary_distribution_{year}.png'), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    os.makedirs(FIG_SAVE_PATH, exist_ok=True)
    
    # --- 数据加载与特征工程 ---
    df_select = load_and_filter_data(DATA_PATH, select_home_same=True)
    
    print("\n[INFO] 2. Performing feature engineering for both years...")
    df_select = calculate_regional_ratios(df_select, '2020')
    df_select = calculate_regional_ratios(df_select, '2021')
    
    # --- 为两年分别进行聚类 ---
    OPTIMAL_K = 7
    
    kmeans_2020 = perform_yearly_clustering(df_select, '2020', k=OPTIMAL_K)
    kmeans_2021 = perform_yearly_clustering(df_select, '2021', k=OPTIMAL_K)
    
    # --- 对齐2021年的标签以匹配2020年 ---
    aligned_labels_2021 = align_cluster_labels(kmeans_2020, kmeans_2021, kmeans_2021.labels_)
    
    df_select['cluster_2020'] = kmeans_2020.labels_
    df_select['cluster_2021'] = aligned_labels_2021
    
    # --- 分别对每年的结果进行分析和可视化 ---
    analyze_and_visualize_yearly_clusters(df_select, '2020', save_path=FIG_SAVE_PATH)
    analyze_and_visualize_yearly_clusters(df_select, '2021', save_path=FIG_SAVE_PATH)
    
    # --- 对比分析：聚类转移 ---
    analyze_transition(df_select, save_path=FIG_SAVE_PATH)

    # --- 新增：三元图可视化 ---
    visualize_ternary_distribution(df_select, '2020', save_path=FIG_SAVE_PATH)    
    visualize_ternary_distribution(df_select, '2021', save_path=FIG_SAVE_PATH)    

    # --- 保存最终结果 ---
    final_results_path = os.path.join(FIG_SAVE_PATH, 'final_analysis_data.csv')
    df_select.to_csv(final_results_path, index=False)
    print(f"\n[INFO] Final data with cluster labels saved to {final_results_path}")
    
    print("\n[INFO] Analysis complete.")