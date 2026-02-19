import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False


def parse_statistics_file(file_path: str):
    """
    从统计文件中解析文档数量数据
    
    Args:
        file_path: 统计文件路径
    
    Returns:
        doc_counts: 文档数量列表
    """
    doc_counts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到"按文档数量从高到低排序"的部分
    in_doc_section = False
    
    for line in lines:
        # 检测是否进入文档数量排序部分
        if "按文档数量从高到低排序" in line:
            in_doc_section = True
            continue
        
        # 如果遇到下一个部分，停止解析
        if in_doc_section and "=" * 80 in line and len(doc_counts) > 0:
            break
        
        # 解析数据行
        if in_doc_section:
            # 匹配格式: 排名 实体类型 文档数量 实体数量 文档占比 实体占比
            # 例如: "1      person                                        20,268       84,716        44.52%   7.81%"
            match = re.match(r'^\s*\d+\s+\S+.*?(\d{1,3}(?:,\d{3})*)\s+(\d{1,3}(?:,\d{3})*)', line)
            if match:
                doc_count_str = match.group(1).replace(',', '')
                try:
                    doc_count = int(doc_count_str)
                    doc_counts.append(doc_count)
                except ValueError:
                    continue
    
    return doc_counts


def plot_doc_count_histogram(doc_counts: list, output_path: str = "entity_doc_count_histogram.png"):
    """
    绘制文档数量的直方图
    
    Args:
        doc_counts: 文档数量列表
        output_path: 输出图片路径
    """
    if not doc_counts:
        print("No data to plot!")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制直方图
    # 使用对数刻度，因为数据分布可能很广
    bins = np.logspace(0, np.log10(max(doc_counts)), 50)
    
    ax.hist(doc_counts, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xscale('log')
    ax.set_xlabel('文档数量 (Document Count)', fontsize=12, fontweight='bold')
    ax.set_ylabel('实体类型数量 (Number of Entity Types)', fontsize=12, fontweight='bold')
    ax.set_title('实体类别文档数分布直方图\n(Entity Type Document Count Distribution)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加统计信息
    stats_text = f'总实体类型数: {len(doc_counts):,}\n'
    stats_text += f'平均文档数: {np.mean(doc_counts):.1f}\n'
    stats_text += f'中位数文档数: {np.median(doc_counts):.1f}\n'
    stats_text += f'最大文档数: {max(doc_counts):,}\n'
    stats_text += f'最小文档数: {min(doc_counts):,}'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {output_path}")
    
    # 也显示一个线性刻度的版本（用于查看小值分布）
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # 对于线性版本，只显示文档数较小的部分（比如前1000个最常见的）
    # 或者使用更细的bins
    max_doc_for_linear = min(5000, max(doc_counts))
    filtered_counts = [c for c in doc_counts if c <= max_doc_for_linear]
    
    if filtered_counts:
        bins_linear = np.linspace(0, max_doc_for_linear, 100)
        ax2.hist(filtered_counts, bins=bins_linear, edgecolor='black', alpha=0.7, color='coral')
        ax2.set_xlabel('文档数量 (Document Count)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('实体类型数量 (Number of Entity Types)', fontsize=12, fontweight='bold')
        ax2.set_title(f'实体类别文档数分布直方图 (线性刻度, ≤{max_doc_for_linear})\n(Entity Type Document Count Distribution - Linear Scale)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        stats_text2 = f'显示范围: 1 - {max_doc_for_linear}\n'
        stats_text2 += f'包含类型数: {len(filtered_counts):,}\n'
        stats_text2 += f'平均文档数: {np.mean(filtered_counts):.1f}\n'
        stats_text2 += f'中位数文档数: {np.median(filtered_counts):.1f}'
        
        ax2.text(0.98, 0.98, stats_text2, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        linear_output_path = output_path.replace('.png', '_linear.png')
        plt.savefig(linear_output_path, dpi=300, bbox_inches='tight')
        print(f"Linear scale histogram saved to {linear_output_path}")
    
    plt.close('all')


if __name__ == "__main__":
    # 读取统计文件
    stats_file = "./dataset/entity_type_statistics.txt"
    output_image = "./dataset/entity_doc_count_histogram.png"
    
    print("Parsing statistics file...")
    doc_counts = parse_statistics_file(stats_file)
    
    print(f"Extracted {len(doc_counts)} entity types")
    print(f"Document count range: {min(doc_counts):,} - {max(doc_counts):,}")
    
    # 绘制直方图
    print("Plotting histogram...")
    plot_doc_count_histogram(doc_counts, output_image)
    
    print("Done!")
