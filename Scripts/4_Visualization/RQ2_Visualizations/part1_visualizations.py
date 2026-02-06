#!/usr/bin/env python3
"""
Visualization PART 1: Project Size (LOC) - H1.0 and H1.1
=========================================================
12-smell analysis (excludes 4 ignored smells).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# ============================================================================
# TUM CORPORATE DESIGN COLORS
# ============================================================================
TUM_COLORS = {
    'blue': '#0065BD',
    'blue2': '#005293',
    'blue3': '#003359',
    'orange': '#E37222',
    'green': '#A2AD00',
    'lblue': '#98C6EA',
    'ablue': '#64A0C8',
    'gray': '#808080',
    'lgray': '#CCCCC6',
    'dgray': '#333333'
}

# ============================================================================
# ACADEMIC THESIS STYLING
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

SIZE_ORDER = ['small', 'medium', 'large']
SIZE_COLORS = [TUM_COLORS['lblue'], TUM_COLORS['ablue'], TUM_COLORS['blue']]

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load and prepare data with 12-smell recalculation"""
    data_file = Path('analysis_data.csv')
    if not data_file.exists():
        raise FileNotFoundError("analysis_data.csv not found in current directory")
    
    df = pd.read_csv(data_file)
    
    # The 12 smells we keep (excluding 4 ignored)
    kept_smells = [
        'gradients_not_cleared_before_backward_propagation',
        'columns_and_datatype_not_explicitly_set',
        'in_place_apis_misused',
        'Chain_Indexing',
        'matrix_multiplication_api_misused',
        'pytorch_call_method_misused',
        'dataframe_conversion_api_misused',
        'nan_equivalence_comparison_misused',
        'unnecessary_iteration',
        'merge_api_parameter_not_explicitly_set',
        'memory_not_freed',
        'tensor_array_not_used'
    ]
    
    # Recalculate for 12 smells only
    df['total_smells_12'] = df[kept_smells].sum(axis=1)
    df['smell_density_12'] = (df['total_smells_12'] / df['loc']) * 1000
    
    print(f"Loaded {len(df)} repositories")
    print(f"Total 12-smell instances: {int(df['total_smells_12'].sum())}")
    
    return df

# ============================================================================
# H1.0: CORRELATION ANALYSIS
# ============================================================================
def plot_h1_0_correlations(data):
    """
    H1.0: Two-panel scatter plots with trend lines.
    Left: LOC vs Absolute Count (validates normalization)
    Right: LOC vs Smell Density (tests hypothesis)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate correlations
    rho_abs, p_abs = stats.spearmanr(data['loc'], data['total_smells_12'])
    rho_den, p_den = stats.spearmanr(data['loc'], data['smell_density_12'])
    
    # ---- Panel A: LOC vs Absolute Smell Count ----
    sns.regplot(data=data, x='loc', y='total_smells_12', ax=ax1,
                scatter_kws={'alpha': 0.5, 's': 20, 'color': TUM_COLORS['gray'], 
                            'edgecolors': 'none'},
                line_kws={'color': TUM_COLORS['orange'], 'linewidth': 2.5})
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Lines of Code (LOC)', fontweight='bold')
    ax1.set_ylabel('Absolute Smell Count', fontweight='bold')
    ax1.set_title('(A) LOC vs. Absolute Smells\n' + 
                  r'$\rho$ = ' + f'{rho_abs:.4f}, p = {p_abs:.2e}',
                  pad=10)
    ax1.grid(True, alpha=0.3, which='both', linestyle='--')
    
    # ---- Panel B: LOC vs Smell Density ----
    sns.regplot(data=data, x='loc', y='smell_density_12', ax=ax2,
                scatter_kws={'alpha': 0.5, 's': 20, 'color': TUM_COLORS['gray'],
                            'edgecolors': 'none'},
                line_kws={'color': TUM_COLORS['blue'], 'linewidth': 2.5})
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Lines of Code (LOC)', fontweight='bold')
    ax2.set_ylabel('Smell Density (per 1K LOC)', fontweight='bold')
    ax2.set_title('(B) LOC vs. Smell Density\n' +
                  r'$\rho$ = ' + f'{rho_den:.4f}, p = {p_den:.3f}',
                  pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fig_h1_correlations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig_h1_correlations.png")
    plt.close()

# ============================================================================
# H1.1: GROUP COMPARISON
# ============================================================================
def plot_h1_1_boxplot(data):
    """
    H1.1: Boxplot comparing small/medium/large projects.
    Shows distribution with jittered points and median annotations.
    """
    plt.figure(figsize=(10, 6))
    
    # Create boxplot
    bp = plt.boxplot(
        [data[data['size_category'] == cat]['smell_density_12'].dropna() 
         for cat in SIZE_ORDER],
        positions=[0, 1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=TUM_COLORS['orange'], linewidth=2.5),
        boxprops=dict(edgecolor=TUM_COLORS['dgray'], linewidth=1.2),
        whiskerprops=dict(color=TUM_COLORS['dgray'], linewidth=1.2),
        capprops=dict(color=TUM_COLORS['dgray'], linewidth=1.2)
    )
    
    # Color boxes with TUM gradient
    for patch, color in zip(bp['boxes'], SIZE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add jittered data points
    for i, cat in enumerate(SIZE_ORDER):
        data_subset = data[data['size_category'] == cat]['smell_density_12'].dropna()
        y = data_subset.values
        x = np.random.normal(i, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.25, s=15, color=TUM_COLORS['dgray'], 
                   edgecolors='none')
    
    # Add median annotations
    ax = plt.gca()
    for i, cat in enumerate(SIZE_ORDER):
        median = data[data['size_category'] == cat]['smell_density_12'].median()
        n = len(data[data['size_category'] == cat])
        ax.text(i, ax.get_ylim()[1] * 0.93,
                f'Median: {median:.3f}\n(n={n})',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=TUM_COLORS['dgray'], alpha=0.9, linewidth=0.8))
    
    plt.xticks([0, 1, 2], 
               ['Small\n(< 30th percentile)', 
                'Medium\n(30th–60th percentile)', 
                'Large\n(> 60th percentile)'],
               fontsize=10)
    plt.ylabel('Smell Density (per 1K LOC)', fontweight='bold')
    plt.title('Code Smell Density by Project Size Category\n' +
              'Kruskal-Wallis H = 3.574, p = 0.167',
              pad=15, fontsize=12)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fig_h1_boxplot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig_h1_boxplot.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print("H1: PROJECT SIZE ANALYSIS (12-Smell Analysis)")
    print("TUM Corporate Design Colors | Academic Thesis Style")
    print("="*70)
    
    try:
        # Load data
        print("\n📊 Loading data...")
        data = load_data()
        
        # Generate plots
        print("\n🎨 Generating H1 visualizations...")
        plot_h1_0_correlations(data)
        plot_h1_1_boxplot(data)
        
        print("\n" + "="*70)
        print("✅ H1 plots generated successfully!")
        print("="*70)
        print("\nFiles created:")
        print("  • fig_h1_correlations.png (2-panel correlation)")
        print("  • fig_h1_boxplot.png (size category comparison)")
        print("\n📁 Both files saved at 300 DPI for thesis quality")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()