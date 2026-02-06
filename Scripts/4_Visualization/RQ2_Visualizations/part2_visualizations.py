#!/usr/bin/env python3
"""
Visualization PART 2: Project Age/Maturity - H2.0 and H2.1
===========================================================
12-smell analysis (uses total_smells_filtered from age_analysis_data.csv).
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

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load age analysis data with 12-smell metrics"""
    data_file = Path('age_analysis_data.csv')
    if not data_file.exists():
        raise FileNotFoundError("age_analysis_data.csv not found in current directory")
    
    df = pd.read_csv(data_file)
    
    # Recalculate smell density based on total_smells_filtered (12 smells)
    df['smell_density_12'] = (df['total_smells_filtered'] / df['loc']) * 1000
    
    print(f"Loaded {len(df)} repositories")
    print(f"Age range: {df['age_years'].min():.1f} - {df['age_years'].max():.1f} years")
    print(f"Median age: {df['age_years'].median():.2f} years")
    
    return df

# ============================================================================
# H2.0: AGE CORRELATION
# ============================================================================
def plot_h2_0_correlation(df):
    """
    H2.0: Scatter plot of project age vs smell density.
    Tests if older projects accumulate technical debt.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate correlation
    rho, p_val = stats.spearmanr(df['age_years'], df['smell_density_12'])
    
    # Scatter plot with trend line
    sns.regplot(data=df, x='age_years', y='smell_density_12',
                scatter_kws={'alpha': 0.5, 's': 25, 'color': TUM_COLORS['gray'],
                            'edgecolors': 'none'},
                line_kws={'color': TUM_COLORS['blue2'], 'linewidth': 2.5})
    
    plt.xlabel('Project Age (years)', fontweight='bold')
    plt.ylabel('Smell Density (per 1K LOC)', fontweight='bold')
    plt.title('Project Age vs. Code Smell Density\n' +
              r'$\rho$ = ' + f'{rho:.4f}, p = {p_val:.3f}',
              pad=10, fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fig_h2_age_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig_h2_age_correlation.png")
    plt.close()

# ============================================================================
# H2.1: YOUNG VS MATURE
# ============================================================================
def plot_h2_1_boxplot(df):
    """
    H2.1: Boxplot comparing young vs mature projects.
    Uses median age split (already in age_category column).
    """
    plt.figure(figsize=(8, 6))
    
    median_age = df['age_years'].median()
    order = ['Young', 'Mature']
    colors = [TUM_COLORS['lblue'], TUM_COLORS['blue']]
    
    # Create boxplot
    bp = plt.boxplot(
        [df[df['age_category'] == cat]['smell_density_12'].dropna() for cat in order],
        positions=[0, 1],
        widths=0.4,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=TUM_COLORS['orange'], linewidth=2.5),
        boxprops=dict(edgecolor=TUM_COLORS['dgray'], linewidth=1.2),
        whiskerprops=dict(color=TUM_COLORS['dgray'], linewidth=1.2),
        capprops=dict(color=TUM_COLORS['dgray'], linewidth=1.2)
    )
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add jittered points
    for i, cat in enumerate(order):
        data_subset = df[df['age_category'] == cat]['smell_density_12'].dropna()
        y = data_subset.values
        x = np.random.normal(i, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.25, s=15, color=TUM_COLORS['dgray'],
                   edgecolors='none')
    
    # Add median annotations
    ax = plt.gca()
    for i, cat in enumerate(order):
        median = df[df['age_category'] == cat]['smell_density_12'].median()
        n = len(df[df['age_category'] == cat])
        ax.text(i, ax.get_ylim()[1] * 0.93,
                f'Median: {median:.3f}\n(n={n})',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=TUM_COLORS['dgray'], alpha=0.9, linewidth=0.8))
    
    plt.xticks([0, 1],
               [f'Young\n(≤ {median_age:.1f} years)',
                f'Mature\n(> {median_age:.1f} years)'],
               fontsize=10)
    plt.ylabel('Smell Density (per 1K LOC)', fontweight='bold')
    plt.title('Code Smell Density: Young vs. Mature Projects\n' +
              'Mann-Whitney U = 10541.5, p = 0.228, δ = 0.083',
              pad=15, fontsize=12)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fig_h2_age_boxplot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig_h2_age_boxplot.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print("H2: PROJECT AGE ANALYSIS (12-Smell Analysis)")
    print("TUM Corporate Design Colors | Academic Thesis Style")
    print("="*70)
    
    try:
        # Load data
        print("\n📊 Loading data...")
        df = load_data()
        
        # Generate plots
        print("\n🎨 Generating H2 visualizations...")
        plot_h2_0_correlation(df)
        plot_h2_1_boxplot(df)
        
        print("\n" + "="*70)
        print("✅ H2 plots generated successfully!")
        print("="*70)
        print("\nFiles created:")
        print("  • fig_h2_age_correlation.png (age scatter plot)")
        print("  • fig_h2_age_boxplot.png (young vs mature comparison)")
        print("\n📁 Both files saved at 300 DPI for thesis quality")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()