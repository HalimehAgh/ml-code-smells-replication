#!/usr/bin/env python3
"""
Visualization PART 3: Contributors/Team Size - H3.0 and H3.1
=============================================================
12-smell analysis (uses total_smells_filtered from contributors_analysis_data.csv).
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
    """Load contributors analysis data with 12-smell metrics"""
    data_file = Path('contributors_analysis_data.csv')
    if not data_file.exists():
        raise FileNotFoundError("contributors_analysis_data.csv not found in current directory")
    
    df = pd.read_csv(data_file)
    
    # Recalculate smell density based on total_smells_filtered (12 smells)
    df['smell_density_12'] = (df['total_smells_filtered'] / df['loc']) * 1000
    
    median_contrib = df['Contributor Count'].median()
    
    print(f"Loaded {len(df)} repositories")
    print(f"Contributors range: {df['Contributor Count'].min()} - {df['Contributor Count'].max()}")
    print(f"Median contributors: {median_contrib:.0f}")
    
    return df

# ============================================================================
# H3.0: CONTRIBUTORS CORRELATION
# ============================================================================
def plot_h3_0_correlation(df):
    """
    H3.0: Scatter plot of contributors vs smell density (log scale).
    Tests if larger teams (Brooks' Law) lead to more smells.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate correlation
    rho, p_val = stats.spearmanr(df['Contributor Count'], df['smell_density_12'])
    
    # Scatter plot with trend line (log x-axis)
    sns.regplot(data=df, x='Contributor Count', y='smell_density_12',
                scatter_kws={'alpha': 0.5, 's': 25, 'color': TUM_COLORS['gray'],
                            'edgecolors': 'none'},
                line_kws={'color': TUM_COLORS['green'], 'linewidth': 2.5})
    
    plt.xscale('log')
    plt.xlabel('Number of Contributors (log scale)', fontweight='bold')
    plt.ylabel('Smell Density (per 1K LOC)', fontweight='bold')
    plt.title('Team Size vs. Code Smell Density\n' +
              r'$\rho$ = ' + f'{rho:.4f}, p = {p_val:.4f}',
              pad=10, fontsize=12)
    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fig_h3_team_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig_h3_team_correlation.png")
    plt.close()

# ============================================================================
# H3.1: SMALL VS LARGE TEAMS
# ============================================================================
def plot_h3_1_boxplot(df):
    """
    H3.1: Boxplot comparing small vs large teams.
    Uses median contributors split (already in team_category column).
    """
    plt.figure(figsize=(8, 6))
    
    median_contrib = df['Contributor Count'].median()
    order = ['Small', 'Large']
    colors = [TUM_COLORS['lblue'], TUM_COLORS['green']]
    
    # Create boxplot
    bp = plt.boxplot(
        [df[df['team_category'] == cat]['smell_density_12'].dropna() for cat in order],
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
        data_subset = df[df['team_category'] == cat]['smell_density_12'].dropna()
        y = data_subset.values
        x = np.random.normal(i, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.25, s=15, color=TUM_COLORS['dgray'],
                   edgecolors='none')
    
    # Add median annotations
    ax = plt.gca()
    for i, cat in enumerate(order):
        median = df[df['team_category'] == cat]['smell_density_12'].median()
        n = len(df[df['team_category'] == cat])
        ax.text(i, ax.get_ylim()[1] * 0.93,
                f'Median: {median:.3f}\n(n={n})',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=TUM_COLORS['dgray'], alpha=0.9, linewidth=0.8))
    
    plt.xticks([0, 1],
               [f'Small Teams\n(≤ {int(median_contrib)} contributors)',
                f'Large Teams\n(> {int(median_contrib)} contributors)'],
               fontsize=10)
    plt.ylabel('Smell Density (per 1K LOC)', fontweight='bold')
    plt.title('Code Smell Density: Small vs. Large Teams\n' +
              'Mann-Whitney U = 10690.5, p = 0.153, δ = 0.099',
              pad=15, fontsize=12)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fig_h3_team_boxplot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig_h3_team_boxplot.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print("H3: TEAM SIZE ANALYSIS (12-Smell Analysis)")
    print("TUM Corporate Design Colors | Academic Thesis Style")
    print("="*70)
    
    try:
        # Load data
        print("\n📊 Loading data...")
        df = load_data()
        
        # Generate plots
        print("\n🎨 Generating H3 visualizations...")
        plot_h3_0_correlation(df)
        plot_h3_1_boxplot(df)
        
        print("\n" + "="*70)
        print("✅ H3 plots generated successfully!")
        print("="*70)
        print("\nFiles created:")
        print("  • fig_h3_team_correlation.png (contributors scatter plot)")
        print("  • fig_h3_team_boxplot.png (small vs large teams comparison)")
        print("\n📁 Both files saved at 300 DPI for thesis quality")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()