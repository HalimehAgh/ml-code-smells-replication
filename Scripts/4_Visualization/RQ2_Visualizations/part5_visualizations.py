#!/usr/bin/env python3
"""
Visualization PART 5: CI/CD Adoption - H5.0
============================================
12-smell analysis (uses total_smells_filtered from ci_analysis_data.csv).
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
    """Load CI/CD analysis data with 12-smell metrics"""
    data_file = Path('ci_analysis_data.csv')
    if not data_file.exists():
        raise FileNotFoundError("ci_analysis_data.csv not found in current directory")
    
    df = pd.read_csv(data_file)
    
    # Recalculate smell density based on total_smells_filtered (12 smells)
    df['smell_density_12'] = (df['total_smells_filtered'] / df['loc']) * 1000
    
    ci_count = len(df[df['CI_Status'] == 'Yes'])
    no_ci_count = len(df[df['CI_Status'] == 'No'])
    
    print(f"Loaded {len(df)} repositories")
    print(f"CI/CD Present: {ci_count} ({ci_count/len(df)*100:.1f}%)")
    print(f"No CI/CD: {no_ci_count} ({no_ci_count/len(df)*100:.1f}%)")
    
    return df

# ============================================================================
# H5.0: CI/CD COMPARISON
# ============================================================================
def plot_h5_0_boxplot(df):
    """
    H5.0: Boxplot comparing CI/CD vs Non-CI/CD projects.
    Tests if automated quality gates improve code quality.
    """
    plt.figure(figsize=(8, 6))
    
    order = ['Yes', 'No']
    colors = [TUM_COLORS['blue'], TUM_COLORS['lgray']]
    
    # Create boxplot
    bp = plt.boxplot(
        [df[df['CI_Status'] == cat]['smell_density_12'].dropna() for cat in order],
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
        data_subset = df[df['CI_Status'] == cat]['smell_density_12'].dropna()
        y = data_subset.values
        x = np.random.normal(i, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.25, s=15, color=TUM_COLORS['dgray'],
                   edgecolors='none')
    
    # Add median annotations
    ax = plt.gca()
    for i, cat in enumerate(order):
        median = df[df['CI_Status'] == cat]['smell_density_12'].median()
        n = len(df[df['CI_Status'] == cat])
        ax.text(i, ax.get_ylim()[1] * 0.93,
                f'Median: {median:.3f}\n(n={n})',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=TUM_COLORS['dgray'], alpha=0.9, linewidth=0.8))
    
    plt.xticks([0, 1],
               ['CI/CD Present', 'No CI/CD'],
               fontsize=10)
    plt.ylabel('Smell Density (per 1K LOC)', fontweight='bold')
    plt.title('Code Smell Density: CI/CD vs. Non-CI/CD Projects\n' +
              'Mann-Whitney U = 3745.0, p = 0.239, δ = -0.123',
              pad=15, fontsize=12)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fig_h5_cicd_boxplot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig_h5_cicd_boxplot.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print("H5: CI/CD ADOPTION ANALYSIS (12-Smell Analysis)")
    print("TUM Corporate Design Colors | Academic Thesis Style")
    print("="*70)
    
    try:
        # Load data
        print("\n📊 Loading data...")
        df = load_data()
        
        # Generate plot
        print("\n🎨 Generating H5 visualization...")
        plot_h5_0_boxplot(df)
        
        print("\n" + "="*70)
        print("✅ H5 plot generated successfully!")
        print("="*70)
        print("\nFile created:")
        print("  • fig_h5_cicd_boxplot.png (CI/CD vs Non-CI/CD comparison)")
        print("\n📁 File saved at 300 DPI for thesis quality")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()