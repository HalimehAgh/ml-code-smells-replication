#!/usr/bin/env python3
"""
Visualization PART 6: Domain Analysis - H6.0 and H6.1
======================================================
12-smell analysis (recalculates from domain_analysis_data.csv).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
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
    """Load domain analysis data with 12-smell recalculation"""
    data_file = Path('domain_analysis_data.csv')
    if not data_file.exists():
        raise FileNotFoundError("domain_analysis_data.csv not found in current directory")
    
    df = pd.read_csv(data_file)
    
    # The 12 smells we keep
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
    
    # Recalculate for 12 smells
    df['total_smells_12'] = df[kept_smells].sum(axis=1)
    df['smell_density_12'] = (df['total_smells_12'] / df['loc']) * 1000
    
    domain_counts = df['domain'].value_counts()
    print(f"Loaded {len(df)} repositories")
    print(f"\nDomain distribution:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count}")
    
    return df, kept_smells

# ============================================================================
# H6.0: DOMAIN COMPARISON
# ============================================================================
def plot_h6_0_boxplot(df):
    """
    H6.0: Boxplot comparing smell density across ML domains.
    Tests if different domains have characteristic quality patterns.
    """
    plt.figure(figsize=(12, 6))
    
    # Sort domains by median smell density
    domain_order = df.groupby('domain')['smell_density_12'].median().sort_values().index.tolist()
    
    # TUM color palette for domains
    domain_colors = {
        'ML Frameworks / Libraries': TUM_COLORS['blue'],
        'Natural Language Processing': TUM_COLORS['ablue'],
        'Computer Vision': TUM_COLORS['lblue'],
        'Domain-Specific Applications': TUM_COLORS['green'],
        'Data Science / Traditional ML': TUM_COLORS['orange'],
        'MLOps / Deployment': TUM_COLORS['blue2'],
        'Reinforcement Learning': TUM_COLORS['blue3']
    }
    
    colors = [domain_colors.get(d, TUM_COLORS['gray']) for d in domain_order]
    
    # Create boxplot
    bp = plt.boxplot(
        [df[df['domain'] == domain]['smell_density_12'].dropna() for domain in domain_order],
        positions=range(len(domain_order)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=TUM_COLORS['orange'], linewidth=2),
        boxprops=dict(edgecolor=TUM_COLORS['dgray'], linewidth=1.2),
        whiskerprops=dict(color=TUM_COLORS['dgray'], linewidth=1.2),
        capprops=dict(color=TUM_COLORS['dgray'], linewidth=1.2)
    )
    
    # Color boxes by domain
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add jittered points for each domain
    for i, domain in enumerate(domain_order):
        data_subset = df[df['domain'] == domain]['smell_density_12'].dropna()
        y = data_subset.values
        x = np.random.normal(i, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.2, s=12, color=TUM_COLORS['dgray'],
                   edgecolors='none')
    
    # Set x-axis labels (wrapped for long domain names)
    ax = plt.gca()
    ax.set_xticks(range(len(domain_order)))
    wrapped_labels = [d.replace(' / ', '/\n') for d in domain_order]
    ax.set_xticklabels(wrapped_labels, rotation=20, ha='right', fontsize=9)
    
    ax.set_ylabel('Smell Density (per 1K LOC)', fontweight='bold')
    ax.set_title('Code Smell Density Across ML Domains\n' +
                 'Kruskal-Wallis H = 21.184, p = 0.0017',
                 pad=15, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fig_h6_0_boxplot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig_h6_0_boxplot.png")
    plt.close()

# ============================================================================
# H6.1: DOMAIN × SMELL TYPE HEATMAP
# ============================================================================
def plot_h6_1_heatmap(df, smell_cols):
    """
    H6.1: Heatmap showing domain-smell type associations.
    Reveals characteristic smell patterns for each domain.
    """
    
    # Create contingency table (domains × smell types)
    domain_smell_counts = df.groupby('domain')[smell_cols].sum()
    
    # Normalize by row (to show proportions within each domain)
    domain_smell_props = domain_smell_counts.div(domain_smell_counts.sum(axis=1), axis=0)
    
    # VERIFICATION: Check that rows sum to 1.0 (100%)
    row_sums = domain_smell_props.sum(axis=1)
    print("\n🔍 Verification - Row sums (should all be ~1.0 = 100%):")
    for domain, row_sum in row_sums.items():
        print(f"  {domain}: {row_sum:.4f}")
    
    # Show top 3 smells per domain
    print("\n📊 Top 3 smells per domain:")
    for domain in domain_smell_props.index:
        top_3 = domain_smell_props.loc[domain].nlargest(3)
        print(f"\n  {domain}:")
        for smell, pct in top_3.items():
            # Use the shortened name
            print(f"    {smell}: {pct*100:.1f}%")
    
    # Shorten smell names for display
    display_names = {
        'gradients_not_cleared_before_backward_propagation': 'Gradients\nNot Cleared',
        'columns_and_datatype_not_explicitly_set': 'Column/Datatype\nNot Explicit',
        'in_place_apis_misused': 'In-Place\nAPIs Misused',
        'Chain_Indexing': 'Chain\nIndexing',
        'matrix_multiplication_api_misused': 'Matrix Mult.\nMisused',
        'pytorch_call_method_misused': 'PyTorch Call\nMisused',
        'dataframe_conversion_api_misused': 'DataFrame Conv.\nMisused',
        'nan_equivalence_comparison_misused': 'NaN Comparison\nMisused',
        'unnecessary_iteration': 'Unnecessary\nIteration',
        'merge_api_parameter_not_explicitly_set': 'Merge Param\nNot Set',
        'memory_not_freed': 'Memory\nNot Freed',
        'tensor_array_not_used': 'Tensor Array\nNot Used'
    }
    
    domain_smell_props.columns = [display_names.get(col, col) for col in domain_smell_props.columns]
    
    # Wrap domain names
    domain_smell_props.index = [idx.replace(' / ', '/\n') for idx in domain_smell_props.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create TUM blue colormap
    tum_cmap = LinearSegmentedColormap.from_list(
        'TUM_Blues', 
        ['#FFFFFF', TUM_COLORS['lblue'], TUM_COLORS['ablue'], TUM_COLORS['blue']]
    )
    
    # Convert proportions to percentages for display
    domain_smell_percentages = domain_smell_props * 100
    
    # Create heatmap with percentage annotations AND percentage colorbar
    sns.heatmap(domain_smell_percentages,  # ← Use percentages for coloring too
                cmap=tum_cmap,
                annot=True,  # Show values
                fmt='.1f',  # Format: 15.2
                cbar_kws={'label': 'Percentage within Domain (%)'},  # ← Better label
                linewidths=0.5,
                linecolor='white',
                annot_kws={'fontsize': 8, 'color': 'black'},
                ax=ax)
    
    ax.set_xlabel('ML-Specific Code Smell Type', fontweight='bold', fontsize=11)
    ax.set_ylabel('ML Domain', fontweight='bold', fontsize=11)
    ax.set_title('Domain × Smell Type Association Heatmap\n' +
                 "Chi-square = 3802.34, p < 0.001, Cramér's V = 0.200",
                 pad=20, fontsize=12)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('fig_h6_1_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig_h6_1_heatmap.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print("H6: DOMAIN ANALYSIS (12-Smell Analysis)")
    print("TUM Corporate Design Colors | Academic Thesis Style")
    print("="*70)
    
    try:
        # Load data
        print("\n📊 Loading data...")
        df, smell_cols = load_data()
        
        # Generate plots
        print("\n🎨 Generating H6 visualizations...")
        plot_h6_0_boxplot(df)
        plot_h6_1_heatmap(df, smell_cols)
        
        print("\n" + "="*70)
        print("✅ H6 plots generated successfully!")
        print("="*70)
        print("\nFiles created:")
        print("  • fig_h6_0_boxplot.png (domain comparison)")
        print("  • fig_h6_1_heatmap.png (domain × smell type heatmap)")
        print("\n📁 Both files saved at 300 DPI for thesis quality")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()