#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Exclude: hyperparameters_not_explicitly_set, empty_column_misinitialization, 
# Broadcasting_Feature_Not_Used, deterministic_algorithm_option_not_used
ML_SMELL_TYPES = [
    'Chain_Indexing', 
    'columns_and_datatype_not_explicitly_set', 
    'dataframe_conversion_api_misused', 
    'in_place_apis_misused', 
    'matrix_multiplication_api_misused', 
    'merge_api_parameter_not_explicitly_set',
    'nan_equivalence_comparison_misused', 
    'unnecessary_iteration', 
    'gradients_not_cleared_before_backward_propagation', 
    'memory_not_freed', 
    'pytorch_call_method_misused', 
    'tensor_array_not_used'
]

def fisher_z_comparison(r1, r2, n):
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    se_diff = np.sqrt(1 / (n - 3) + 1 / (n - 3))
    z = (z1 - z2) / se_diff
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

def main():
    try:
        ml_data = pd.read_csv('combined_analysis_data.csv')
    except FileNotFoundError:
        return

    cols_present = [c for c in ML_SMELL_TYPES if c in ml_data.columns]
    ml_data['ml_count_filtered'] = ml_data[cols_present].sum(axis=1)
    
    ml_data = ml_data[ml_data['loc'] > 0].copy()
    ml_data['ml_density_filtered'] = ml_data['ml_count_filtered'] / (ml_data['loc'] / 1000)
    ml_data['repo_name'] = ml_data['repo_name'].str.strip()

    try:
        py_data = pd.read_csv('h0_paired_data.csv')
    except FileNotFoundError:
        return
        
    if 'Repo' in py_data.columns:
        py_data = py_data.rename(columns={'Repo': 'repo_name'})
    
    if 'LOC' not in py_data.columns and 'Lines of Code' in py_data.columns:
        py_data = py_data.rename(columns={'Lines of Code': 'LOC'})
        
    py_data['repo_name'] = py_data['repo_name'].str.strip()

    merged = pd.merge(ml_data[['repo_name', 'loc', 'ml_density_filtered']], 
                      py_data[['repo_name', 'Python_Density']], 
                      on='repo_name', how='inner')
    
    rho_ml, p_ml = stats.spearmanr(merged['loc'], merged['ml_density_filtered'])
    rho_py, p_py = stats.spearmanr(merged['loc'], merged['Python_Density'])
    
    print(f"{'Relationship':<30} | {'Spearman rho':<15} | {'p-value'}")
    print("-" * 65)
    print(f"{'LOC vs ML-Specific':<30} | {rho_ml:<15.4f} | {p_ml:.4e}")
    print(f"{'LOC vs General Python':<30} | {rho_py:<15.4f} | {p_py:.4e}")
    
    z_stat, p_z = fisher_z_comparison(rho_ml, rho_py, len(merged))
    delta_rho = abs(rho_ml - rho_py)
    
    print("\nFisher's z-transformation Results:")
    print(f"  Difference (Δρ): {delta_rho:.4f}")
    print(f"  z-statistic:     {z_stat:.4f}")
    print(f"  p-value:         {p_z:.4e}")
    
    if delta_rho < 0.1: es = "Negligible"
    elif delta_rho < 0.3: es = "Small"
    elif delta_rho < 0.5: es = "Moderate"
    else: es = "Large"
    print(f"  Effect Size:     {es}")
    
    if p_z < 0.05:
        decision = "REJECT H7.0"
    else:
        decision = "FAIL TO REJECT H7.0"
        
    print(f"  Decision:        {decision}")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.regplot(data=merged, x='loc', y='ml_density_filtered', 
                scatter_kws={'alpha':0.5}, line_kws={'color':'#3498db'})
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Lines of Code (Log)')
    plt.ylabel('ML Smell Density (Log)')
    plt.title(f'LOC vs ML-Specific Density\nrho={rho_ml:.3f}')
    
    plt.subplot(1, 2, 2)
    sns.regplot(data=merged, x='loc', y='Python_Density', 
                scatter_kws={'alpha':0.5}, line_kws={'color':'#9b59b6'})
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Lines of Code (Log)')
    plt.ylabel('Python Smell Density (Log)')
    plt.title(f'LOC vs General Python Density\nrho={rho_py:.3f}')
    
    plt.tight_layout()
    plt.savefig('H7_0_Correlation_Comparison.png')

    res_df = pd.DataFrame([{
        'Metric_1': 'ML_Density', 'Rho_1': rho_ml, 'P_1': p_ml,
        'Metric_2': 'Python_Density', 'Rho_2': rho_py, 'P_2': p_py,
        'Delta_Rho': delta_rho, 'Z_Stat': z_stat, 'P_Value_Diff': p_z,
        'Decision': decision
    }])
    res_df.to_csv('h7_results.csv', index=False)

if __name__ == "__main__":
    main()