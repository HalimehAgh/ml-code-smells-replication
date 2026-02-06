#!/usr/bin/env python3
"""
Hypothesis Testing H11.0: Comparison of CI/CD Effect Sizes (UPDATED with Dynamic Filtering)
===========================================================================================
Updates:
- ML Smells: 12 types only (excluding 4 unwanted)
- Python Smells: Dynamically extracted Top 20 (excluding imports/style/docstrings)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from collections import Counter
import warnings
import json
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

class H11Analyzer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir).resolve()
        
        # Will be populated dynamically
        self.pylint_dirs = {}
        self.top_20_smells = []  # Will be extracted dynamically
        
        # Updated to include ONLY the 12 relevant ML smells
        self.ml_smell_types = [
            'Chain_Indexing', 'columns_and_datatype_not_explicitly_set', 'dataframe_conversion_api_misused',
            'in_place_apis_misused', 'matrix_multiplication_api_misused', 'merge_api_parameter_not_explicitly_set',
            'nan_equivalence_comparison_misused', 'unnecessary_iteration', 'gradients_not_cleared_before_backward_propagation',
            'memory_not_freed', 'pytorch_call_method_misused', 'tensor_array_not_used'
        ]
        
        # Pylint smell exclusion list (same as h0_combined.py)
        self.ignored_categories = {
            'Style/Naming': {
                'invalid-name', 'line-too-long', 'trailing-whitespace', 
                'missing-final-newline', 'trailing-newlines', 'bad-indentation',
                'super-with-arguments', 'useless-object-inheritance',
                'bad-continuation', 'bad-whitespace', 'bad-option-value'
            },
            'Docstrings': {
                'missing-module-docstring', 'missing-class-docstring', 
                'missing-function-docstring', 'empty-docstring'
            },
            'Imports/Environment': {
                'import-error', 'no-name-in-module', 'unable-to-import', 
                'unused-import', 'wildcard-import', 'reimported', 
                'cyclic-import', 'relative-beyond-top-level', 
                'import-outside-toplevel', 'ungrouped-imports',
                'useless-import-alias', 'multiple-imports', 'wrong-import-position', 
                'wrong-import-order', 'unused-wildcard-import'
            }
        }
        self.ignored_lookup = {sym: cat for cat, syms in self.ignored_categories.items() for sym in syms}
        
        self.df = pd.DataFrame()

    def _classify_smell(self, symbol):
        if not symbol: return 'Unknown'
        return self.ignored_lookup.get(symbol, 'Valid')

    def _extract_top_20_smells(self):
        """Extract Top 20 valid functional Python smells from Pylint results"""
        print("\nSTEP 0: Extracting Top 20 Valid Python Smells...")
        valid_smell_counts = Counter()
        
        for size in self.pylint_dirs:
            py_path = self.pylint_dirs[size]
            
            for repo_folder in py_path.iterdir():
                if not repo_folder.is_dir(): continue
                json_files = list(repo_folder.glob("*.json"))
                if not json_files: continue
                
                try:
                    with open(json_files[0], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for issue in data:
                                symbol = issue.get('symbol', 'unknown')
                                category = self._classify_smell(symbol)
                                
                                if category == 'Valid':
                                    valid_smell_counts[symbol] += 1
                except: pass
        
        self.top_20_smells = [s[0] for s in valid_smell_counts.most_common(20)]
        print(f"  [+] Extracted Top 20 Python Smells: {', '.join(self.top_20_smells[:5])}... (+{len(self.top_20_smells)-5} more)")

    def _locate_results_folders(self):
        """Finds where the result folders are hiding."""
        print("STEP 1: Locating result directories...")
        targets = {
            'small': 'small_pylint_results',
            'medium': 'medium_pylint_results',
            'large': 'large_pylint_results'
        }
        
        # Search in current dir and up 2 levels
        search_paths = [self.base_dir, self.base_dir.parent, self.base_dir.parent.parent]
        
        for size, target_name in targets.items():
            found = None
            for root in search_paths:
                candidate = root / target_name
                if candidate.exists() and candidate.is_dir():
                    found = candidate
                    break
            
            if found:
                self.pylint_dirs[size] = found
                print(f"  [+] Found {size} results at: {found}")
            else:
                # Fallback: Recursive search (expensive but necessary if lost)
                for root, dirs, files in os.walk(self.base_dir):
                    if target_name in dirs:
                        found = Path(root) / target_name
                        self.pylint_dirs[size] = found
                        print(f"  [+] Found {size} results (deep search) at: {found}")
                        break
            
            if not found:
                print(f"  [!] CRITICAL: Could not find folder '{target_name}'")

    def _find_file(self, filename):
        candidates = [self.base_dir / filename, self.base_dir.parent / filename, Path(filename)]
        for c in candidates:
            if c.exists() and c.is_file(): return c.resolve()
        return None

    def load_hybrid_data(self):
        self._locate_results_folders()
        self._extract_top_20_smells()  # NEW: Extract Top 20 dynamically
        
        print("\nSTEP 2: Loading ML Density & CI Status...")
        ml_file = self._find_file('combined_analysis_data.csv')
        
        ml_lookup = {}
        if ml_file:
             try:
                ml_df = pd.read_csv(ml_file)
                ml_df['repo_name'] = ml_df['repo_name'].str.strip()
                
                # Recalculate density with only the 12 types
                cols_present = [c for c in self.ml_smell_types if c in ml_df.columns]
                ml_df['ml_count_filtered'] = ml_df[cols_present].sum(axis=1)
                ml_df = ml_df[ml_df['loc'] > 0]
                ml_df['ml_density_filtered'] = ml_df['ml_count_filtered'] / (ml_df['loc'] / 1000)
                
                ml_lookup = dict(zip(ml_df['repo_name'], ml_df['ml_density_filtered']))
                print(f"  [+] Recalculated ML densities for {len(ml_lookup)} repos using 12 types.")
             except Exception as e:
                 print(f"  [!] Error processing combined_analysis_data.csv: {e}")
                 age_file = self._find_file('age_analysis_data.csv')
                 if age_file:
                    ml_df = pd.read_csv(age_file)
                    ml_df['GitHub Repo'] = ml_df['GitHub Repo'].str.strip()
                    ml_lookup = dict(zip(ml_df['GitHub Repo'], ml_df['smell_density']))
                    print(f"  [!] Fallback: Loaded ML densities from age_analysis_data.csv for {len(ml_lookup)} repos")
        else:
             print("[!] combined_analysis_data.csv not found! Trying fallback...")
             age_file = self._find_file('age_analysis_data.csv')
             if age_file:
                ml_df = pd.read_csv(age_file)
                ml_df['GitHub Repo'] = ml_df['GitHub Repo'].str.strip()
                ml_lookup = dict(zip(ml_df['GitHub Repo'], ml_df['smell_density']))
                print(f"  [!] Fallback: Loaded ML densities from age_analysis_data.csv for {len(ml_lookup)} repos")
             else:
                 print("[!] No ML data source found.")
                 return

        # Load CI Status
        repo_data = []
        files = {'small': 'repos_small_checked.csv', 'medium': 'repos_medium_checked.csv', 'large': 'repos_large_checked.csv'}

        for size, csv_file in files.items():
            csv_path = self._find_file(csv_file) or self._find_file(f'repos_{size}.csv')
            if not csv_path: 
                print(f"  [!] {csv_file} not found, skipping {size}")
                continue
            
            try:
                temp_df = pd.read_csv(csv_path)
                
                ci_col = 'CI_Exists_Checked'
                if ci_col not in temp_df.columns:
                    print(f"  [!] Expected column 'CI_Exists_Checked' not found in {size}")
                    continue

                print(f"  [+] Using CI column: '{ci_col}' in {size}")
                
                temp_df['Lines of Code'] = pd.to_numeric(temp_df['Lines of Code'], errors='coerce')
                temp_df = temp_df.dropna(subset=['Lines of Code'])
                temp_df = temp_df[temp_df['Lines of Code'] > 0]
                
                for _, row in temp_df.iterrows():
                    repo_name = row['GitHub Repo'].strip()
                    raw_ci = str(row[ci_col]).strip()
                    has_ci = raw_ci.upper() == 'YES'
                    
                    repo_data.append({
                        'Repo': repo_name,
                        'Size': size,
                        'LOC': row['Lines of Code'],
                        'Has_CI': has_ci
                    })
                
            except Exception as e:
                print(f"  [!] Error processing {size}: {e}")

        total_ci = sum(1 for r in repo_data if r['Has_CI'])
        total_no_ci = len(repo_data) - total_ci
        print(f"\nOverall CI Status: {total_ci} with CI, {total_no_ci} without CI")

        print("\nSTEP 3: Scanning Pylint Results...")
        final_records = []
        
        for record in repo_data:
            repo_name = record['Repo']
            if repo_name not in ml_lookup: continue
            
            ml_density = ml_lookup[repo_name]
            size = record['Size']
            
            if size not in self.pylint_dirs: continue
            py_path = self.pylint_dirs[size]
            
            patterns = [repo_name.replace('/', '_'), repo_name.split('/')[-1]]
            
            py_count = 0
            found_py = False
            
            for p in patterns:
                matches = list(py_path.glob(f"*{p}*"))
                if matches:
                    best_match = min(matches, key=lambda x: len(str(x)))
                    jsons = list(best_match.glob("*.json"))
                    if jsons:
                        try:
                            with open(jsons[0], 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                py_count = sum(1 for x in data if x.get('symbol') in self.top_20_smells)
                            found_py = True
                        except: pass
                    break
            
            if found_py:
                final_records.append({
                    'Repo': repo_name,
                    'Has_CI': record['Has_CI'],
                    'ML_Density': ml_density,
                    'Python_Density': (py_count / record['LOC']) * 1000
                })

        self.df = pd.DataFrame(final_records)
        
        ci_final = sum(self.df['Has_CI'])
        no_ci_final = len(self.df) - ci_final
        print(f"[+] Final Paired Dataset: {len(self.df)} repositories.")
        print(f"[+] Final CI Distribution: {ci_final} with CI, {no_ci_final} without CI")

    def cliffs_delta(self, x, y):
        x, y = np.array(x), np.array(y)
        if len(x) == 0 or len(y) == 0: return 0
        return np.mean(np.sign(np.subtract.outer(x, y)))

    def run_h11_test(self):
        if self.df.empty: 
            print("[!] No data available for testing.")
            return

        print("\n" + "="*60)
        print("H11.0: COMPARISON OF CI/CD EFFECT SIZES")
        print("="*60)
        
        ci_group = self.df[self.df['Has_CI']]
        no_ci_group = self.df[~self.df['Has_CI']]
        
        print(f"Sample Sizes: CI={len(ci_group)}, No-CI={len(no_ci_group)}")
        
        if len(ci_group) < 5 or len(no_ci_group) < 5:
            print("[!] Insufficient data in one or both groups for meaningful comparison.")
            return
        
        d_ml_obs = self.cliffs_delta(ci_group['ML_Density'], no_ci_group['ML_Density'])
        d_py_obs = self.cliffs_delta(ci_group['Python_Density'], no_ci_group['Python_Density'])
        diff_obs = abs(d_ml_obs - d_py_obs)
        
        print(f"\nObserved Effect Sizes (Cliff's Delta: CI vs No-CI):")
        print(f"  δ_ML (ML-Specific):     {d_ml_obs:.4f}")
        print(f"  δ_Python (General Py):  {d_py_obs:.4f}")
        print(f"  Difference |Δδ|:        {diff_obs:.4f}")
        
        print("\nRunning Bootstrap Test (1000 iterations)...")
        n_boot = 1000
        raw_diffs = []
        pool = self.df.copy()
        
        for i in range(n_boot):
            resampled = pool.sample(n=len(pool), replace=True)
            r_ci = resampled[resampled['Has_CI']]
            r_no_ci = resampled[~resampled['Has_CI']]
            
            if len(r_ci) == 0 or len(r_no_ci) == 0: continue
            
            d_ml = self.cliffs_delta(r_ci['ML_Density'], r_no_ci['ML_Density'])
            d_py = self.cliffs_delta(r_ci['Python_Density'], r_no_ci['Python_Density'])
            raw_diffs.append(d_ml - d_py)
        
        if len(raw_diffs) < 100:
            print(f"[!] Warning: Only {len(raw_diffs)} valid bootstrap iterations out of {n_boot}")
            
        if len(raw_diffs) == 0:
            print("[!] Bootstrap failed - no valid iterations.")
            return
            
        ci_lower = np.percentile(raw_diffs, 2.5)
        ci_upper = np.percentile(raw_diffs, 97.5)
        is_sig = (ci_lower > 0) or (ci_upper < 0)
        
        print(f"\nBootstrap Results:")
        print(f"  Mean Difference: {np.mean(raw_diffs):.4f}")
        print(f"  95% CI:          [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Valid Iterations: {len(raw_diffs)}/{n_boot}")
        
        if is_sig:
            print("\n  Decision: REJECT H11.0")
            print("  Conclusion: CI/CD affects ML-specific and general Python quality differently.")
        else:
            print("\n  Decision: FAIL TO REJECT H11.0")
            print("  Conclusion: CI/CD has similar effects on both quality types.")
            
        self._plot()

    def _plot(self):
        if len(self.df[self.df['Has_CI']]) == 0 or len(self.df[~self.df['Has_CI']]) == 0:
            print("[!] Cannot create plots - missing CI groups")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.boxplot(data=self.df, x='Has_CI', y='ML_Density', ax=axes[0], 
                   palette=['#e74c3c', '#2ecc71'], showfliers=False)
        axes[0].set_title('Impact of CI on ML-Specific Density')
        axes[0].set_ylabel('ML Smell Density')
        axes[0].set_xticklabels(['No CI', 'Has CI'])
        
        sns.boxplot(data=self.df, x='Has_CI', y='Python_Density', ax=axes[1], 
                   palette=['#e74c3c', '#2ecc71'], showfliers=False)
        axes[1].set_title('Impact of CI on General Python Density')
        axes[1].set_ylabel('Python Smell Density')
        axes[1].set_xticklabels(['No CI', 'Has CI'])
        
        plt.tight_layout()
        plt.savefig('H11_0_CI_Effect_Comparison.png', dpi=300)
        print("\n[+] Saved plot: H11_0_CI_Effect_Comparison.png")

if __name__ == "__main__":
    print("="*70)
    print("H11.0: CI/CD EFFECT SIZE COMPARISON (UPDATED)")
    print("="*70)
    print("Using corrected smell filtering:")
    print("  - ML: 12 types (excluding 4 unwanted)")
    print("  - Python: Top 20 valid functional (dynamically extracted)")
    print("="*70)
    
    an = H11Analyzer()
    an.load_hybrid_data()
    an.run_h11_test()