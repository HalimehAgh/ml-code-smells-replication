#!/usr/bin/env python3
"""
Hypothesis Testing H0.0: ML-Specific vs. General Python Code Smells
====================================================================================
Methodology:
1. PYLINT: Filters out imports/style/docstrings, extracts Top 20 valid functional smells
2. ML SMELLS: Uses ONLY the 12 specified CodeSmile smell types (excludes 4 ml smells)
3. TEST: Paired Wilcoxon Signed-Rank Test comparing densities
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

# --- ML SMELL CONFIGURATION (12 types only) ---
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

class SmellComparator:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir).resolve()
        
        self.dirs = {
            'small': {'pylint': 'small_pylint_results', 'codesmile': 'small_codesmile_results'},
            'medium': {'pylint': 'medium_pylint_results', 'codesmile': 'medium_codesmile_results'},
            'large': {'pylint': 'large_pylint_results', 'codesmile': 'large_codesmile_results'}
        }
        
        self.top_20_smells = []
        self.df = pd.DataFrame()
        
        # Stats containers
        self.ignored_stats = Counter()
        self.valid_smell_counts = Counter()
        self.repo_smell_data = [] 
        
        # EXCLUDE
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

    def _find_dir(self, dirname):
        candidates = [self.base_dir / dirname, self.base_dir.parent / dirname, self.base_dir / ".." / dirname]
        for c in candidates:
            if c.exists() and c.is_dir(): return c.resolve()
        return None

    def _classify_smell(self, symbol):
        if not symbol: return 'Unknown'
        return self.ignored_lookup.get(symbol, 'Valid')

    def analyze_pylint_smells(self):
        """Scan Pylint results and extract Top 20 valid functional smells"""
        print("STEP 1: Scanning Pylint Results & Generating Statistics...")
        total_files = 0
        
        for size, paths in self.dirs.items():
            p_dir_name = paths['pylint']
            path = self._find_dir(p_dir_name)
            if not path: continue
            
            for repo_folder in path.iterdir():
                if not repo_folder.is_dir(): continue
                json_files = list(repo_folder.glob("*.json"))
                if not json_files: continue
                
                total_files += 1
                current_repo_counts = Counter()
                
                try:
                    with open(json_files[0], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for issue in data:
                                symbol = issue.get('symbol', 'unknown')
                                category = self._classify_smell(symbol)
                                
                                if category == 'Valid':
                                    self.valid_smell_counts[symbol] += 1
                                    current_repo_counts[symbol] += 1
                                else:
                                    self.ignored_stats[category] += 1
                except: pass
                
                self.repo_smell_data.append({
                    'repo': repo_folder.name, 
                    'size': size, 
                    'counts': current_repo_counts
                })

        self.top_20_smells = [s[0] for s in self.valid_smell_counts.most_common(20)]
        
        # REPORT 1: IGNORED
        total_ignored = sum(self.ignored_stats.values())
        print(f"\n[+] Scanned {total_files} Pylint files.")
        print(f"[+] Ignored {total_ignored} messages (Noise). Breakdown:")
        for cat, count in self.ignored_stats.most_common():
            print(f"    {cat:<25} : {count}")

        # REPORT 2: TOP 20 VALID
        print("\n" + "="*80)
        print("TOP 20 VALID PYTHON SMELLS (Functional Logic Only)")
        print("="*80)
        print(f"{'Rank':<5} {'Smell Symbol':<30} {'Total':<10} {'Median':<8} {'Max'}")
        print("-" * 80)
        for i, smell in enumerate(self.top_20_smells, 1):
            total = self.valid_smell_counts[smell]
            counts = [d['counts'].get(smell, 0) for d in self.repo_smell_data]
            median = np.median(counts) if counts else 0
            max_val = np.max(counts) if counts else 0
            print(f"{i:<5} {smell:<30} {total:<10} {median:<8.1f} {max_val}")

        # REPORT 3: SIZE BREAKDOWN
        print("\n" + "="*80)
        print("MOST FREQUENT SMELLS BY PROJECT SIZE (Top 5)")
        print("="*80)
        for size in ['small', 'medium', 'large']:
            size_entries = [d for d in self.repo_smell_data if d['size'] == size]
            size_counts = Counter()
            for entry in size_entries: 
                size_counts.update(entry['counts'])
            
            print(f"[{size.upper()}] (n={len(size_entries)})")
            for rank, (s, c) in enumerate(size_counts.most_common(5), 1):
                print(f"  {rank}. {s:<30} : {c}")
            print()

    def load_paired_data(self):
        """Match ML (12 types only) and Python (Top 20) data"""
        print("STEP 2: Matching ML & Python Data...")
        
        # Try to load from combined_analysis_data.csv first
        use_combined = False
        ml_lookup = {}
        
        combined_path = Path('combined_analysis_data.csv')
        if combined_path.exists():
            try:
                print("[+] Loading ML data from combined_analysis_data.csv...")
                ml_data = pd.read_csv(combined_path)
                
                # Filter ML smells to only the 12 types
                cols_present = [c for c in ML_SMELL_TYPES if c in ml_data.columns]
                if cols_present:
                    ml_data['ml_count_filtered'] = ml_data[cols_present].sum(axis=1)
                    ml_data = ml_data[ml_data['loc'] > 0].copy()
                    ml_data['ml_density_filtered'] = ml_data['ml_count_filtered'] / (ml_data['loc'] / 1000)
                    ml_data['repo_name'] = ml_data['repo_name'].str.strip()
                    
                    # Create lookup dictionary
                    ml_lookup = dict(zip(ml_data['repo_name'], ml_data['ml_density_filtered']))
                    use_combined = True
                    
                    print(f"[+] Loaded {len(ml_data)} repositories from combined_analysis_data.csv")
                    print(f"[+] Using {len(cols_present)} ML smell types: {', '.join(cols_present)}")
                else:
                    print("[!] Warning: No ML smell columns found in combined_analysis_data.csv")
            except Exception as e:
                print(f"[!] Error loading combined_analysis_data.csv: {e}")
        
        if not use_combined:
            print("[+] Will read from individual overview.csv files instead...")
        
        # Now match with Pylint data
        paired_records = []
        files = {
            'small': 'repos_small_checked.csv', 
            'medium': 'repos_medium_checked.csv', 
            'large': 'repos_large_checked.csv'
        }

        for size, csv_file in files.items():
            csv_path = self._find_dir(csv_file) or Path(csv_file)
            if not csv_path.exists(): 
                csv_path = Path(f'repos_{size}.csv')
            if not csv_path.exists(): 
                continue
            
            try:
                repo_df = pd.read_csv(csv_path)
                repo_df['Lines of Code'] = pd.to_numeric(repo_df['Lines of Code'], errors='coerce')
                repo_df = repo_df.dropna(subset=['Lines of Code'])
                repo_df = repo_df[repo_df['Lines of Code'] > 0]
                
                cs_path = self._find_dir(self.dirs[size]['codesmile']) if not use_combined else None
                py_path = self._find_dir(self.dirs[size]['pylint'])
                if not py_path: 
                    continue

                for _, row in repo_df.iterrows():
                    repo_name = row['GitHub Repo']
                    loc = row['Lines of Code']
                    patterns = [repo_name.replace('/', '_'), repo_name.split('/')[-1]]
                    
                    # --- ML SMELL DENSITY ---
                    ml_density = 0
                    
                    if use_combined:
                        # Use combined_analysis_data.csv lookup
                        ml_density = ml_lookup.get(repo_name, 0)
                    else:
                        # Read from individual overview.csv files
                        if cs_path:
                            for p in patterns:
                                matches = list(cs_path.glob(f"*{p}*"))
                                if matches:
                                    ov = matches[0] / "overview.csv"
                                    if ov.exists():
                                        try:
                                            ov_df = pd.read_csv(ov)
                                            # Count only the 12 smell types
                                            if 'smell_name' in ov_df.columns:
                                                ml_count = ov_df[ov_df['smell_name'].isin(ML_SMELL_TYPES)].shape[0]
                                                ml_density = (ml_count / loc) * 1000
                                        except: 
                                            pass
                                    break
                    
                    # --- PYTHON SMELL COUNT (Top 20) ---
                    py_count = 0
                    for p in patterns:
                        matches = list(py_path.glob(f"*{p}*"))
                        if matches:
                            jsons = list(matches[0].glob("*.json"))
                            if jsons:
                                try:
                                    with open(jsons[0], 'r') as f:
                                        data = json.load(f)
                                        py_count = sum(1 for x in data if x.get('symbol') in self.top_20_smells)
                                except: 
                                    pass
                            break
                    
                    paired_records.append({
                        'Repo': repo_name, 
                        'Size': size, 
                        'LOC': loc,
                        'ML_Density': ml_density, 
                        'Python_Density': (py_count/loc)*1000
                    })
            except Exception as e:
                print(f"[!] Error processing {size}: {e}")

        self.df = pd.DataFrame(paired_records)
        print(f"[+] Paired {len(self.df)} repositories.")

    def run_test(self):
        """Perform statistical test and generate visualizations"""
        if self.df.empty: 
            print("ERROR: No paired data available!")
            return
            
        print("\n" + "="*70)
        print("H0.0: STATISTICAL TEST (ML-Specific vs General Python)")
        print("="*70)
        
        # Descriptive stats by size
        print(f"\n{'Group':<10} {'ML-Specific':<15} {'General Python'}")
        print("-" * 45)
        for size in ['small', 'medium', 'large']:
            subset = self.df[self.df['Size'] == size]
            if not subset.empty:
                print(f"{size:<10} {subset['ML_Density'].median():<15.3f} {subset['Python_Density'].median():.3f}")

        # Overall statistics
        ml, py = self.df['ML_Density'], self.df['Python_Density']
        
        print(f"\n{'Metric':<20} | {'Median':<10} | {'Mean':<10} | {'Std Dev':<10}")
        print("-" * 58)
        print(f"{'ML-Specific':<20} | {ml.median():<10.3f} | {ml.mean():<10.3f} | {ml.std():<10.3f}")
        print(f"{'General Python':<20} | {py.median():<10.3f} | {py.mean():<10.3f} | {py.std():<10.3f}")
        print("-" * 58)
        
        # Wilcoxon test
        w_stat, p = stats.wilcoxon(ml, py)
        
        # Cliff's Delta
        delta = np.mean(np.sign(np.subtract.outer(np.array(ml), np.array(py))))
        
        def interpret_effect(d):
            d = abs(d)
            if d < 0.147: return "Negligible"
            elif d < 0.33: return "Small"
            elif d < 0.474: return "Medium"
            else: return "Large"
        
        print(f"\nSample Size (N): {len(self.df)}")
        print(f"Wilcoxon W:      {w_stat:.1f}")
        print(f"p-value:         {p:.4e}")
        print(f"Cliff's Delta:   {delta:.4f} ({interpret_effect(delta)})")
        print("-" * 70)
        
        if p < 0.05: 
            print("Result: REJECT H0.0 (Significant difference exists)")
        else: 
            print("Result: FAIL TO REJECT H0.0 (No significant difference)")
        
        # Save results
        self._save_results(w_stat, p, delta, interpret_effect(delta))
        self._plot(ml, py)

    def _save_results(self, w_stat, p_val, delta, effect):
        """Save test results to CSV"""
        results_df = pd.DataFrame([{
            'Hypothesis': 'H0.0',
            'Test': 'Wilcoxon Signed-Rank',
            'N': len(self.df),
            'Statistic': w_stat,
            'p-value': p_val,
            'Cliffs_Delta': delta,
            'Effect_Size': effect,
            'Median_ML': self.df['ML_Density'].median(),
            'Median_Python': self.df['Python_Density'].median(),
            'Mean_ML': self.df['ML_Density'].mean(),
            'Mean_Python': self.df['Python_Density'].mean()
        }])
        results_df.to_csv('h0_combined_results.csv', index=False)
        print("\n[+] Saved: h0_combined_results.csv")
        
        # CRITICAL: Save paired data for H7-H12
        self.df.to_csv('h0_paired_data.csv', index=False)
        print("[+] Saved: h0_paired_data.csv (for H7-H12)")
        print(f"    Contains {len(self.df)} repositories with ML and Python densities")

    def _plot(self, ml, py):
        """Generate comparison visualizations"""
        # Plot 1: Box plot
        plt.figure(figsize=(8, 6))
        data = pd.DataFrame({
            'ML-Specific': ml, 
            'General Python': py
        }).melt(var_name='Smell Type', value_name='Density')
        
        sns.boxplot(data=data, x='Smell Type', y='Density', 
                   showfliers=False, palette=['#3498db', '#9b59b6'])
        plt.title('Code Smell Density Comparison\n(12 ML Types vs Top 20 Python Smells)')
        plt.ylabel('Smells per 1,000 LOC')
        plt.tight_layout()
        plt.savefig('h0_combined_boxplot.png', dpi=300)
        print("[+] Saved: h0_combined_boxplot.png")
        
        # Plot 2: Scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(ml, py, alpha=0.5, c='purple', s=30)
        
        max_limit = max(ml.max(), py.max())
        plt.plot([0, max_limit], [0, max_limit], 'r--', 
                label='x=y (Equal Density)', linewidth=2)
        
        plt.xlabel('ML-Specific Smell Density (per 1K LOC)')
        plt.ylabel('General Python Smell Density (per 1K LOC)')
        plt.title('Scatter Plot: ML vs. General Python Density')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('h0_combined_scatter.png', dpi=300)
        print("[+] Saved: h0_combined_scatter.png")

if __name__ == "__main__":
    print("="*70)
    print("H0.0: ML-SPECIFIC VS GENERAL PYTHON CODE SMELLS (COMBINED)")
    print("="*70)
    print("\nML Smells Configuration:")
    print("  ✓ Including 12 types:")
    for smell in ML_SMELL_TYPES:
        print(f"    - {smell}")
    print("\n  ✗ Excluding 4 types:")
    excluded = ['hyperparameters_not_explicitly_set', 'empty_column_misinitialization',
                'Broadcasting_Feature_Not_Used', 'deterministic_algorithm_option_not_used']
    for smell in excluded:
        print(f"    - {smell}")
    print("\nPython Smells: Top 20 functional (excluding imports/style/docstrings)")
    print("="*70)
    
    analyzer = SmellComparator()
    analyzer.analyze_pylint_smells()
    analyzer.load_paired_data()
    analyzer.run_test()