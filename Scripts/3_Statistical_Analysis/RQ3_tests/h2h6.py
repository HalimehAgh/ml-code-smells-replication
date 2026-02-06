#!/usr/bin/env python3
"""
Hypothesis Testing H2-H6: Pylint Code Smells Analysis (FIXED)
=============================================================
H2: Age vs Smells
H3: Contributors vs Smells  
H4: Commit Frequency vs Smells
H5: CI/CD vs Smells
H6: Domain vs Smells

Data Sources:
- age_analysis_data.csv: Age, domain, commits, contributor count
- repos_{size}_checked.csv: CI/CD status
- {size}_pylint_results/: Pylint JSON files

Uses Top 20 Valid Python Smells (excludes imports, style, docstrings)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

class PylintComprehensiveAnalyzer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir).resolve()
        
        # Pylint result directories
        self.pylint_dirs = {
            'small': 'small_pylint_results',
            'medium': 'medium_pylint_results',
            'large': 'large_pylint_results'
        }
        
        # Top 20 Valid Python Smells
        self.top_20_smells = [
            'unused-argument', 'no-member', 'protected-access', 'no-self-use',
            'too-many-arguments', 'redefined-outer-name', 'too-many-locals',
            'unsubscriptable-object', 'syntax-error', 'too-few-public-methods',
            'attribute-defined-outside-init', 'abstract-method', 'no-else-return',
            'unused-variable', 'fixme', 'too-many-instance-attributes',
            'redefined-builtin', 'arguments-differ', 'logging-fstring-interpolation',
            'no-value-for-parameter'
        ]
        
        self.df = pd.DataFrame()
        self.alpha = 0.05

    def _find_dir(self, dirname):
        """Robustly find directory"""
        candidates = [self.base_dir / dirname, self.base_dir.parent / dirname]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c.resolve()
        return None

    def _find_file(self, filename):
        """Robustly find file"""
        candidates = [self.base_dir / filename, self.base_dir.parent / filename, Path(filename)]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    def load_comprehensive_data(self):
        """Load metadata from age_analysis_data.csv and merge with CI info"""
        print("="*70)
        print("COMPREHENSIVE PYLINT ANALYSIS (H2-H6)")
        print("="*70)
        print("\nStep 1: Loading age_analysis_data.csv...")
        
        # Load age/domain/metadata file
        age_file = self._find_file('age_analysis_data.csv')
        if not age_file:
            raise ValueError("age_analysis_data.csv not found!")
        
        age_df = pd.read_csv(age_file)
        print(f"  [+] Loaded {len(age_df)} repositories with age/domain data")
        
        # Clean repo names
        age_df['GitHub Repo'] = age_df['GitHub Repo'].str.strip()
        
        domain_file = self._find_file('repos_with_domains_final_complete.csv')
        if domain_file:
            domain_df = pd.read_csv(domain_file)
            domain_df['GitHub Repo'] = domain_df['GitHub Repo'].str.strip()
            # Extract only repo name and domain
            domain_df = domain_df[['GitHub Repo', 'domain']]
            # Drop domain from age_df if it exists, then merge with correct domains
            if 'domain' in age_df.columns:
                age_df = age_df.drop(columns=['domain'])
            age_df = age_df.merge(domain_df, on='GitHub Repo', how='left')
            print(f"  [+] Updated domain information from repos_with_domains_final_complete.csv")
        
        # Check required columns
        req_cols = ['GitHub Repo', 'Lines of Code', 'age_days', 'Contributor Count', 
                    'Commits', 'domain']
        missing = [c for c in req_cols if c not in age_df.columns]
        if missing:
            raise ValueError(f"Missing columns in age_analysis_data.csv: {missing}")
        
        print("\nStep 2: Loading CI/CD data from repos_*_checked.csv...")
        
        # Load CI data from checked files
        ci_data = []
        for size in ['small', 'medium', 'large']:
            csv_file = self._find_file(f'repos_{size}_checked.csv')
            if not csv_file:
                print(f"  [!] Warning: repos_{size}_checked.csv not found")
                continue
            
            size_df = pd.read_csv(csv_file)
            size_df['GitHub Repo'] = size_df['GitHub Repo'].str.strip()
            
            # Extract only repo name and CI status
            if 'CI_Exists_Checked' in size_df.columns:
                size_df['has_ci'] = size_df['CI_Exists_Checked'].apply(
                    lambda x: str(x).strip().upper() == 'YES'
                )
                size_df['size_category'] = size
                ci_data.append(size_df[['GitHub Repo', 'has_ci', 'size_category']])
                print(f"  [+] Loaded {size}: {len(size_df)} repos with CI data")
            else:
                print(f"  [!] Warning: CI_Exists_Checked not found in {size}")
        
        if not ci_data:
            raise ValueError("No CI data loaded!")
        
        ci_df = pd.concat(ci_data, ignore_index=True)
        
        # Merge age data with CI data
        merged_df = age_df.merge(ci_df, on='GitHub Repo', how='inner')
        print(f"\n  [+] Merged dataset: {len(merged_df)} repositories")
        
        # Filter valid data
        merged_df = merged_df[merged_df['Lines of Code'] > 0]
        merged_df = merged_df[merged_df['age_days'] > 0]
        merged_df = merged_df.dropna(subset=['Contributor Count', 'Commits'])
        
        print(f"  [+] After filtering: {len(merged_df)} valid repositories")
        
        print("\nStep 3: Scanning Pylint results...")
        
        # Calculate commit frequency
        merged_df['commit_freq'] = merged_df['Commits'] / (merged_df['age_days'] / 365.25)
        
        # Scan Pylint results
        smell_counts = []
        
        for _, row in merged_df.iterrows():
            repo_name = row['GitHub Repo']
            size = row['size_category']
            
            # Find Pylint results directory
            pylint_path = self._find_dir(self.pylint_dirs[size])
            if not pylint_path:
                smell_counts.append(0)
                continue
            
            patterns = [repo_name.replace('/', '_'), repo_name.split('/')[-1]]
            
            total_smells = 0
            found = False
            
            for pattern in patterns:
                matches = list(pylint_path.glob(f"*{pattern}*"))
                if matches:
                    json_files = list(matches[0].glob("*.json"))
                    if json_files:
                        try:
                            with open(json_files[0], 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for issue in data:
                                        if issue.get('symbol', '') in self.top_20_smells:
                                            total_smells += 1
                            found = True
                        except:
                            pass
                    break
            
            smell_counts.append(total_smells)
        
        merged_df['total_smells'] = smell_counts
        merged_df['smell_density'] = (merged_df['total_smells'] / merged_df['Lines of Code']) * 1000
        
        # Create categorical variables
        merged_df['age_category'] = merged_df['age_days'].apply(
            lambda x: 'Young' if x < merged_df['age_days'].median() else 'Mature'
        )
        
        merged_df['team_size'] = merged_df['Contributor Count'].apply(
            lambda x: 'Small' if x <= merged_df['Contributor Count'].median() else 'Large'
        )
        
        merged_df['activity_level'] = merged_df['commit_freq'].apply(
            lambda x: 'Low' if x < merged_df['commit_freq'].median() else 'High'
        )
        
        self.df = merged_df
        
        # Save
        self.df.to_csv('pylint_comprehensive_data.csv', index=False)
        print(f"\n  [+] Final dataset: {len(self.df)} repositories")
        print(f"  [+] Saved to: pylint_comprehensive_data.csv")
        
        # Summary stats
        print(f"\n  CI Status: {sum(self.df['has_ci'])} with CI, {len(self.df) - sum(self.df['has_ci'])} without")
        print(f"  Domains: {self.df['domain'].nunique()} unique domains")
        print(f"  Age range: {self.df['age_days'].min():.0f} to {self.df['age_days'].max():.0f} days")

    def _cliffs_delta(self, x, y):
        x, y = np.array(x), np.array(y)
        if len(x) == 0 or len(y) == 0:
            return np.nan
        return np.mean(np.sign(np.subtract.outer(x, y)))

    def _interpret_rho(self, rho):
        abs_rho = abs(rho)
        return "Weak" if abs_rho < 0.3 else "Moderate" if abs_rho < 0.7 else "Strong"

    def _interpret_delta(self, delta):
        abs_delta = abs(delta)
        if abs_delta < 0.15: return "Negligible"
        elif abs_delta < 0.33: return "Small"
        elif abs_delta < 0.47: return "Medium"
        else: return "Large"

    def run_h2_age(self):
        """H2.0 & H2.1: Age correlation and group comparison"""
        print("\n" + "="*70)
        print("H2: AGE VS CODE SMELLS")
        print("="*70)
        
        # H2.0: Correlation
        print("\nH2.0: Correlation Analysis")
        rho, p = stats.spearmanr(self.df['age_days'], self.df['smell_density'])
        print(f"  Spearman ρ = {rho:.4f}, p = {p:.2e}")
        print(f"  Strength: {self._interpret_rho(rho)}")
        print(f"  Decision: {'REJECT H2.0' if p < self.alpha else 'FAIL TO REJECT H2.0'}")
        
        # H2.1: Group comparison
        print("\nH2.1: Group Comparison (Young vs Mature)")
        young = self.df[self.df['age_category'] == 'Young']['smell_density']
        mature = self.df[self.df['age_category'] == 'Mature']['smell_density']
        
        median_age_days = self.df['age_days'].median()
        median_age_years = median_age_days / 365.25
        
        print(f"  Threshold: {median_age_years:.2f} years ({median_age_days:.0f} days)")
        print(f"  Young (< threshold): n={len(young)}, median={young.median():.3f}")
        print(f"  Mature (>= threshold): n={len(mature)}, median={mature.median():.3f}")
        
        u_stat, p_mw = stats.mannwhitneyu(young, mature, alternative='two-sided')
        delta = self._cliffs_delta(young, mature)
        
        print(f"  Mann-Whitney U = {u_stat:.1f}, p = {p_mw:.2e}")
        print(f"  Cliff's δ = {delta:.4f} ({self._interpret_delta(delta)})")
        print(f"  Decision: {'REJECT H2.1' if p_mw < self.alpha else 'FAIL TO REJECT H2.1'}")

    def run_h3_contributors(self):
        """H3.0 & H3.1: Contributors correlation and group comparison"""
        print("\n" + "="*70)
        print("H3: CONTRIBUTORS VS CODE SMELLS")
        print("="*70)
        
        # H3.0: Correlation
        print("\nH3.0: Correlation Analysis")
        rho, p = stats.spearmanr(self.df['Contributor Count'], self.df['smell_density'])
        print(f"  Spearman ρ = {rho:.4f}, p = {p:.2e}")
        print(f"  Strength: {self._interpret_rho(rho)}")
        print(f"  Decision: {'REJECT H3.0' if p < self.alpha else 'FAIL TO REJECT H3.0'}")
        
        # H3.1: Group comparison
        print("\nH3.1: Group Comparison (Small vs Large Teams)")
        small = self.df[self.df['team_size'] == 'Small']['smell_density']
        large = self.df[self.df['team_size'] == 'Large']['smell_density']
        
        threshold = self.df['Contributor Count'].median()
        
        print(f"  Threshold: {threshold} contributors")
        print(f"  Small Teams (<= threshold): n={len(small)}, median={small.median():.3f}")
        print(f"  Large Teams (> threshold): n={len(large)}, median={large.median():.3f}")
        
        u_stat, p_mw = stats.mannwhitneyu(small, large, alternative='two-sided')
        delta = self._cliffs_delta(small, large)
        
        print(f"  Mann-Whitney U = {u_stat:.1f}, p = {p_mw:.2e}")
        print(f"  Cliff's δ = {delta:.4f} ({self._interpret_delta(delta)})")
        print(f"  Decision: {'REJECT H3.1' if p_mw < self.alpha else 'FAIL TO REJECT H3.1'}")

    def run_h4_commits(self):
        """H4.0 & H4.1: Commit frequency correlation and group comparison"""
        print("\n" + "="*70)
        print("H4: COMMIT FREQUENCY VS CODE SMELLS")
        print("="*70)
        
        # H4.0: Correlation
        print("\nH4.0: Correlation Analysis")
        rho, p = stats.spearmanr(self.df['commit_freq'], self.df['smell_density'])
        print(f"  Spearman ρ = {rho:.4f}, p = {p:.2e}")
        print(f"  Strength: {self._interpret_rho(rho)}")
        print(f"  Decision: {'REJECT H4.0' if p < self.alpha else 'FAIL TO REJECT H4.0'}")
        
        # H4.1: Group comparison
        print("\nH4.1: Group Comparison (Low vs High Activity)")
        low = self.df[self.df['activity_level'] == 'Low']['smell_density']
        high = self.df[self.df['activity_level'] == 'High']['smell_density']
        
        threshold = self.df['commit_freq'].median()
        
        print(f"  Threshold: {threshold:.1f} commits/year")
        print(f"  Low Activity (< threshold): n={len(low)}, median={low.median():.3f}")
        print(f"  High Activity (>= threshold): n={len(high)}, median={high.median():.3f}")
        
        u_stat, p_mw = stats.mannwhitneyu(low, high, alternative='two-sided')
        delta = self._cliffs_delta(low, high)
        
        print(f"  Mann-Whitney U = {u_stat:.1f}, p = {p_mw:.2e}")
        print(f"  Cliff's δ = {delta:.4f} ({self._interpret_delta(delta)})")
        print(f"  Decision: {'REJECT H4.1' if p_mw < self.alpha else 'FAIL TO REJECT H4.1'}")

    def run_h5_cicd(self):
        """H5.0: CI/CD comparison"""
        print("\n" + "="*70)
        print("H5: CI/CD VS CODE SMELLS")
        print("="*70)
        
        ci_count = sum(self.df['has_ci'])
        no_ci_count = len(self.df) - ci_count
        
        if ci_count < 5 or no_ci_count < 5:
            print("  [!] Insufficient data for CI/CD comparison")
            print(f"      CI: {ci_count}, No-CI: {no_ci_count}")
            return
        
        print("\nH5.0: CI vs Non-CI Comparison")
        ci = self.df[self.df['has_ci']]['smell_density']
        no_ci = self.df[~self.df['has_ci']]['smell_density']
        
        print(f"  With CI: n={len(ci)}, median={ci.median():.3f}")
        print(f"  Without CI: n={len(no_ci)}, median={no_ci.median():.3f}")
        
        u_stat, p_mw = stats.mannwhitneyu(ci, no_ci, alternative='two-sided')
        delta = self._cliffs_delta(ci, no_ci)
        
        print(f"  Mann-Whitney U = {u_stat:.1f}, p = {p_mw:.2e}")
        print(f"  Cliff's δ = {delta:.4f} ({self._interpret_delta(delta)})")
        print(f"  Decision: {'REJECT H5.0' if p_mw < self.alpha else 'FAIL TO REJECT H5.0'}")

    def run_h6_domain(self):
        """H6.0: Domain comparison"""
        print("\n" + "="*70)
        print("H6: DOMAIN VS CODE SMELLS")
        print("="*70)
        
        # Filter domains with sufficient data
        domain_counts = self.df['domain'].value_counts()
        valid_domains = domain_counts[domain_counts >= 5].index
        
        if len(valid_domains) < 2:
            print("  [!] Insufficient domains for comparison")
            return
        
        print(f"\nH6.0: Kruskal-Wallis Test ({len(valid_domains)} domains with n>=5)")
        
        groups = [self.df[self.df['domain'] == d]['smell_density'].values 
                 for d in valid_domains]
        
        h_stat, p_kw = stats.kruskal(*groups)
        
        print(f"  H-statistic = {h_stat:.4f}, p = {p_kw:.2e}")
        print(f"  Decision: {'REJECT H6.0' if p_kw < self.alpha else 'FAIL TO REJECT H6.0'}")
        
        # Show medians
        print("\n  Domain Medians (sorted by smell density):")
        domain_stats = []
        for domain in valid_domains:
            med = self.df[self.df['domain'] == domain]['smell_density'].median()
            n = len(self.df[self.df['domain'] == domain])
            domain_stats.append((domain, med, n))
        
        domain_stats.sort(key=lambda x: x[1], reverse=True)
        for domain, med, n in domain_stats:
            print(f"    {domain:<35} median={med:6.3f} (n={n})")

    def run_all_tests(self):
        """Run complete H2-H6 analysis"""
        self.load_comprehensive_data()
        self.run_h2_age()
        self.run_h3_contributors()
        self.run_h4_commits()
        self.run_h5_cicd()
        self.run_h6_domain()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("\nGenerated file: pylint_comprehensive_data.csv")


if __name__ == "__main__":
    analyzer = PylintComprehensiveAnalyzer()
    analyzer.run_all_tests()