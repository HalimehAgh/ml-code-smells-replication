#!/usr/bin/env python3
"""
Parse Pylint outputs (Top 20)
====================================================================
Creates a CSV with top 20 Python smell type columns for each repository,
import errors are ignored.

Outputs:
- pylint_smells.csv (complete dataset)
- python_smell_heatmap_by_domain.png (visualization)
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

class PythonSmellExtractor:
    def __init__(self, base_dir=".", domain_file='repos_with_domains_final_complete.csv'):
        self.base_dir = Path(base_dir).resolve()
        self.domain_file = domain_file
        
        self.dirs = {
            'small': 'small_pylint_results',
            'medium': 'medium_pylint_results',
            'large': 'large_pylint_results'
        }
        
        self.top_20_smells = []
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

    def _find_file(self, filename):
        """Find file in multiple locations"""
        candidates = [
            Path(filename),
            self.base_dir / filename,
            self.base_dir.parent / filename,
            self.base_dir / ".." / filename
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        return None

    def _find_dir(self, dirname):
        """Find directory in multiple locations"""
        candidates = [
            self.base_dir / dirname,
            self.base_dir.parent / dirname,
            self.base_dir / ".." / dirname
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c.resolve()
        return None

    def _classify_smell(self, symbol):
        """Classify smell as Valid or ignored category"""
        if not symbol:
            return 'Unknown'
        return self.ignored_lookup.get(symbol, 'Valid')

    def scan_pylint_results(self):
        """Scan Pylint results and extract Top 20 valid functional smells"""
        print("=" * 80)
        print("STEP 1: Scanning Pylint Results")
        print("=" * 80)
        
        for size, dir_name in self.dirs.items():
            path = self._find_dir(dir_name)
            if not path:
                print(f"[!] Warning: {dir_name} not found, skipping...")
                continue
            
            print(f"\n[+] Processing {size}_pylint_results/...")
            repo_count = 0
            
            for repo_folder in path.iterdir():
                if not repo_folder.is_dir():
                    continue
                
                json_files = list(repo_folder.glob("*.json"))
                if not json_files:
                    continue
                
                repo_count += 1
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
                except Exception as e:
                    print(f"    [!] Error reading {repo_folder.name}: {e}")
                    continue
                
                self.repo_smell_data.append({
                    'repo_folder': repo_folder.name,
                    'size': size,
                    'counts': current_repo_counts
                })
            
            print(f"    ✓ Processed {repo_count} repositories")
        
        # Get Top 20 smells
        self.top_20_smells = [s[0] for s in self.valid_smell_counts.most_common(20)]
        
        print(f"\n[+] Total repositories scanned: {len(self.repo_smell_data)}")
        print(f"[+] Top 20 valid functional smells identified")
        print("\nTop 20 smells:")
        for i, (smell, count) in enumerate(self.valid_smell_counts.most_common(20), 1):
            print(f"  {i:2d}. {smell:<40} {count:>8} occurrences")

    def create_dataset_with_domains(self):
        """Create dataset with smell counts merged with domain and project data"""
        print("\n" + "=" * 80)
        print("STEP 2: Merging with Domain and Project Data")
        print("=" * 80)
        
        # Load domain data
        domain_path = self._find_file(self.domain_file)
        if not domain_path:
            print(f"[!] ERROR: {self.domain_file} not found!")
            return None
        
        domain_df = pd.read_csv(domain_path)
        
        # Filter to repos with LOC > 0
        domain_df = domain_df[domain_df['Lines of Code'] > 0].copy()
        
        # Prepare results
        final_records = []
        
        for _, row in domain_df.iterrows():
            repo_name = row['GitHub Repo']
            
            # Find matching Pylint data
            patterns = [
                repo_name.replace('/', '_'),
                repo_name.split('/')[-1]
            ]
            
            matched = False
            for pattern in patterns:
                for data in self.repo_smell_data:
                    if pattern in data['repo_folder']:
                        # Create record with project characteristics
                        record = {
                            'GitHub Repo': repo_name,
                            'domain': row.get('domain', 'Unknown'),
                            'Lines of Code': row['Lines of Code'],
                            'Stars': row.get('Stars', 0),
                            'Commits': row.get('Commits', 0),
                            'Contributor Count': row.get('Contributor Count', 0),
                            'Project Size': row.get('Project Size', data['size']),
                        }
                        
                        # Add count for each of the Top 20 smells
                        for smell in self.top_20_smells:
                            record[smell] = data['counts'].get(smell, 0)
                        
                        final_records.append(record)
                        matched = True
                        break
                if matched:
                    break
            
            if not matched:
                # Repo exists in domain file but no Pylint data - add with zeros
                record = {
                    'GitHub Repo': repo_name,
                    'domain': row.get('domain', 'Unknown'),
                    'Lines of Code': row['Lines of Code'],
                    'Stars': row.get('Stars', 0),
                    'Commits': row.get('Commits', 0),
                    'Contributor Count': row.get('Contributor Count', 0),
                    'Project Size': row.get('Project Size', 'unknown'),
                }
                for smell in self.top_20_smells:
                    record[smell] = 0
                final_records.append(record)
        
        df = pd.DataFrame(final_records)
        
        # Calculate total smells and density
        df['Total_Python_Smells'] = df[self.top_20_smells].sum(axis=1)
        df['Python_Smell_Density'] = (df['Total_Python_Smells'] / df['Lines of Code']) * 1000
                
        return df

    def create_heatmap(self, df, output_file='python_smell_heatmap_by_domain.png'):
        """Create heatmap showing smell distribution by domain"""
        print("\n" + "=" * 80)
        print("STEP 3: Creating Heatmap by Domain")
        print("=" * 80)
        
        # Calculate mean smell counts by domain
        domain_means = df.groupby('domain')[self.top_20_smells].mean()
        
        # Calculate percentages (normalize each domain to 100%)
        domain_percentages = domain_means.div(domain_means.sum(axis=1), axis=0) * 100
        
        # Sort domains by total smell density
        domain_totals = df.groupby('domain')['Python_Smell_Density'].median()
        domain_order = domain_totals.sort_values(ascending=False).index
        domain_percentages = domain_percentages.loc[domain_order]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create heatmap
        sns.heatmap(
            domain_percentages,
            annot=False,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Percentage of Total Smells in Domain (%)'},
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )
        
        ax.set_xlabel('Python Smell Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('ML Domain', fontsize=12, fontweight='bold')
        ax.set_title('Python Smell Distribution by ML Domain\n(Top 20 Functional Smells)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n Saved heatmap: {output_file}")
        plt.close()
        
    def save_csv(self, df, filename='pylint_smells.csv'):
        """Save the final CSV"""
        print("\n" + "=" * 80)
        print("STEP 4: Saving Complete Dataset")
        print("=" * 80)
        
        df.to_csv(filename, index=False)
        
        print(f"\n Saved: {filename}")


def main():
    
    extractor = PythonSmellExtractor()
    extractor.scan_pylint_results()
    df = extractor.create_dataset_with_domains()
    
    if df is not None and not df.empty:
        extractor.save_csv(df)
        extractor.create_heatmap(df)
    else:
        print("\n[!] ERROR: No data generated. Check that domain file exists.")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
if __name__ == "__main__":
    main()