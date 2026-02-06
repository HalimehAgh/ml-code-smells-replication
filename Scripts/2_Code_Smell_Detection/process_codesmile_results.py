import pandas as pd
import os
from pathlib import Path

print("=" * 80)
print("CodeSmile Results ANALYSIS")
print("=" * 80)

# ===== CONFIGURATION =====
CODESMILE_DIR = os.getcwd()
PROJECT_SIZES = ['small', 'medium', 'large']

# 1. Collect all unique smell names first to build the column list
all_smell_types = set()
for size in PROJECT_SIZES:
    analysis_dir = os.path.join(CODESMILE_DIR, f'{size}_codesmile_analysis')
    status_file = os.path.join(analysis_dir, 'analysis_status.csv')
    if os.path.exists(status_file):
        status_df = pd.read_csv(status_file)
        # Only check successful ones to find possible smell names
        for _, row in status_df[status_df['status'] == 'success'].iterrows():
            # Use the output_path saved in the analysis_status.csv
            ov_path = os.path.join(row['output_path'], 'overview.csv')
            if os.path.exists(ov_path):
                ov_df = pd.read_csv(ov_path)
                if not ov_df.empty and 'smell_name' in ov_df.columns:
                    all_smell_types.update(ov_df['smell_name'].unique())

smell_columns = sorted(list(all_smell_types))
all_results = []

# 2. Process EVERY repository from the original source CSVs
for size in PROJECT_SIZES:
    print(f"\nProcessing {size.upper()} batch...")
    
    REPOS_CSV = f'repos_{size}.csv' # Your source of truth
    ANALYSIS_STATUS = os.path.join(CODESMILE_DIR, f'{size}_codesmile_analysis', 'analysis_status.csv')
    
    if not os.path.exists(REPOS_CSV):
        print(f"✗ Skipping {size}: {REPOS_CSV} not found")
        continue

    # Load the base list of repositories
    base_df = pd.read_csv(REPOS_CSV)
    
    # Load analysis results for lookup
    analysis_lookup = {}
    if os.path.exists(ANALYSIS_STATUS):
        status_df = pd.read_csv(ANALYSIS_STATUS)
        # Map original index to the output directory
        analysis_lookup = status_df[status_df['status'] == 'success'].set_index('index')['output_path'].to_dict()

    for idx, row in base_df.iterrows():
        # Get the repo name from the GitHub URL in your source CSV
        repo_url = str(row['GitHub Repo'])
        display_name = repo_url.replace('https://github.com/', '').strip('/')
        loc = row['Lines of Code']
        
        # Initialize the record with 0s 
        record = {
            'repo_name': display_name,
            'loc': loc,
            'size_category': size,
            'total_smells': 0,
            'smell_density': 0.0
        }
        
        # Pre-fill all smell columns with 0.0
        for s in smell_columns:
            record[s] = 0.0

        # If we have a successful analysis for this index, fill the data
        output_dir = analysis_lookup.get(idx)
        if output_dir:
            overview_file = os.path.join(output_dir, 'overview.csv')
            if os.path.exists(overview_file):
                ov_df = pd.read_csv(overview_file)
                if not ov_df.empty:
                    counts = ov_df['smell_name'].value_counts()
                    total = counts.sum()
                    record['total_smells'] = total
                    record['smell_density'] = (total / loc * 1000) if loc > 0 else 0.0
                    
                    # Add raw counts for individual smells 
                    for s_name, s_count in counts.items():
                        record[s_name] = float(s_count)

        all_results.append(record)

# 3. Create Final DataFrame and enforce structure
final_df = pd.DataFrame(all_results)

# Arrange columns to match combined_analysis_data.csv exactly 
ordered_cols = ['repo_name', 'loc', 'size_category', 'total_smells', 'smell_density'] + smell_columns
final_df = final_df[ordered_cols]

# Save output
output_file = os.path.join(CODESMILE_DIR, 'combined_analysis_data.csv')
final_df.to_csv(output_file, index=False)

print(f"✓ Output saved to: {output_file}")