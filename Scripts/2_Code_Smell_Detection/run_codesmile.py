import pandas as pd
import subprocess
import os
import time
from datetime import datetime
from pathlib import Path

# ===== GLOBAL CONFIGURATION =====
CODESMILE_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CODESMILE_DIR)
SSD_DIR = CODESMILE_DIR 
MAX_WORKERS = 5
PARALLEL = True
PYTHON_EXEC = 'python'

# Define the order of processing
PROJECT_SIZES = ['small', 'medium', 'large']

# ============================================================================
# MAIN LOOP OVER SIZES
# ============================================================================
for size in PROJECT_SIZES:
    print("\n" + "#" * 80)
    print(f"STARTING ANALYSIS FOR: {size.upper()} REPOSITORIES")
    print("#" * 80)

    # ===== DYNAMIC CONFIGURATION FOR CURRENT SIZE =====
    CSV_FILE = f'../repos_{size}.csv'  # e.g., ../repos_small.csv
    
    # Update directory paths based on size
    REPOS_DIR = os.path.join(SSD_DIR, f'{size}_repos')
    RESULTS_DIR = os.path.join(SSD_DIR, f'{size}_codesmile_results')
    ANALYSIS_DIR = os.path.join(SSD_DIR, f'{size}_codesmile_analysis')

    # Create directories for this size
    os.makedirs(REPOS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print(f"📂 Input: {CSV_FILE}")
    print(f"📂 Repos Dir: {REPOS_DIR}")
    print(f"📂 Results Dir: {RESULTS_DIR}")
    print(f"📂 Analysis Dir: {ANALYSIS_DIR}")

    # ============================================================================
    # Step 0: Read CSV
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"STEP 0: Reading CSV ({size})")
    print("=" * 80)

    if not os.path.exists(CSV_FILE):
        print(f"✗ {CSV_FILE} not found! Skipping {size} batch...")
        continue

    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✓ Loaded {len(df)} repositories")
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        continue

    if not df.empty:
        print(f"\nFirst 3 repos in {size}:")
        print(df[['GitHub Repo', 'Stars', 'Lines of Code', 'Project Size']].head(3))
    else:
        print("⚠ CSV is empty")
        continue

    # ============================================================================
    # Step 1: Clone Repositories
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"STEP 1: Cloning Repositories ({size})")
    print("=" * 80)

    def extract_repo_info(github_repo):
        if pd.isna(github_repo):
            return None, None
        repo_str = str(github_repo).replace('https://github.com/', '').strip('/')
        parts = repo_str.split('/')
        if len(parts) >= 2:
            return parts[0], parts[1]
        return None, None

    def clone_repo(github_repo, idx, total, repos_dir):
        owner, repo = extract_repo_info(github_repo)
        if not owner or not repo:
            return None, 'invalid', owner, repo
        
        local = os.path.join(repos_dir, f"{idx:03d}_{owner}_{repo}")
        if os.path.exists(local):
            print(f"  ✓ [{idx}/{total}] Exists: {owner}/{repo}")
            return local, 'existing', owner, repo
        
        try:
            print(f"  → [{idx}/{total}] Cloning: {owner}/{repo}")
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', f"https://github.com/{owner}/{repo}.git", local],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=300,
                text=True
            )
            if result.returncode == 0:
                print(f"  ✓ [{idx}/{total}] Done")
                return local, 'cloned', owner, repo
            return None, 'failed', owner, repo
        except subprocess.TimeoutExpired:
            return None, 'timeout', owner, repo
        except Exception:
            return None, 'error', owner, repo

    repo_status = []
    for idx, row in df.iterrows():
        # Pass the dynamic REPOS_DIR to the clone function
        local, status, owner, repo = clone_repo(row['GitHub Repo'], idx + 1, len(df), REPOS_DIR)
        repo_status.append({
            'index': idx,
            'github_repo': row['GitHub Repo'],
            'owner': owner,
            'repo_name': repo,
            'stars': row['Stars'],
            'lines_of_code': row['Lines of Code'],
            'local_path': local,
            'clone_status': status,
            'project_size': row['Project Size'],
            'domain': row.get('domain', '')
        })

    clone_df = pd.DataFrame(repo_status)
    clone_df.to_csv(os.path.join(ANALYSIS_DIR, 'clone_status.csv'), index=False)

    successful = clone_df[clone_df['clone_status'].isin(['cloned', 'existing'])]
    print(f"\n✓ Prepared {len(successful)}/{len(df)} repos for {size}")

    # ============================================================================
    # Step 2: Run CodeSmile
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"STEP 2: CodeSmile Analysis ({size})")
    print("=" * 80)

    analysis_file = os.path.join(ANALYSIS_DIR, 'analysis_status.csv')

    # RESUME: Load existing results for this size
    if os.path.exists(analysis_file):
        print(f"\n📂 Resuming {size} from: {analysis_file}")
        existing = pd.read_csv(analysis_file)
        already_done = set(existing['index'].values)
        print(f"   Already analyzed: {len(already_done)} repos")
        analysis_results = existing.to_dict('records')
    else:
        already_done = set()
        analysis_results = []

    def save_progress(results, filepath):
        """Save after each repo"""
        pd.DataFrame(results).to_csv(filepath, index=False)

    for idx, row in successful.iterrows():
        # Skip if already done
        if row['index'] in already_done:
            print(f"\n⏭ [{idx + 1}/{len(successful)}] Skipping: {row['github_repo']}")
            continue
        
        repo_name = f"{row['owner']}_{row['repo_name']}"
        local_path = row['local_path']
        output = os.path.join(RESULTS_DIR, f"{row['index']:03d}_{repo_name}")
        
        print(f"\n[{idx + 1}/{len(successful)}] {row['github_repo']}")
        
        # Build command
        cmd = [PYTHON_EXEC, '-m', 'cli.cli_runner', '--input', os.path.abspath(local_path), '--output', os.path.abspath(output)]
        if PARALLEL:
            cmd.extend(['--parallel', '--max_walkers', str(MAX_WORKERS)])
        
        start = time.time()
        
        try:
            # Run from smell_ai directory
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=600, 
                cwd=CODESMILE_DIR
            )
            elapsed = time.time() - start
            
            analysis_results.append({
                'index': row['index'],
                'github_repo': row['github_repo'],
                'repo_name': repo_name,
                'stars': row['stars'],
                'lines_of_code': row['lines_of_code'],
                'project_size': row['project_size'],
                'domain': row['domain'],
                'status': 'success' if result.returncode == 0 else 'failed',
                'elapsed_time': elapsed,
                'output_path': output
            })
            
            print(f"  {'✓' if result.returncode == 0 else '✗'} {elapsed:.1f}s")
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  ✗ Timeout {elapsed:.1f}s")
            analysis_results.append({
                'index': row['index'],
                'github_repo': row['github_repo'],
                'repo_name': repo_name,
                'stars': row['stars'],
                'lines_of_code': row['lines_of_code'],
                'project_size': row['project_size'],
                'domain': row['domain'],
                'status': 'timeout',
                'elapsed_time': elapsed,
                'output_path': output
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            analysis_results.append({
                'index': row['index'],
                'github_repo': row['github_repo'],
                'repo_name': repo_name,
                'stars': row['stars'],
                'lines_of_code': row['lines_of_code'],
                'project_size': row['project_size'],
                'domain': row['domain'],
                'status': 'error',
                'elapsed_time': 0,
                'output_path': output
            })
        
        # SAVE PROGRESS
        save_progress(analysis_results, analysis_file)
        print(f"Saved")

    # Final save for this size
    save_progress(analysis_results, analysis_file)

    results_df = pd.DataFrame(analysis_results)
    print(f"\n✓ Final {size} results: {analysis_file}")

    # ============================================================================
    # Summary for this size
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"SUMMARY FOR {size.upper()}")
    print("=" * 80)

    success_count = len(results_df[results_df['status'] == 'success'])
    print(f"\nRepositories: {len(df)}")
    print(f"Success: {success_count}")
    if len(successful) > 0:
        print(f"Success rate: {success_count/len(successful)*100:.1f}%")

    if success_count > 0:
        avg = results_df[results_df['status'] == 'success']['elapsed_time'].mean()
        total_time = results_df['elapsed_time'].sum()
        print(f"\nAvg time: {avg:.1f}s")
        print(f"Total: {total_time/60:.1f}m")

    print(f"\n📁 {ANALYSIS_DIR}/analysis_status.csv")
    print(f"📁 {RESULTS_DIR}/")
    print("-" * 80)

print("\n" + "=" * 80)
print("ALL BATCHES (SMALL -> MEDIUM -> LARGE) COMPLETE")
print("=" * 80)