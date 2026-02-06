import pandas as pd
import subprocess
import os
import time
import json
from datetime import datetime
from pathlib import Path

# ===== GLOBAL CONFIGURATION =====
PYLINT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(PYLINT_DIR)
SSD_DIR = PYLINT_DIR
PYTHON_EXEC = 'python'

print(f"\n Working Directory: {PYLINT_DIR}")
print(f" Parent Directory: {PARENT_DIR}")
print(f" Python Executable: {PYTHON_EXEC}")

# Define the order of processing
PROJECT_SIZES = ['small', 'medium', 'large']

# ============================================================================
# MAIN LOOP OVER SIZES
# ============================================================================
for size in PROJECT_SIZES:
    print("\n" + "#" * 80)
    print(f"STARTING PYLINT ANALYSIS FOR: {size.upper()} REPOSITORIES")
    print("#" * 80)

    # ===== DYNAMIC CONFIGURATION FOR CURRENT SIZE =====
    CSV_FILE = f'repos_{size}.csv'
    
    # Update directory paths based on size
    REPOS_DIR = os.path.join(SSD_DIR, f'{size}_repos')
    RESULTS_DIR = os.path.join(SSD_DIR, f'{size}_pylint_results')
    ANALYSIS_DIR = os.path.join(SSD_DIR, f'{size}_pylint_analysis')

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
    # Step 1: Reuse Existing Repository Clones
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"STEP 1: Reusing Existing Repository Clones ({size})")
    print("=" * 80)

    def extract_repo_info(github_repo):
        if pd.isna(github_repo):
            return None, None
        repo_str = str(github_repo).replace('https://github.com/', '').strip('/')
        parts = repo_str.split('/')
        if len(parts) >= 2:
            return parts[0], parts[1]
        return None, None

    def find_existing_repo(github_repo, idx):
        """Find existing repository clone without attempting to clone"""
        owner, repo = extract_repo_info(github_repo)
        if not owner or not repo:
            return None, 'invalid', owner, repo
        
        # Uses the current loop's REPOS_DIR variable automatically
        local = os.path.join(REPOS_DIR, f"{idx+1:03d}_{owner}_{repo}")
        if os.path.exists(local):
            return local, 'existing', owner, repo
        else:
            return None, 'not_found', owner, repo

    print("Scanning for existing repository clones...")
    repo_status = []
    found_count = 0

    for idx, row in df.iterrows():
        local, status, owner, repo = find_existing_repo(row['GitHub Repo'], idx)
        
        if status == 'existing':
            found_count += 1
            if found_count % 20 == 0 or found_count <= 5:
                print(f"  ✓ Found: {owner}/{repo}")
        
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
    clone_df.to_csv(os.path.join(ANALYSIS_DIR, 'clone_status_pylint.csv'), index=False)

    successful = clone_df[clone_df['clone_status'] == 'existing']
    missing = clone_df[clone_df['clone_status'] == 'not_found']

    print(f"\n✓ Found existing clones: {len(successful)}/{len(df)} repos")
    if len(missing) > 0:
        print(f"⚠ Missing clones: {len(missing)} repos")
        print("   (These were likely skipped in the original CodeSmile analysis)")
        print(f"   First few missing: {list(missing['github_repo'].head(3))}")

    if len(successful) == 0:
        print(f"✗ No existing repository clones found in {REPOS_DIR}!")
        print("✗ Make sure CodeSmile analysis has been run first.")
        continue  # Skip to next size instead of exiting

    # ============================================================================
    # Step 2: Run Pylint Analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"STEP 2: Pylint Analysis ({size})")
    print("=" * 80)

    analysis_file = os.path.join(ANALYSIS_DIR, 'pylint_analysis_status.csv')

    # RESUME: Load existing results
    if os.path.exists(analysis_file):
        print(f"\n📂 Resuming {size} from: {analysis_file}")
        existing = pd.read_csv(analysis_file)
        already_done = set(existing['index'].values)
        print(f"   Already analyzed: {len(already_done)} repos")
        analysis_results = existing.to_dict('records')
    else:
        already_done = set()
        analysis_results = []

    def save_progress(results):
        """Save after each repo"""
        pd.DataFrame(results).to_csv(analysis_file, index=False)

    def find_python_files(repo_path):
        """Find all Python files in the repository"""
        python_files = []
        for root, dirs, files in os.walk(repo_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files

    def run_pylint_on_repo(repo_path, output_dir):
        """Run pylint on all Python files in repository"""
        python_files = find_python_files(repo_path)
        
        if not python_files:
            return {
                'status': 'no_python_files',
                'total_files': 0,
                'issues_count': 0,
                'score': 0.0,
                'output_file': None
            }
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Output files
        json_output = os.path.join(output_dir, 'pylint_results.json')
        txt_output = os.path.join(output_dir, 'pylint_results.txt')
        
        all_issues = []
        analyzed_files = 0
        
        # Run pylint on each Python file
        for py_file in python_files:
            try:
                # Run pylint with JSON output format
                cmd = [PYTHON_EXEC, '-m', 'pylint', '--output-format=json', py_file]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                # Parse JSON output
                if result.stdout.strip():
                    try:
                        file_issues = json.loads(result.stdout)
                        for issue in file_issues:
                            issue['file_path'] = py_file
                            all_issues.append(issue)
                        analyzed_files += 1
                    except json.JSONDecodeError:
                        # If JSON parsing fails, just count the file as analyzed
                        analyzed_files += 1
                
            except subprocess.TimeoutExpired:
                print(f"    ⚠ Timeout analyzing: {py_file}")
                continue
            except Exception as e:
                print(f"    ⚠ Error analyzing {py_file}: {e}")
                continue
        
        # Save results
        with open(json_output, 'w') as f:
            json.dump(all_issues, f, indent=2)
        
        # Create summary
        summary = {
            'total_python_files': len(python_files),
            'analyzed_files': analyzed_files,
            'total_issues': len(all_issues),
            'issues_by_type': {},
            'issues_by_severity': {}
        }
        
        # Categorize issues
        for issue in all_issues:
            issue_type = issue.get('type', 'unknown')
            severity = issue.get('message-id', 'unknown')
            
            summary['issues_by_type'][issue_type] = summary['issues_by_type'].get(issue_type, 0) + 1
            summary['issues_by_severity'][severity] = summary['issues_by_severity'].get(severity, 0) + 1
        
        # Write text summary
        with open(txt_output, 'w') as f:
            f.write(f"Pylint Analysis Summary\n")
            f.write(f"======================\n\n")
            f.write(f"Repository: {os.path.basename(repo_path)}\n")
            f.write(f"Total Python Files: {summary['total_python_files']}\n")
            f.write(f"Analyzed Files: {summary['analyzed_files']}\n")
            f.write(f"Total Issues: {summary['total_issues']}\n\n")
            
            if summary['issues_by_type']:
                f.write("Issues by Type:\n")
                for issue_type, count in summary['issues_by_type'].items():
                    f.write(f"  {issue_type}: {count}\n")
            
            f.write(f"\nDetailed Issues:\n")
            f.write(f"================\n\n")
            for issue in all_issues:
                f.write(f"File: {issue.get('path', 'unknown')}\n")
                f.write(f"Line: {issue.get('line', 'unknown')}\n")
                f.write(f"Type: {issue.get('type', 'unknown')}\n")
                f.write(f"Message ID: {issue.get('message-id', 'unknown')}\n")
                f.write(f"Message: {issue.get('message', 'unknown')}\n")
                f.write(f"Symbol: {issue.get('symbol', 'unknown')}\n\n")
        
        return {
            'status': 'success',
            'total_files': len(python_files),
            'analyzed_files': analyzed_files,
            'issues_count': len(all_issues),
            'score': 10.0 - min(10.0, len(all_issues) / max(1, analyzed_files)),  # Simple scoring
            'output_file': json_output,
            'summary': summary
        }

    for idx, row in successful.iterrows():
        # Skip if already done
        if row['index'] in already_done:
            print(f"\n⏭ [{idx + 1}/{len(successful)}] Skipping: {row['github_repo']}")
            continue
        
        repo_name = f"{row['owner']}_{row['repo_name']}"
        local_path = row['local_path']
        output = os.path.join(RESULTS_DIR, f"{row['index']+1:03d}_{repo_name}")  # Use 1-based indexing
        
        print(f"\n[{idx + 1}/{len(successful)}] {row['github_repo']}")
        
        start = time.time()
        
        try:
            # Run pylint analysis
            pylint_result = run_pylint_on_repo(local_path, output)
            elapsed = time.time() - start
            
            analysis_results.append({
                'index': row['index'],
                'github_repo': row['github_repo'],
                'repo_name': repo_name,
                'stars': row['stars'],
                'lines_of_code': row['lines_of_code'],
                'project_size': row['project_size'],
                'domain': row['domain'],
                'status': pylint_result['status'],
                'total_python_files': pylint_result['total_files'],
                'analyzed_files': pylint_result['analyzed_files'],
                'issues_count': pylint_result['issues_count'],
                'pylint_score': pylint_result['score'],
                'elapsed_time': elapsed,
                'output_path': output
            })
            
            print(f"  {'✓' if pylint_result['status'] == 'success' else '◯'} {elapsed:.1f}s | Files: {pylint_result['analyzed_files']} | Issues: {pylint_result['issues_count']}")
            
        except Exception as e:
            elapsed = time.time() - start
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
                'total_python_files': 0,
                'analyzed_files': 0,
                'issues_count': 0,
                'pylint_score': 0.0,
                'elapsed_time': elapsed,
                'output_path': output
            })
        
        # SAVE PROGRESS
        save_progress(analysis_results)
        print(f"Saved")

    # Final save
    save_progress(analysis_results)

    results_df = pd.DataFrame(analysis_results)
    print(f"\n✓ Final {size} results: {analysis_file}")

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"SUMMARY FOR {size.upper()}")
    print("=" * 80)

    success = len(results_df[results_df['status'] == 'success'])
    no_python = len(results_df[results_df['status'] == 'no_python_files'])
    print(f"\nRepositories: {len(df)}")
    print(f"Success: {success}")
    print(f"No Python files: {no_python}")
    if len(successful) > 0:
        print(f"Success rate: {(success + no_python)/len(successful)*100:.1f}%")

    if success > 0:
        avg = results_df[results_df['status'] == 'success']['elapsed_time'].mean()
        total_time = results_df['elapsed_time'].sum()
        avg_issues = results_df[results_df['status'] == 'success']['issues_count'].mean()
        print(f"\nAvg time: {avg:.1f}s")
        print(f"Total: {total_time/60:.1f}m")
        print(f"Avg issues per repo: {avg_issues:.1f}")

    print(f"\n📁 {ANALYSIS_DIR}/pylint_analysis_status.csv")
    print(f"📁 {RESULTS_DIR}/")
    print(f"\n✅ {size.upper()} Progress saved after each repo - safe to interrupt!")
    print("-" * 80)

print("\n" + "=" * 80)
print("ALL BATCHES (SMALL -> MEDIUM -> LARGE) COMPLETE")
print("=" * 80)