#!/usr/bin/env python3
"""
CI/CD Existence Checker
Scans GitHub repositories for common CI configuration files.
"""

import pandas as pd
import requests
import os
import time
import argparse

class CIFilter:
    def __init__(self, github_token=None, verbose=False):
        self.token = github_token
        self.headers = {'Authorization': f'token {github_token}'} if github_token else {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.api_calls = 0
        self.rate_limit_remaining = None
        self.verbose = verbose

    def get_contents(self, repo, path=""):
        """Fetch repo contents with rate limit handling"""
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        
        # Rate limit protection
        if self.rate_limit_remaining is not None and self.rate_limit_remaining < 10:
            print(f"    [!] Rate limit low ({self.rate_limit_remaining}), waiting 60s...")
            time.sleep(60)
            self.rate_limit_remaining = None
        
        try:
            resp = self.session.get(url, timeout=10)
            self.api_calls += 1
            
            if 'X-RateLimit-Remaining' in resp.headers:
                self.rate_limit_remaining = int(resp.headers['X-RateLimit-Remaining'])
            
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 403:
                # Check if it's rate limit or permission error
                if 'rate limit' in resp.text.lower():
                    print(f"    [!] Rate limited, waiting 60s...")
                    time.sleep(60)
                    return self.get_contents(repo, path)
                else:
                    return None # Likely 403 forbidden (private repo?)
            elif resp.status_code == 404:
                return None # Path doesn't exist
            else:
                return None
        except Exception as e:
            if self.verbose:
                print(f"    [!] API error: {e}")
            return None

    def check_ci_status(self, repo):
        """
        Checks for existence of various CI/CD configuration files.
        Returns a dictionary with status and detected tools.
        """
        print(f"    Scanning {repo}...")
        detected_tools = []
        
        # 1. Check Root Directory for standard config files
        root_items = self.get_contents(repo, "")
        
        if not root_items:
            return {'has_ci': False, 'tools': [], 'ci_evidence': 'Repo not accessible or empty'}

        root_files = [item['name'] for item in root_items if isinstance(item, dict)]

        # Direct file checks
        if '.travis.yml' in root_files: detected_tools.append('Travis CI')
        if '.gitlab-ci.yml' in root_files: detected_tools.append('GitLab CI')
        if 'Jenkinsfile' in root_files: detected_tools.append('Jenkins')
        if 'azure-pipelines.yml' in root_files: detected_tools.append('Azure Pipelines')
        if 'appveyor.yml' in root_files: detected_tools.append('AppVeyor')
        if 'bitbucket-pipelines.yml' in root_files: detected_tools.append('Bitbucket')
        if 'tox.ini' in root_files: detected_tools.append('Tox (Python Testing)')
        if 'Makefile' in root_files: detected_tools.append('Makefile (Possible CI)')
        
        # 2. Check CircleCI (Folder check)
        if '.circleci' in root_files:
            # We assume if the folder exists, they use it. 
            # Could go deeper to check config.yml if needed.
            detected_tools.append('CircleCI')

        # 3. Check GitHub Actions (Deep check)
        # Look for .github folder -> workflows folder -> .yml files
        if '.github' in root_files:
            github_contents = self.get_contents(repo, ".github")
            if github_contents:
                gh_names = [i['name'] for i in github_contents if isinstance(i, dict)]
                if 'workflows' in gh_names:
                    workflows = self.get_contents(repo, ".github/workflows")
                    if workflows:
                        # Check if any YAML files exist in workflows
                        has_workflow_files = any(
                            w['name'].endswith(('.yml', '.yaml')) 
                            for w in workflows if isinstance(w, dict)
                        )
                        if has_workflow_files:
                            detected_tools.append('GitHub Actions')

        # Summary
        has_ci = len(detected_tools) > 0
        evidence_str = ", ".join(detected_tools) if has_ci else "No CI config found"
        
        if self.verbose and has_ci:
            print(f"      [+] Found: {evidence_str}")

        return {
            'has_ci': has_ci,
            'tools': detected_tools,
            'ci_evidence': evidence_str
        }

def process_file(filepath, ci_checker):
    """Reads a CSV, checks CI for each repo, and saves the output."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print(f"{'='*60}")

    df = pd.read_csv(filepath)
    
    # Ensure GitHub Repo column exists
    if 'GitHub Repo' not in df.columns:
        print("Error: Column 'GitHub Repo' not found in CSV.")
        return

    results = []

    try:
        for idx, row in df.iterrows():
            repo = row['GitHub Repo']
            print(f"[{idx+1}/{len(df)}] Checking {repo}...")
            
            # Perform Check
            result = ci_checker.check_ci_status(repo)
            
            results.append({
                'GitHub Repo': repo,
                'CI_Exists_Checked': 'Yes' if result['has_ci'] else 'No',
                'CI_Tools_Detected': result['ci_evidence']
            })
            
            # Optional: Sleep to be nice to API
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving partial results...")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge back with original data (optional, or just save new columns)
    # Here we just join on index assuming order is preserved or use merge
    df_final = pd.merge(df, results_df, on='GitHub Repo', how='left')

    # Save output
    output_filename = filepath.replace('.csv', '_checked.csv')
    df_final.to_csv(output_filename, index=False)
    print(f"\nSaved results to: {output_filename}")
    print(f"Total API Calls: {ci_checker.api_calls}")

def main():
    parser = argparse.ArgumentParser(description='Check GitHub Repos for CI/CD existence.')
    parser.add_argument('--file', type=str, help='Path to a specific CSV file to check')
    parser.add_argument('--scan-all', action='store_true', help='Look for default files (small/medium/large)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs')
    
    args = parser.parse_args()
    
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print("\nWARNING: No GITHUB_TOKEN environment variable found.")
        print("API calls will be strictly limited (60/hour).")
        print("Export it using: export GITHUB_TOKEN='your_token_here'\n")
        time.sleep(2)

    ci_checker = CIFilter(github_token, verbose=args.verbose)

    if args.file:
        process_file(args.file, ci_checker)
    
    elif args.scan_all:
        # Define your default file names here
        default_files = [
            'repos_small.csv', 
            'repos_medium.csv', 
            'repos_large.csv'
    ]
        
        found_any = False
        for f in default_files:
            if os.path.exists(f):
                process_file(f, ci_checker)
                found_any = True
        
        if not found_any:
            print("No default CSV files found in current directory.")
            print(f" looked for: {default_files}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()