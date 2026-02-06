#!/usr/bin/env python3
"""
NICHE Dataset Updater

Updates repository metrics including stars, commits, python LOC, and last commit dates.

# To use the GitHub API set your token in the terminal before running:
# export GITHUB_TOKEN="your_token_here"
"""

import pandas as pd
import requests
import subprocess
import tempfile
import shutil
import json
import time
import os
import re
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetUpdater:
    """updates NICHE.csv (in the same directory) dataset with current repository information"""
    
    def __init__(self, input_file="NICHE.csv", github_token=None, batch_size=50):
        self.input_file = input_file
        self.token = github_token
        self.batch_size = batch_size
        self.progress_file = "update_progress.json"
        self.temp_dir = tempfile.mkdtemp(prefix="niche_update_")
        
        # *****setup GITHUB API******
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({
                'Authorization': f'token {self.token}',
                'Accept': 'application/vnd.github.v3+json'
            })
        
        self.stats = {
            'processed': 0,
            'success': 0,
            'api_fail': 0,
            'clone_fail': 0,
            'loc_fail': 0,
            'deleted': 0,
            'rate_limited': 0,
            'failures': defaultdict(int)
        }
        self.metrics = {
            'times': [],
            'loc': [],
            'stars': [],
            'commits': []
        }
        
        self.load_data()
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_file)
            logger.info(f"loaded {len(self.df)} repositories from dataset")
            
            if 'Last Commit Date' not in self.df.columns:
                self.df['Last Commit Date'] = None
                
        except FileNotFoundError:
            raise FileNotFoundError(f"input file {self.input_file} not found")
    def save_progress(self, idx):
        data = {
            'timestamp': datetime.now().isoformat(),
            'index': idx,
            'total': len(self.df),
            'stats': dict(self.stats),
            'metrics': self.metrics
        }
        
        # need to convert defaultdict for json serialization
        data['stats']['failures'] = dict(self.stats['failures'])
        
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_progress(self):
        if not os.path.exists(self.progress_file):
            return None
            
        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"could not load progress file!: {e}")
            return None
    
    def fetch_repo_info(self, repo):
        try:
            url = f"https://api.github.com/repos/{repo}"
            resp = self.session.get(url, timeout=30)
            
            if resp.status_code == 404:
                self.stats['deleted'] += 1
                self.stats['failures']['deleted'] += 1
                return None
            elif resp.status_code == 403:
                self.stats['rate_limited'] += 1
                self.stats['failures']['rate_limited'] += 1
                logger.warning(f"rate limited when fetching {repo}")
                return None
            elif resp.status_code != 200:
                self.stats['api_fail'] += 1
                self.stats['failures'][f'api_{resp.status_code}'] += 1
                logger.error(f"api returned {resp.status_code} for {repo}")
                return None
            
            return resp.json()
            
        except requests.exceptions.Timeout:
            self.stats['api_fail'] += 1
            self.stats['failures']['timeout'] += 1
            logger.error(f"api timeout for {repo}")
            return None
        except Exception as e:
            self.stats['api_fail'] += 1
            self.stats['failures']['api_error'] += 1
            logger.error(f"api error for {repo}: {e}")
            return None
    
    def fetch_commits(self, repo):
        """Get commit count and latest commit date from repo"""
        try:
            url = f"https://api.github.com/repos/{repo}/commits"
            resp = self.session.get(url, params={'per_page': 1}, timeout=30)
            
            if resp.status_code != 200:
                return None, None
            
            commits = resp.json()
            if not commits:
                return None, None
            
            #latest commit date
            commit_date = commits[0]['commit']['committer']['date'][:10]
            
            # GET total count from pagination headers (for commit count)
            commit_count = None
            link = resp.headers.get('Link', '')
            if 'rel="last"' in link:
                match = re.search(r'[?&]page=(\d+)[^>]*>;\s*rel="last"', link)
                if match:
                    commit_count = int(match.group(1))
            
            return commit_count, commit_date
            
        except Exception as e:
            logger.error(f"error fetching commit info for {repo}: {e}")
            return None, None
    
    def clone_repo(self, clone_url, repo):
        """clone repository - 1)Tries shallow clone first for speed"""
        repo_dir = os.path.join(self.temp_dir, repo.replace('/', '_'))
        
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        
        # 1) Tries shallow clone first (faster)
        try:
            cmd = ['git', 'clone', '--depth', '1', '--quiet', clone_url, repo_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return repo_dir
            # if shallow fails->> try full clone
            logger.debug(f"shallow clone failed for {repo}, attempting full clone")
            cmd = ['git', 'clone', '--quiet', clone_url, repo_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return repo_dir
            else:
                self.stats['clone_fail'] += 1
                self.stats['failures']['clone_failed'] += 1
                logger.error(f"failed to clone {repo}")
                return None
           # TIMEOUT    
        except subprocess.TimeoutExpired:
            self.stats['clone_fail'] += 1
            self.stats['failures']['clone_timeout'] += 1
            logger.error(f"clone timeout for {repo}")
            return None
        except Exception as e:
            self.stats['clone_fail'] += 1
            self.stats['failures']['clone_error'] += 1
            logger.error(f"clone error for {repo}: {e}")
            return None
    
    def count_python_loc(self, repo_dir):
        """
        count python lines of code using cloc.
        ******falls back to manual counting if cloc is unavailable.
        """
        # try cloc first (more accurate)
        try:
            cmd = ['cloc', '--json', '--quiet', '--include-lang=Python', repo_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get('Python', {}).get('code', 0)
                
        except subprocess.TimeoutExpired:
            logger.warning(f"cloc timed out for {repo_dir}")
        except Exception as e:
            logger.warning(f"cloc failed for {repo_dir}: {e}")
        
        #  manual counting
        try:
            count = 0
            for root, dirs, files in os.walk(repo_dir):
                # NO build directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and 
                          d not in ['__pycache__', 'node_modules', 'venv', 'env']]
                
                for f in files:
                    if f.endswith('.py'):
                        path = os.path.join(root, f)
                        try:
                            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                                lines = file.readlines()
                                # count only non emptyand no comment lines
                                code = [l.strip() for l in lines 
                                       if l.strip() and not l.strip().startswith('#')]
                                count += len(code)
                        except Exception:
                            continue
            return count
        except Exception as e:
            self.stats['loc_fail'] += 1
            self.stats['failures']['loc_error'] += 1
            logger.error(f"manual loc count failed: {e}")
            return None
        

    
    def update_repo(self, idx, row):
        repo = row['GitHub Repo']
        start = time.time()
        logger.info(f"[{idx + 1}/{len(self.df)}] processing {repo}")
        
        info = self.fetch_repo_info(repo)
        if not info:
            return False
        # update  stars
        self.df.at[idx, 'Stars'] = info.get('stargazers_count', 0)
        self.metrics['stars'].append(info.get('stargazers_count', 0))
        # get commit info
        commits, last = self.fetch_commits(repo)
        if commits:
            self.df.at[idx, 'Commits'] = commits
            self.metrics['commits'].append(commits)
        if last:
            self.df.at[idx, 'Last Commit Date'] = last
        
        # clone and count loc
        url = info.get('clone_url')
        if url:
            repo_dir = self.clone_repo(url, repo)
            if repo_dir:
                loc = self.count_python_loc(repo_dir)
                if loc is not None:
                    self.df.at[idx, 'Lines of Code'] = loc
                    self.metrics['loc'].append(loc)
                try:                # cleanup cloned repo
                    shutil.rmtree(repo_dir)
                except Exception as e:
                    logger.warning(f"failed to cleanup {repo_dir}: {e}")
        
        elapsed = time.time() - start
        self.metrics['times'].append(elapsed)
        logger.info(f"  stars: {info.get('stargazers_count', 0)}, "
                   f"commits: {commits or 'n/a'}, "
                   f"loc: {loc if 'loc' in locals() else 'n/a'}")
        self.stats['success'] += 1
        return True
    
    def run(self, resume=False):
        start_idx = 0
        if resume:
            prog = self.load_progress()
            if prog:
                start_idx = prog.get('index', 0)
                self.stats.update(prog.get('stats', {}))
                self.metrics.update(prog.get('metrics', {}))
                self.stats['failures'] = defaultdict(int, 
                    self.stats.get('failures', {}))
                logger.info(f"resuming from repository {start_idx + 1}")
        remaining = len(self.df) - start_idx
        logger.info(f"processing {remaining} repositories (batch size: {self.batch_size})")
        
        try:
            for i in range(start_idx, len(self.df)):
                row = self.df.iloc[i]
                
                self.update_repo(i, row)
                self.stats['processed'] += 1
                
                # save progress periodically (every batch_size)
                if (i + 1) % self.batch_size == 0:
                    self.save_progress(i + 1)
                    logger.info(f"progress checkpoint: {i + 1}/{len(self.df)}")
                    time.sleep(5)  # pause between batches
                else:
                    time.sleep(1)  # rate limiting
            self.save_progress(len(self.df))
            return True
            
        except KeyboardInterrupt:
            logger.info("process interrupted by user")
            self.save_progress(i)
            return False
        except Exception as e:
            logger.error(f"processing failed: {e}")
            self.save_progress(i)
            return False
    
    def save_data(self, output="niche_updated.csv"):
        """save updated dataset to csv"""

        if 'Last Commit Date' in self.df.columns:
            initial_count = len(self.df)
            self.df = self.df[self.df['Last Commit Date'].notna() & (self.df['Last Commit Date'] != '')]
            
            dropped_count = initial_count - len(self.df)
            if dropped_count > 0:
                logger.info(f"Filtered out {dropped_count} repositories with missing Last Commit Date")

        if output == self.input_file:
            backup = self.input_file.replace('.csv', '_backup.csv')
            self.df.to_csv(backup, index=False)
            logger.info(f"backup created: {backup}")
        self.df.to_csv(output, index=False)
        logger.info(f"updated dataset saved to: {output}")

    
    def print_stats(self):
        """display processing statistics summary"""
        print("\nprocessing statistics:")
        print("=" * 50)
        print(f"total processed: {self.stats['processed']}")
        print(f"successful: {self.stats['success']}")
        print(f"api failures: {self.stats['api_fail']}")
        print(f"clone failures: {self.stats['clone_fail']}")
        print(f"loc failures: {self.stats['loc_fail']}")
        print(f"deleted repos: {self.stats['deleted']}")
        print(f"rate limited: {self.stats['rate_limited']}")
        
        if self.metrics['times']:
            avg = sum(self.metrics['times']) / len(self.metrics['times'])
            print(f"average processing time: {avg:.2f} seconds")
        if self.stats['failures']:
            print("\nfailure breakdown:")
            for reason, count in self.stats['failures'].items():
                print(f"  {reason}: {count}")
        
        # calculate success rate
        total = (self.stats['success'] + self.stats['api_fail'] + 
                self.stats['clone_fail'] + self.stats['deleted'])
        if total > 0:
            rate = (self.stats['success'] / total) * 100
            print(f"\noverall success rate: {rate:.1f}%")
    
    def create_plots(self, output_dir="charts"):
        """plots& visualization """
        os.makedirs(output_dir, exist_ok=True)
        
        sns.set_style("whitegrid")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        labels = ['successful', 'api failures', 'clone failures', 'deleted']
        sizes = [self.stats['success'], self.stats['api_fail'],
                self.stats['clone_fail'], self.stats['deleted']]
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('processing results')
        
        if self.stats['failures']:
            reasons = list(self.stats['failures'].keys())
            counts = list(self.stats['failures'].values())
            ax2.bar(range(len(reasons)), counts)
            ax2.set_title('failure reasons')
            ax2.set_xticks(range(len(reasons)))
            ax2.set_xticklabels(reasons, rotation=45, ha='right')
            ax2.set_ylabel('count')
        
        if self.metrics['stars']:
            ax3.hist(self.metrics['stars'], bins=50, alpha=0.7)
            ax3.set_title('stars distribution')
            ax3.set_xlabel('stars')
            ax3.set_ylabel('frequency')
            ax3.set_yscale('log')
        if self.metrics['loc']:
            ax4.hist(self.metrics['loc'], bins=50, alpha=0.7)
            ax4.set_title('python loc distribution')
            ax4.set_xlabel('lines of code')
            ax4.set_ylabel('frequency')
            ax4.set_yscale('log')


        plt.tight_layout()
        plt.savefig(f"{output_dir}/overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.metrics['times']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.hist(self.metrics['times'], bins=30, alpha=0.7)
            ax1.set_title('processing time distribution')
            ax1.set_xlabel('time (seconds)')
            ax1.set_ylabel('frequency')

            ax2.plot(self.metrics['times'])
            ax2.set_title('processing time trend')
            ax2.set_xlabel('repository index')
            ax2.set_ylabel('time (seconds)')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/times.png", dpi=300, bbox_inches='tight')
            plt.close()
        logger.info(f"charts saved to {output_dir}/")
    


    def cleanup(self):
        """delete  temporary files and directories"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
        except Exception as e:
            logger.warning(f"cleanup warning: {e}")


def main():
    import argparse
    # ******PARSING*********
    parser = argparse.ArgumentParser(description='niche dataset updater')
    parser.add_argument('--input', default='NICHE.csv', 
                       help='input csv file path')
    parser.add_argument('--output', default='niche_updated.csv',
                       help='output csv file path')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='batch size for progress saving')
    parser.add_argument('--resume', action='store_true',
                       help='resume from previous progress')
    parser.add_argument('--github-token',
                       help='github personal access token')
    parser.add_argument('--charts', action='store_true',
                       help='generate analysis charts')
    parser.add_argument('--no-token', action='store_true',
                       help='continue without token (not recommended)')
    args = parser.parse_args()
    
    print("*NICHE Dataset Updater*")
    print("=" * 40)
    
    # Checks for github token
    token = args.github_token or os.environ.get('GITHUB_TOKEN')
    if not token:
        if not args.no_token:
            print("error: no github token provided")
            print("this will hit rate limits very quickly.")
            print("either:")
            print("  1. Set GITHUB_TOKEN environment variable")
            print("  2. Use --github-token option")
            print("  3. Add --no-token to continue anyway (not recommended)")
            return 1
        else:
            logger.warning("Running without token - expect rate limits!")
    
    updater = DatasetUpdater(
        input_file=args.input,
        github_token=token,
        batch_size=args.batch_size
    )
    try:
        success = updater.run(resume=args.resume)
        
        if success:
            logger.info("update completed successfully")
            updater.save_data(args.output)
            updater.print_stats()
            if args.charts:
                updater.create_plots()
        else:
            logger.info("update incomplete - use --resume to continue")
            return 1
        
    except Exception as e:
        logger.error(f"fatal error: {e}")
        return 1
    finally:
        updater.cleanup()

    return 0
if __name__ == "__main__":
    exit(main())