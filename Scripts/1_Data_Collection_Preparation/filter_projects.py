#!/usr/bin/env python3
"""
GROUPING PROJECTS STEP 1 & 2

FILTERING BASED ON:
- recent commits (since 2024)
- minimum stars (100+)
- minimum commits (100+)
- minimum contributors (5+)
- minimum branches (2+)


# To use the GitHub API, set your token in the terminal before running:
# export GITHUB_TOKEN="your_token_here"
"""

import pandas as pd
import requests
import time
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import math

# logger initialization
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class RepositoryActivityFilter:
    def __init__(self, github_token=None):
        self.github_token = github_token
        self.session = requests.Session()
        
        if github_token:
            self.session.headers.update({
                'Authorization': 'token {}'.format(github_token),
                'Accept': 'application/vnd.github.v3+json'
            })
    def load_dataset(self, csv_file):
        df = pd.read_csv(csv_file)
        logger.info(f"loaded {len(df)} repositories from {csv_file}")
        return df

    def filter_by_activity(self, df, min_stars=100, min_commits=100, 
                                   since_date='2024-01-01'):
        """
        filters by:
        - recent commits (since 2024)
        - minimum stars (100+)
        - minimum commits (100+)
        """
        logger.info("\nfiltering by activity metrics...")
        
        original_count = len(df)
        
        df['Last Commit Date'] = pd.to_datetime(df['Last Commit Date'], errors='coerce')
        recent_commits = df['Last Commit Date'] >= since_date
        df = df[recent_commits]
        logger.info(f"  after recency filter (>={since_date}): {len(df)} repositories")
        
        df = df[df['Stars'] >= min_stars]
        logger.info("  after popularity filter (>={} stars): {} repositories".format(min_stars,len(df)))
        
        df = df[df['Commits'] >= min_commits]
        logger.info(f"  after activity filter (>={min_commits} commits): {len(df)} repositories")
        
        logger.info(f"removed {original_count - len(df)} repositories in initial filtering")
        return df

    
    def _get_paginated_count(self, url, max_pages=50):
        """gets paginated count from a github api endpoint for counting branches & contributors"""
        total_count = 0
        page = 1
        
        while page <= max_pages:
            paginated_url = "{}?per_page=100&page={}".format(url,page)
            
            try:
                resp = self.session.get(paginated_url, timeout=30)
                
                if resp.status_code != 200:
                    if page == 1:
                        return None
                    break
                
                items = resp.json()
                
                if not isinstance(items, list):
                    if page == 1:
                        return None
                    break
                
                if not items:
                    break
                
                items_in_page = len(items)
                total_count += items_in_page
                
                if items_in_page < 100:
                    break
                
                page += 1
                
                if page <= max_pages:
                    time.sleep(0.5)
                
            except Exception:
                if page == 1:
                    return None
                break
        
        return total_count


    def _get_commit_authors_graphql(self, repo_name):
        """
        stops after 500+ for 
        # GraphQL for commit history -> it was the most accurate one / pagination failed mostly after 300+

        """
        try:
            owner,repo = repo_name.split('/')
            query = """
            query($owner: String!, $repo: String!, $cursor: String) {
              repository(owner: $owner, name: $repo) {
                defaultBranchRef {
                  target {
                    ... on Commit {
                      history(first: 100, after: $cursor) {
                        nodes {
                          author {
                            user {
                              login
                            }
                          }
                        }
                        pageInfo {
                          hasNextPage
                          endCursor
                        }
                      }
                    }
                  }
                }
              }
            }
            """
            graphql_url = "https://api.github.com/graphql"
            unique_logins = set()
            cursor = None
            page = 0
            max_pages = 200
            contributor_threshold = 500
            
            while page < max_pages:
                variables = {'owner': owner, 'repo': repo,'cursor': cursor}
                response = self.session.post(
                    graphql_url,
                    json={'query': query, 'variables': variables},
                    timeout=30
                )
                
                if response.status_code != 200:
                    if page == 0:
                        logger.debug(f"    graphql request failed: {response.status_code}")
                        return None
                    break
                
                data = response.json()
                
                if 'errors' in data:
                    if page == 0:
                        logger.debug("    graphql error: {}".format(data['errors'][0].get('message', 'unknown')))
                        return None
                    break
                
                if not data.get('data', {}).get('repository',{}).get('defaultBranchRef'):
                    return None
                
                history = data['data']['repository']['defaultBranchRef']['target']['history']
                commits = history['nodes']
                page_info = history['pageInfo']
                
                #  unique logins -> same user may have multiple commits /two mails & github accounts
                for commit in commits:
                    author = commit.get('author', {})
                    if author:
                        user = author.get('user',{})
                        if user and user.get('login'):
                            unique_logins.add(user['login'])
                # threshold reached -> exit!
                if len(unique_logins) > contributor_threshold:
                    logger.debug(f"    reached {contributor_threshold}+ contributors, stopping count")
                    return contributor_threshold
                
                page += 1
                
                if page == 1:
                    logger.debug(f"    scanning commits (page {page})...", )
                elif page % 5 == 0:
                    logger.debug(f"    page {page} ({len(unique_logins)} unique contributors)...")
                
                if not page_info['hasNextPage']:
                    break
                
                cursor = page_info['endCursor']
                time.sleep(0.3)
            
            count = len(unique_logins)
            
            if count > 0:
                if page >= max_pages:
                    logger.warning(f"    reached pagination limit ({max_pages} pages), count may be underestimated")
                return count
            
            return None
            
        except Exception as e:
            logger.debug(f"    graphql scan error: {e}")
            return None


    def get_contributor_count_graphql(self, repo_name):
        contributor_count = self._get_commit_authors_graphql(repo_name)
        
        if contributor_count is None:
            logger.debug(f"  could not retrieve contributor data for {repo_name}")
        
        return contributor_count

    

    def get_repo_details(self, repo_name):
        """Grab branch count and contributor count for a repo"""
        try:
            #  branch counts -> with pagination
            branches_url = f"https://api.github.com/repos/{repo_name}/branches"
            branch_count = self._get_paginated_count(branches_url, max_pages=50)
            
            if branch_count is None:
                logger.debug(f"  failed to retrieve branches for {repo_name}")
                return None,None
            
            #  count -> via graphql
            contributor_count = self.get_contributor_count_graphql(repo_name)
            if contributor_count is None:
                logger.debug(f"  failed to retrieve contributors for {repo_name}")
                return branch_count, None
            return branch_count, contributor_count
            
        except Exception as e:
            logger.debug("  ERROR! retrieving details for {}: {}".format(repo_name, e))
            return None, None

    def filter_by_teamwork(self, df, min_contributors=5, 
                                       min_branches=2):
        """
        Filter by:
        **min 5 contributors and 2 branches**
        """
        logger.info(f"\ngathering collaboration data for {len(df)} repositories...")
        logger.info("using graphql commit scanning for contributor counts")
        filtered_repos = []
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            repo_name = row['GitHub Repo']
            logger.info(f"[{i}/{len(df)}] analyzing {repo_name}")
            
            branch_count,contributor_count = self.get_repo_details(repo_name)
            
            repo_data = row.to_dict()
            repo_data['Branch Count'] = branch_count
            repo_data['Contributor Count'] = contributor_count
            
            meets_criteria = True
            if branch_count is None or branch_count < min_branches:
                meets_criteria = False
                reason = f"insufficient branches ({branch_count})"
            elif contributor_count is None or contributor_count < min_contributors:
                meets_criteria = False
                reason = "insufficient contributors ({})".format(contributor_count)
            
            if meets_criteria:
                filtered_repos.append(repo_data)
                contributor_display = f"{contributor_count}+" if contributor_count == 500 else str(contributor_count)
                logger.info(f"  included: {branch_count} branches, {contributor_display} contributors")
            else:
                logger.info(f"  excluded: {reason}")
            time.sleep(1.5)
        
        return pd.DataFrame(filtered_repos)


    def show_summary(self, original_df, filtered_df):
        logger.info(f"\n" + "="*50)
        logger.info("filtering summary")
        logger.info("="*50)
        
        logger.info(f"original dataset: {len(original_df)} repositories")
        logger.info(f"filtered dataset: {len(filtered_df)} repositories")
        logger.info(f"retention rate: {len(filtered_df)/len(original_df)*100:.1f}%")
        
        # Stats for filtered dataset
        logger.info(f"\nfiltered dataset characteristics:")
        logger.info("  stars: {:,} to {:,}".format(filtered_df['Stars'].min(), filtered_df['Stars'].max()))
        logger.info(f"  commits: {filtered_df['Commits'].min():,} to {filtered_df['Commits'].max():,}")
        
        if 'Branch Count' in filtered_df.columns and not filtered_df['Branch Count'].isna().all():
            logger.info(f"  branches: {int(filtered_df['Branch Count'].min())} to {int(filtered_df['Branch Count'].max())}")
        
        if 'Contributor Count' in filtered_df.columns and not filtered_df['Contributor Count'].isna().all():
            min_contributors = int(filtered_df['Contributor Count'].min())
            max_contributors = int(filtered_df['Contributor Count'].max())
            max_display = f"{max_contributors}+" if max_contributors == 500 else str(max_contributors)
            logger.info(f"  contributors: {min_contributors} to {max_display}")
        
        if len(filtered_df) > 0:
            logger.info(f"\nsample repositories meeting criteria:")
            sample_repos = filtered_df.head(5)['GitHub Repo'].tolist()
            for i,repo in enumerate(sample_repos, 1):
                logger.info(f"  {i}. {repo}")
        
        return {
            'original_count': len(original_df),
            'filtered_count': len(filtered_df),
            'retention_rate': len(filtered_df)/len(original_df)*100
        }

    
    def _plot_size_comparison(self, ax, original_df,filtered_df):
        """plots dataset size comparison"""
        datasets = ['original dataset', 'filtered dataset']
        counts = [len(original_df), len(filtered_df)]
        colors = ['#3498db','#e74c3c']
        
        bars = ax.bar(datasets, counts, color=colors, alpha=0.7)
        ax.set_title('dataset size comparison')
        ax.set_ylabel('number of repositories')
        
        for bar, count in zip(bars,counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   str(count), ha='center', va='bottom',fontweight='bold')

    
    def _plot_stars_dist(self, ax, original_df, filtered_df):
        """plots stars distribution"""
        ax.hist(original_df['Stars'], bins=30, alpha=0.6, 
               label='original',color='#3498db')
        ax.hist(filtered_df['Stars'], bins=30, alpha=0.6, 
               label='filtered', color='#e74c3c')
        ax.set_title('stars distribution')
        ax.set_xlabel('number of stars')
        ax.set_ylabel('frequency')
        ax.legend()
        ax.set_yscale('log')

    
    def _plot_commits_dist(self, ax, original_df, filtered_df):
        """plots commits distribution"""
        ax.hist(original_df['Commits'], bins=30, alpha=0.6, 
               label='original', color='#3498db')
        ax.hist(filtered_df['Commits'], bins=30, alpha=0.6, 
               label='filtered',color='#e74c3c')
        ax.set_title('commits distribution')
        ax.set_xlabel('number of commits')
        ax.set_ylabel('frequency')
        ax.legend()
        ax.set_yscale('log')

    
    def _plot_collab_scatter(self, ax, filtered_df):
        if not filtered_df.empty:
            ax.scatter(filtered_df['Branch Count'], 
                      filtered_df['Contributor Count'], 
                      alpha=0.6,color='#2ecc71')
            ax.set_title('collaboration metrics')
            ax.set_xlabel('number of branches')
            ax.set_ylabel('number of contributors')


    def make_comparison_plots(self, original_df, filtered_df, 
                                   output_dir="filtering_analysis"):
        
         #visualizations & plots

        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        self._plot_size_comparison(ax1, original_df, filtered_df)
        self._plot_stars_dist(ax2, original_df, filtered_df)
        self._plot_commits_dist(ax3, original_df, filtered_df)
        self._plot_collab_scatter(ax4, filtered_df)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/filtering_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        

        if not filtered_df.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # branch
            ax1.hist(filtered_df['Branch Count'], bins=20, alpha=0.7, color='#9b59b6')
            ax1.set_title('branch count distribution')
            ax1.set_xlabel('number of branches')
            ax1.set_ylabel('frequency')
            ax1.axvline(x=2, color='red', linestyle='--', label='minimum threshold')
            ax1.legend()
            
            # contributor 
            ax2.hist(filtered_df['Contributor Count'], bins=20, alpha=0.7,color='#f39c12')
            ax2.set_title('contributor count distribution')
            ax2.set_xlabel('number of contributors')
            ax2.set_ylabel('frequency')
            ax2.axvline(x=5, color='red', linestyle='--', label='minimum threshold')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/collaboration_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"visualizations saved to {output_dir}/")


    def save_top_repos(self, filtered_df, output_dir="filtering_analysis"):
        """
        Create summary table of top repos by composite score
        
        Ranks repositories based on weighted combination of stars,
        commits, and contributor count
        """
        if filtered_df.empty:
            return
        
        # Calculate composite score
        filtered_df['Composite Score'] = (
            filtered_df['Stars'] * 0.4 + 
            filtered_df['Commits'] * 0.3 + 
            filtered_df['Contributor Count'] * 30
        )
        
        top_repos = filtered_df.nlargest(20, 'Composite Score')[
            ['GitHub Repo', 'Stars', 'Commits', 'Branch Count', 
             'Contributor Count','Last Commit Date']
        ].copy()
        
        # Format date
        top_repos['Last Commit Date'] = pd.to_datetime(
            top_repos['Last Commit Date']
        ).dt.strftime('%Y-%m-%d')
        
        # save file
        top_repos.to_csv(f"{output_dir}/top_active_repos.csv", index=False)
        
        # Display
        logger.info(f"\ntop 20 active repositories:")
        logger.info("-" * 100)
        for _, row in top_repos.head(10).iterrows():
            contributor_display = "{}+".format(int(row['Contributor Count'])) if row['Contributor Count'] == 500 else str(int(row['Contributor Count']))
            logger.info(
                f"{row['GitHub Repo']:<40} | "
                f"{row['Stars']:>6} stars | "
                "{:>6} commits | ".format(row['Commits']) +
                f"{row['Branch Count']:>3} branches | "
                f"{contributor_display:>4} contributors"
            )
        
        logger.info(f"\nfull table saved to {output_dir}/top_active_repos.csv")



    def group_by_size(self, df):
        """
        Grouping repos by lines of code using percentile distribution,
        divides dataset into small, medium, and large categories based on
        30th and 60th percentiles of code size
        """
        logger.info(f"\n" + "="*50)
        logger.info("code size categorization")
        logger.info("="*50)
        
        # calculate percentiles
        loc_30th = df['Lines of Code'].quantile(0.30)
        loc_60th = df['Lines of Code'].quantile(0.60)
        logger.info(f"lines of code percentiles:")
        logger.info("  30th percentile: {:,.0f} lines".format(loc_30th))
        logger.info(f"  60th percentile: {loc_60th:,.0f} lines")
        
        # grouping
        small_mask = df['Lines of Code'] < loc_30th
        medium_mask = (df['Lines of Code'] >= loc_30th) & (df['Lines of Code'] < loc_60th)
        large_mask = df['Lines of Code'] >= loc_60th
        small_group = df[small_mask].copy()
        medium_group = df[medium_mask].copy()
        large_group = df[large_mask].copy()
        small_group['Size Group'] = 'Small'
        medium_group['Size Group'] = 'Medium'
        large_group['Size Group'] = 'Large'
        #logging
        logger.info(f"\ngroup distribution:")
        logger.info(f"  small projects: {len(small_group)} (< {loc_30th:,.0f} lines)")
        logger.info("  medium projects: {} ({:,.0f} - {:,.0f} lines)".format(len(medium_group), loc_30th, loc_60th))
        logger.info(f"  large projects: {len(large_group)} (> {loc_60th:,.0f} lines)")
        
        return {
            'small': small_group,
            'medium': medium_group, 
            'large': large_group,
            'thresholds': {'30th': loc_30th, '60th': loc_60th}
        }

    def save_groups(self, groups, output_dir="filtering_analysis"):
        """Save categorized datasets to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for group_name, group_df in groups.items():
            if group_name == 'thresholds':
                continue
            if len(group_df) > 0:
                filename = "{}/{}_group.csv".format(output_dir, group_name)
                group_df.to_csv(filename, index=False)
                logger.info(f"saved {group_name} group ({len(group_df)} repos) to {filename}")
        
        # save combined dataset
        all_groups = []
        for group_name,group_df in groups.items():
            if group_name != 'thresholds' and len(group_df) > 0:
                all_groups.append(group_df)
        if all_groups:
            combined = pd.concat(all_groups, ignore_index=True)
            combined.to_csv("niche_categorized_full.csv", index=False)
            logger.info(f"saved combined dataset to niche_categorized_full.csv")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='filter repository dataset for active, collaborative projects'
    )
    parser.add_argument('--input', default='niche_last.csv',
                        help='input csv file (default: niche_last.csv)')
    parser.add_argument('--min-stars', type=int, default=100,
                        help='minimum stars threshold (default: 100)')
    parser.add_argument('--min-commits',type=int, default=100,
                        help='minimum commits threshold (default: 100)')
    parser.add_argument('--min-contributors', type=int, default=5,
                        help='minimum contributors threshold (default: 5)')
    parser.add_argument('--min-branches', type=int,default=2,
                        help='minimum branches threshold (default: 2)')
    
    args = parser.parse_args()
    
    logger.info("repository activity and collaboration filter")
    logger.info("=" * 50)
    logger.info("methodology: graphql commit history scanning (single method)")
    logger.info("")
    
    # checks github token
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        logger.error("error: GITHUB_TOKEN environment variable is required")
        logger.error("graphql api requires authentication")
        logger.error("set with: export GITHUB_TOKEN='your_token_here'")
        return
    
    filter_tool = RepositoryActivityFilter(github_token)
    
    # load dataset
    try:
        df = filter_tool.load_dataset(args.input)
    except FileNotFoundError:
        logger.error(f"error: {args.input} not found")
        return
    
    #  filtering (stars, commits, latest update date)
    basic_filtered = filter_tool.filter_by_activity(
        df,
        min_stars=args.min_stars,
        min_commits=args.min_commits
    )
    
    if len(basic_filtered) == 0:
        logger.info("no repositories passed activity filters")
        return
    
    # asks for confirmation
    logger.info(f"\nthis will scan commit history for {len(basic_filtered)} repositories")
    logger.info(f"estimated time: {len(basic_filtered) * 5} seconds minimum")
    
    proceed = input("continue? (y/n): ").strip().lower()
    
    if proceed not in ['y', 'yes']:
        logger.info("operation cancelled")
        return
    
    # FILTERING 2) **** apply branches & contr filtering
    final_filtered = filter_tool.filter_by_teamwork(
        basic_filtered,
        min_contributors=args.min_contributors,
        min_branches=args.min_branches
    )
    
    if len(final_filtered) == 0:
        logger.info("no repositories passed collaboration filters")
        return
    
    # ---- Remove LOC = 0  PROJECTS----
    logger.info("removing repositories with 0 lines of code...")
    prev_len = len(final_filtered)
    final_filtered = final_filtered[final_filtered['Lines of Code'] != 0]
    if len(final_filtered) < prev_len:
        logger.info(f"removed {prev_len - len(final_filtered)} repositories with loc=0")
    # -----------------------------------

    stats = filter_tool.show_summary(df, final_filtered)
    size_groups = filter_tool.group_by_size(final_filtered)
    
    # plots and visualizations
    filter_tool.make_comparison_plots(df, final_filtered)
    filter_tool.save_top_repos(final_filtered)
    
    # save 
    final_filtered.to_csv('niche_active_repos.csv', index=False)
    filter_tool.save_groups(size_groups)
    
    logger.info(f"\nfiltering and categorization complete")
    logger.info(f"results saved:")
    logger.info(f"  - niche_active_repos.csv (all {len(final_filtered)} filtered repos)")
    logger.info(f"  - niche_categorized_full.csv (all repos with size categories)")
    logger.info(f"  - individual group files in filtering_analysis/ directory")


if __name__ == "__main__":
    main()