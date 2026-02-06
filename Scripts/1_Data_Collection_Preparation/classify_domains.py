#!/usr/bin/env python3
"""
domain classification for github repos
improved version with better matching and manual review support

set your github token before running:
export GITHUB_TOKEN="your_token_here"
"""

import pandas as pd
import requests
import time
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns


# domain keywords - split into primary and secondary for weighting
DOMAINS = {
    'Natural Language Processing': {
        'primary': ['nlp', 'natural-language-processing', 'natural language processing',
                   'text-classification', 'text classification', 'sentiment-analysis', 
                   'sentiment analysis', 'named-entity', 'named entity', 'language-model', 
                   'language model'],
        'secondary': ['bert', 'gpt', 'transformer', 'tokenization', 'text-mining',
                     'text mining', 'chatbot', 'translation', 'question-answering',
                     'question answering', 'text-generation', 'text generation', 'llm']
    },
    
    'Computer Vision': {
        'primary': ['computer-vision', 'computer vision', 'image-processing', 
                   'image processing', 'object-detection', 'object detection',
                   'image-classification', 'image classification', 'segmentation', 
                   'face-recognition', 'face recognition'],
        'secondary': ['cnn', 'convolutional', 'opencv', 'yolo', 'pose-estimation',
                     'pose estimation', 'video-processing', 'video processing',
                     'rcnn', 'mask-rcnn', 'detection', 'vision']
    },
    
    'Reinforcement Learning': {
        'primary': ['reinforcement-learning', 'reinforcement learning', 'q-learning', 
                   'deep-q', 'dqn', 'policy-gradient', 'policy gradient', 'actor-critic'],
        'secondary': ['openai-gym', 'openai gym', 'gym', 'ppo', 'a3c', 'rl-agent', 
                     'rl agent', 'mdp', 'td-learning', 'sarsa', 'monte-carlo']
    },
    
    'Data Science / Traditional ML': {
        'primary': ['automl', 'auto-ml', 'feature-engineering', 'feature engineering',
                   'hyperparameter', 'scikit-learn', 'sklearn', 'xgboost', 'lightgbm', 
                   'catboost'],
        'secondary': ['data-science', 'data science', 'data-analysis', 'data analysis', 
                     'pandas', 'numpy', 'random-forest', 'random forest',
                     'gradient-boosting', 'gradient boosting', 'ensemble',
                     'cross-validation', 'model-selection', 'model selection', 
                     'optuna', 'hyperopt']
    },
    
    'MLOps / Deployment': {
        'primary': ['mlops', 'deployment', 'serving', 'mlflow', 'kubeflow',
                   'model-serving', 'model serving', 'production', 'ci-cd', 'ci/cd'],
        'secondary': ['docker', 'kubernetes', 'k8s', 'pipeline', 'airflow',
                     'monitoring', 'tracking', 'orchestration', 'inference-server',
                     'inference server']
    },
    
    'ML Frameworks / Libraries': {
        'primary': ['tensorflow', 'pytorch', 'keras', 'jax', 'flax', 'mxnet'],
        'secondary': ['caffe', 'theano', 'framework', 'deep-learning-framework',
                     'deep learning framework', 'neural-network-library', 
                     'neural network library', 'onnx', 'tvm', 'tensorrt']
    },
    
    'Domain-Specific Applications': {
        'primary': ['speech-recognition', 'speech recognition', 'asr', 
                   'text-to-speech', 'text to speech', 'time-series', 'time series',
                   'forecasting', 'recommendation', 'recommender', 'gnn',
                   'graph-neural', 'graph neural', 'medical-imaging', 'medical imaging',
                   'genomics', 'bioinformatics'],
        'secondary': ['audio', 'speech', 'voice', 'arima', 'prophet',
                     'collaborative-filtering', 'collaborative filtering',
                     'knowledge-graph', 'knowledge graph', 'healthcare',
                     'diagnosis', 'explainable', 'interpretability', 'multimodal']
    }
}

# these keywords in topics/description trigger immediate classification
# found that topics are more reliable than descriptions for these
PRIORITY_INDICATORS = {
    'Natural Language Processing': ['nlp', 'natural-language-processing', 'natural language processing',
                                    'language-model', 'language model',
                                    'text-classification', 'text classification', 
                                    'sentiment-analysis', 'sentiment analysis'],
    'Computer Vision': ['computer-vision', 'computer vision', 'image-processing', 
                       'image processing', 'object-detection', 'object detection',
                       'image-classification', 'image classification'],
    'Reinforcement Learning': ['reinforcement-learning', 'reinforcement learning', 
                              'q-learning', 'rl'],
    'MLOps / Deployment': ['mlops', 'kubeflow', 'mlflow', 'deployment'],
    'ML Frameworks / Libraries': ['tensorflow', 'pytorch', 'keras', 'jax'],
}


class GitHubClient:
    """handles github api requests with rate limiting"""
    
    def __init__(self, token=None):
        self.token = token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            print("error: no github token found")
            print("set it like this:")
            print("  export GITHUB_TOKEN='your_token_here'")
            sys.exit(1)
        
        self.headers = {'Authorization': f'token {self.token}'}
        self.base_url = 'https://api.github.com'
        self.request_count = 0
        self.rate_limit = None
    
    def get(self, endpoint, timeout=10):
        url = f'{self.base_url}{endpoint}'
        
        try:
            resp = requests.get(url, headers=self.headers, timeout=timeout)
            self.request_count += 1
            
            if 'X-RateLimit-Remaining' in resp.headers:
                self.rate_limit = int(resp.headers['X-RateLimit-Remaining'])
            
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                return None
            elif resp.status_code == 403:
                # hit rate limit, wait and retry
                print("\n    rate limit hit, waiting 60 sec...")
                time.sleep(60)
                return self.get(endpoint, timeout)
            else:
                return None
                
        except requests.exceptions.Timeout:
            return None
        except Exception:
            return None
    
    def get_readme(self, owner, repo):
        # get first 3000 chars of readme
        readme_info = self.get(f'/repos/{owner}/{repo}/readme')
        if readme_info and 'download_url' in readme_info:
            try:
                resp = requests.get(readme_info['download_url'], timeout=10)
                if resp.status_code == 200:
                    return resp.text[:3000]
            except:
                pass
        return ''


def classify_repo(repo_name, api_client):
    """
    classify a repo into a domain
    uses priority indicators first, then weighted scoring
    """
    
    try:
        owner, repo = repo_name.split('/')
    except:
        return {
            'domain': 'Error',
            'score': 0,
            'confidence': 'error',
            'method': 'error',
            'description': '',
            'topics': '',
            'language': ''
        }
    
    # get repo data from github
    repo_data = api_client.get(f'/repos/{owner}/{repo}')
    if not repo_data:
        return {
            'domain': 'API Error',
            'score': 0,
            'confidence': 'error',
            'method': 'api_error',
            'description': '',
            'topics': '',
            'language': ''
        }
    
    topics = repo_data.get('topics', [])
    desc = str(repo_data.get('description', '')).lower()
    lang = str(repo_data.get('language', '')).lower()
    readme = api_client.get_readme(owner, repo).lower()
    
    # check for strong domain indicators first
    # these give immediate classification with high confidence
    topics_lower = [t.lower() for t in topics]
    topics_str = ' '.join(topics_lower)
    
    for domain, indicators in PRIORITY_INDICATORS.items():
        for indicator in indicators:
            if indicator in topics_str or indicator in desc:
                # found a priority match
                primary_kw = DOMAINS[domain]['primary']
                secondary_kw = DOMAINS[domain]['secondary']
                full_text = f"{desc} {lang} {topics_str} {readme}"
                
                # calculate score with weighting
                score = 0
                score += sum(3 for kw in primary_kw if kw in topics_str)
                score += sum(2 for kw in secondary_kw if kw in topics_str)
                score += sum(2 for kw in primary_kw if kw in desc)
                score += sum(1 for kw in primary_kw if kw in full_text)
                
                return {
                    'domain': domain,
                    'score': max(score, 5),  # min score 5 for priority
                    'confidence': 'high',
                    'method': 'priority_indicator',
                    'description': repo_data.get('description', ''),
                    'topics': ', '.join(topics),
                    'language': repo_data.get('language', '')
                }
    
    # no priority match, do weighted keyword scoring
    full_text = f"{desc} {lang} {' '.join(topics_lower)} {readme}"
    
    domain_scores = {}
    for domain, keywords in DOMAINS.items():
        score = 0
        
        # topics get 3x weight
        score += sum(3 for kw in keywords['primary'] if kw in topics_str)
        score += sum(2 for kw in keywords['secondary'] if kw in topics_str)
        
        # description and readme
        score += sum(2 for kw in keywords['primary'] if kw in desc)
        score += sum(1 for kw in keywords['primary'] if kw in readme)
        score += sum(1 for kw in keywords['secondary'] if kw in full_text)
        
        domain_scores[domain] = score
    
    max_score = max(domain_scores.values())
    
    # apply minimum threshold
    # anything below 3 goes to manual review
    if max_score < 3:
        return {
            'domain': 'Unclassified',
            'score': max_score,
            'confidence': 'low',
            'method': 'below_threshold',
            'description': repo_data.get('description', ''),
            'topics': ', '.join(topics),
            'language': repo_data.get('language', '')
        }
    
    best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
    
    # figure out confidence level
    if max_score >= 10:
        confidence = 'very_high'
    elif max_score >= 6:
        confidence = 'high'
    elif max_score >= 4:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'domain': best_domain,
        'score': max_score,
        'confidence': confidence,
        'method': 'weighted_scoring',
        'description': repo_data.get('description', ''),
        'topics': ', '.join(topics),
        'language': repo_data.get('language', '')
    }


def classify_all_repos(df, api_client):
    """run classification on all repos in dataframe"""
    
    print(f"\nclassifying {len(df)} repositories...")
    print("="*70)
    
    results = []
    
    for idx, row in df.iterrows():
        repo_name = row['GitHub Repo']
        size = row.get('Size Group', 'Unknown')
        stars = row.get('Stars', 0)
        
        print(f"\n[{idx+1}/{len(df)}] {repo_name}")
        print(f"  size: {size}, stars: {stars}")
        
        result = classify_repo(repo_name, api_client)
        
        results.append({
            'repo': repo_name,
            'size_group': size,
            'stars': stars,
            'domain': result['domain'],
            'match_score': result['score'],
            'confidence': result['confidence'],
            'classification_method': result['method'],
            'description': result['description'],
            'topics': result['topics'],
            'language': result['language']
        })
        
        # show result
        if result['domain'] in ['Error', 'API Error']:
            print(f"  x {result['domain']}")
        elif result['domain'] == 'Unclassified':
            print(f"  ? unclassified (score {result['score']} too low)")
        else:
            # show confidence with stars
            conf_icon = {
                'very_high': '***',
                'high': '**',
                'medium': '*',
                'low': '.'
            }.get(result['confidence'], '.')
            
            method_label = {
                'priority_indicator': 'PRIORITY',
                'priority_topic': 'PRIORITY',
                'weighted_scoring': 'WEIGHTED'
            }.get(result['method'], 'STANDARD')
            
            print(f"  ok -> {result['domain']}")
            print(f"    {conf_icon} score:{result['score']} | {method_label} | {result['confidence']}")
        
        # show rate limit warning if getting low
        if api_client.rate_limit and api_client.rate_limit < 100:
            print(f"  [rate limit: {api_client.rate_limit} remaining]")
        
        time.sleep(1)  # be nice to github api
    
    return pd.DataFrame(results)


def print_summary(df):
    """print classification summary stats"""
    
    print("\n" + "="*70)
    print("classification summary")
    print("="*70)
    
    total = len(df)
    classified = len(df[~df['domain'].isin(['Unclassified', 'Error', 'API Error'])])
    unclassified = len(df[df['domain'] == 'Unclassified'])
    
    print(f"\ntotal repositories: {total}")
    print(f"successfully classified: {classified} ({classified/total*100:.1f}%)")
    print(f"unclassified (below threshold): {unclassified} ({unclassified/total*100:.1f}%)")
    print(f"number of domains: {len(DOMAINS)}")
    
    print("\n" + "-"*70)
    print("domain distribution")
    print("-"*70)
    
    domain_counts = df['domain'].value_counts()
    for domain, count in domain_counts.items():
        pct = (count / total) * 100
        avg_score = df[df['domain'] == domain]['match_score'].mean()
        
        # count high confidence
        domain_df = df[df['domain'] == domain]
        high_conf = len(domain_df[domain_df['confidence'].isin(['high', 'very_high'])])
        
        print(f"  {domain:40s}: {count:3d} ({pct:5.1f}%) | avg score: {avg_score:.1f} | high conf: {high_conf}")
    
    print("\n" + "-"*70)
    print("classification quality")
    print("-"*70)
    
    method_counts = df['classification_method'].value_counts()
    confidence_counts = df['confidence'].value_counts()
    
    print("\nby method:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} ({count/total*100:.1f}%)")
    
    print("\nby confidence:")
    for conf, count in confidence_counts.items():
        print(f"  {conf}: {count} ({count/total*100:.1f}%)")
    
    return df


def create_plots(df, output_file='domain_classification_improved.png'):
    """generate visualization plots"""
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    
    classified_df = df[~df['domain'].isin(['Unclassified', 'Error', 'API Error'])]
    
    # domain distribution bar chart
    ax1 = fig.add_subplot(gs[0, :])
    domain_counts = classified_df['domain'].value_counts()
    colors = plt.cm.Set2(range(len(domain_counts)))
    
    y_pos = range(len(domain_counts))
    ax1.barh(y_pos, domain_counts.values, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(domain_counts.index, fontsize=10)
    ax1.set_xlabel('number of repositories')
    ax1.set_title('domain classification results', fontweight='bold', fontsize=14)
    ax1.invert_yaxis()
    
    for i, v in enumerate(domain_counts.values):
        pct = v / len(classified_df) * 100
        ax1.text(v + 0.5, i, f'{v} ({pct:.1f}%)', va='center', fontweight='bold')
    

    # confidence distribution
    ax2 = fig.add_subplot(gs[1, 0])
    conf_order = ['very_high', 'high', 'medium', 'low', 'error']
    conf_counts = df['confidence'].value_counts().reindex(conf_order, fill_value=0)
    conf_colors = {'very_high': 'darkgreen', 'high': 'green', 'medium': 'orange', 
                   'low': 'red', 'error': 'gray'}
    colors_conf = [conf_colors.get(c, 'gray') for c in conf_counts.index]
    
    ax2.bar(range(len(conf_counts)), conf_counts.values, color=colors_conf, alpha=0.7)
    ax2.set_xticks(range(len(conf_counts)))
    ax2.set_xticklabels(conf_counts.index, rotation=45, ha='right')
    ax2.set_ylabel('number of projects')
    ax2.set_title('classification confidence', fontweight='bold')
    
    for i, v in enumerate(conf_counts.values):
        if v > 0:
            ax2.text(i, v + 1, str(v), ha='center', fontweight='bold')
    

    # method distribution pie
    ax3 = fig.add_subplot(gs[1, 1])
    method_counts = df['classification_method'].value_counts()
    ax3.pie(method_counts, labels=method_counts.index, autopct='%1.1f%%',
            colors=plt.cm.Pastel1(range(len(method_counts))), startangle=90)
    ax3.set_title('classification method', fontweight='bold')
    

    # score distribution by domain
    ax4 = fig.add_subplot(gs[2, :])
    score_by_domain = []
    labels = []
    for domain in domain_counts.index[:7]:
        scores = classified_df[classified_df['domain'] == domain]['match_score']
        if len(scores) > 0:
            score_by_domain.append(scores)
            labels.append(domain[:30])
    
    bp = ax4.boxplot(score_by_domain, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax4.set_ylabel('match score')
    ax4.set_title('score distribution by domain', fontweight='bold')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    

    # quality summary
    ax5 = fig.add_subplot(gs[3, :])
    
    high_quality = len(df[df['confidence'].isin(['high', 'very_high'])])
    medium_quality = len(df[df['confidence'] == 'medium'])
    low_quality = len(df[df['confidence'] == 'low'])
    unclassified = len(df[df['domain'] == 'Unclassified'])
    
    categories = ['high\nconfidence', 'medium\nconfidence', 'low\nconfidence', 'unclassified']
    values = [high_quality, medium_quality, low_quality, unclassified]
    bar_colors = ['darkgreen', 'orange', 'red', 'gray']
    
    bars = ax5.bar(categories, values, color=bar_colors, alpha=0.7)
    ax5.set_ylabel('number of projects')
    ax5.set_title('classification quality summary', fontweight='bold', fontsize=12)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}\n({val/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nvisualization saved: {output_file}")
    plt.close()


def save_results(df, df_original):
    """save classification results to files"""
    
    print("\n" + "="*70)
    print("saving results")
    print("="*70)
    
    # merge with original data
    df_merged = df_original.copy()
    df_merged['domain'] = df_merged['GitHub Repo'].map(df.set_index('repo')['domain'])
    df_merged['match_score'] = df_merged['GitHub Repo'].map(df.set_index('repo')['match_score'])
    df_merged['confidence'] = df_merged['GitHub Repo'].map(df.set_index('repo')['confidence'])
    df_merged['classification_method'] = df_merged['GitHub Repo'].map(df.set_index('repo')['classification_method'])
    df_merged['domain_description'] = df_merged['GitHub Repo'].map(df.set_index('repo')['description'])
    df_merged['domain_topics'] = df_merged['GitHub Repo'].map(df.set_index('repo')['topics'])
    
    df_merged.to_csv('repos_with_domains_improved.csv', index=False)
    print(f"saved: repos_with_domains_improved.csv (all {len(df_merged)} projects)")
    
    # save summary stats
    summary_data = []
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        summary_data.append({
            'Domain': domain,
            'Count': len(domain_df),
            'Percentage': len(domain_df) / len(df) * 100,
            'Avg_Score': domain_df['match_score'].mean(),
            'High_Confidence': len(domain_df[domain_df['confidence'].isin(['high', 'very_high'])]),
            'Medium_Confidence': len(domain_df[domain_df['confidence'] == 'medium']),
            'Low_Confidence': len(domain_df[domain_df['confidence'] == 'low'])
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Count', ascending=False)
    summary_df.to_csv('domain_summary_improved.csv', index=False)
    print("saved: domain_summary_improved.csv")
    
    # save by domain
    classified_df = df[~df['domain'].isin(['Unclassified', 'Error', 'API Error'])]
    for domain in classified_df['domain'].unique():
        domain_df = classified_df[classified_df['domain'] == domain]
        safe_name = domain.lower().replace('/', '').replace(' ', '_').replace('-', '_')
        filename = f'repos_{safe_name}_improved.csv'
        domain_df.to_csv(filename, index=False)
        print(f"saved: {filename} ({len(domain_df)} projects)")
    
    # save unclassified for manual review
    unclassified = df[df['domain'] == 'Unclassified']
    if len(unclassified) > 0:
        unclassified.to_csv('repos_unclassified_review.csv', index=False)
        print(f"saved: repos_unclassified_review.csv ({len(unclassified)} projects for manual review)")


def main():
    print("="*70)
    print("domain classification - improved version")
    print("="*70)
    print("\nimprovements:")
    print("  - priority indicators for high confidence matches")
    print("  - weighted scoring (topics 3x more important)")
    print("  - minimum threshold (score >= 3)")
    print("  - confidence tracking")
    print("  - classification method logging")
    
    input_file = 'niche_active_repos.csv'
    
    if not os.path.exists(input_file):
        print(f"\nerror: {input_file} not found")
        print("run the activity filter script first")
        sys.exit(1)
    
    print(f"\nloading repositories from {input_file}")
    df = pd.read_csv(input_file)
    print(f"found {len(df)} repositories")
    
    api_client = GitHubClient()
    print("github api client ready")
    print(f"using {len(DOMAINS)} domain categories")
    
    # run classification
    df_results = classify_all_repos(df, api_client)
    
    # show summary
    summary_df = print_summary(df_results)
    
    # create plots
    create_plots(df_results)
    
    # save everything
    save_results(df_results, df)
    
    print("\n" + "="*70)
    print("classification complete")
    print("="*70)
    print(f"total api requests: {api_client.request_count}")
    
    # show quality metrics
    high_conf = len(df_results[df_results['confidence'].isin(['high', 'very_high'])])
    total = len(df_results)
    print(f"\nquality metrics:")
    print(f"  high confidence: {high_conf}/{total} ({high_conf/total*100:.1f}%)")
    print(f"  classified into {len(DOMAINS)} domains")
    print("\nreview repos_unclassified_review.csv for manual classification")


if __name__ == "__main__":
    main()