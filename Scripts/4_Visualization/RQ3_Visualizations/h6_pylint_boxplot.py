import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('pylint_comprehensive_data.csv')

domain_mapping = {
    'Natural Language Processing': 'NLP',
    'Reinforcement Learning': 'RL',
    'Computer Vision': 'CV',
    'MLOps / Deployment': 'MLOps',
    'ML Frameworks / Libraries': 'Frameworks',
    'Data Science / Traditional ML': 'Data Sci',
    'Domain-Specific Applications': 'Apps'
}
df['domain_short'] = df['domain'].map(domain_mapping)

order = df.groupby('domain_short')['smell_density'].median().sort_values().index

plt.figure(figsize=(5, 3))
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.1)

ax = sns.boxplot(
    x='domain_short', 
    y='smell_density', 
    data=df, 
    order=order,
    palette="muted", 
    linewidth=1.2,
    fliersize=2
)

plt.xlabel("Domain", fontsize=10)
plt.ylabel("Smell Density / KLOC", fontsize=10)

sns.despine()

plt.tight_layout(pad=0.2)
plt.savefig('h6_boxplot_muted.png', dpi=300, bbox_inches='tight')

plt.close()

print("Successfully saved to h6_boxplot_muted.png")