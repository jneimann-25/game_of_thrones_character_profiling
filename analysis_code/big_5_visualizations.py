import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import pi

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

def plot_radar_chart(df, character, save_dir):
    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    row = df[df['Character'] == character]
    if row.empty:
        return
    values = row[traits].values.flatten().tolist()
    values += values[:1]
    
    angles = [n / float(len(traits)) * 2 * pi for n in range(len(traits))]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], traits)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'skyblue', alpha=0.4)
    plt.title(f"{character} - Big Five Personality Traits", size=12)
    
    plt.savefig(f"{save_dir}/{character}_radar_chart.png")
    plt.close()

def plot_all_radars(df, save_dir):
    for character in df['Character']:
        plot_radar_chart(df, character, save_dir)

def plot_trait_bars(df, trait, save_dir):
    plt.figure(figsize=(12, 6))
    sorted_df = df.sort_values(trait, ascending=False)
    sns.barplot(data=sorted_df, x='Character', y=trait, palette="coolwarm")
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{trait} Scores by Character')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{trait}_bar_plot.png")
    plt.close()

def plot_all_trait_bars(df, save_dir):
    for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
        plot_trait_bars(df, trait, save_dir)

def plot_pairwise_traits(df, save_dir):
    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    sns.pairplot(df[traits])
    plt.suptitle("Big Five Trait Pairplot", y=1.02)
    plt.savefig(f"{save_dir}/big_five_pairplot.png")
    plt.close()

def plot_trait_heatmap(df, save_dir):
    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    trait_data = df.set_index("Character")[traits]
    sns.clustermap(trait_data, metric="euclidean", method="ward", cmap="coolwarm", figsize=(10, 10))
    plt.savefig(f"{save_dir}/big_five_heatmap.png")
    plt.close()

def main():
    df = pd.read_csv("GOT_Character_Big_Five_Traits.csv")
    save_dir = "big_5_vis"
    ensure_dir(save_dir)

    print("🔄 Generating radar charts...")
    plot_all_radars(df, save_dir)

    print("📊 Generating bar plots...")
    plot_all_trait_bars(df, save_dir)

    print("🔗 Generating trait pairplot...")
    plot_pairwise_traits(df, save_dir)

    print("🧱 Generating cluster heatmap...")
    plot_trait_heatmap(df, save_dir)

    print(f"✅ All visualizations saved to ./{save_dir}/")

if __name__ == "__main__":
    main()
