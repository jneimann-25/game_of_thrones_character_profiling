import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the character profiles
profiles_df = pd.read_csv("Game_of_Thrones_Character_Profiles.csv")

# 1. Basic Character Comparison Plot
def plot_character_comparison(profiles_df, metric, top_n=10):
    """
    Create a horizontal bar chart comparing characters on a specific metric.
    
    Args:
        profiles_df: DataFrame with character profiles
        metric: Column name to compare
        top_n: Number of top characters to show
    """
    plt.figure(figsize=(12, 8))
    
    # Sort and get top N characters for this metric
    sorted_df = profiles_df.sort_values(by=metric, ascending=False).head(top_n)
    
    # Create the horizontal bar chart
    sns.barplot(x=sorted_df[metric], y=sorted_df["Character"], 
                palette="viridis", orient="h")
    
    # Add labels and title
    plt.title(f"Top {top_n} Characters by {metric}", fontsize=16)
    plt.xlabel(metric, fontsize=12)
    plt.ylabel("Character", fontsize=12)
    plt.tight_layout()
    
    return plt

# 2. Character Traits Spider Chart
def plot_character_spider(profiles_df, characters, metrics=None):
    """
    Create a spider/radar chart comparing multiple characters across metrics.
    
    Args:
        profiles_df: DataFrame with character profiles
        characters: List of character names to compare
        metrics: List of metrics to include (default: use all numeric columns)
    """
    if metrics is None:
        # Use all numeric columns except Character
        metrics = profiles_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter the dataframe
    char_df = profiles_df[profiles_df["Character"].isin(characters)]
    
    # Number of metrics (angles)
    N = len(metrics)
    
    # Create angle values (in radians)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    
    # Add metric labels
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Add radial labels (0-1 scale)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each character
    for i, character in enumerate(characters):
        char_data = char_df[char_df["Character"] == character]
        
        if len(char_data) == 0:
            continue
        
        # Get values and scale to 0-1
        values = []
        for metric in metrics:
            value = char_data[metric].values[0]
            min_val = profiles_df[metric].min()
            max_val = profiles_df[metric].max()
            # Scale to 0-1
            scaled_val = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
            values.append(scaled_val)
        
        # Complete the loop
        values += values[:1]
        
        # Plot the character line
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=character)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Character Trait Comparison", size=20, y=1.1)
    
    return plt

# 3. Character Similarity Map (PCA)
def plot_character_pca(profiles_df, color_by="Avg Sentiment"):
    """
    Create a 2D PCA plot showing character similarities based on all metrics.
    
    Args:
        profiles_df: DataFrame with character profiles
        color_by: Metric to use for coloring points
    """
    # Extract character names and numeric features
    characters = profiles_df["Character"].tolist()
    features = profiles_df.select_dtypes(include=[np.number])
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    # Create a dataframe with the PCA results
    pca_df = pd.DataFrame({
        "Character": characters,
        "PCA1": pca_result[:, 0],
        "PCA2": pca_result[:, 1],
        "ColorMetric": profiles_df[color_by]
    })
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(pca_df["PCA1"], pca_df["PCA2"], 
                         c=pca_df["ColorMetric"], cmap="coolwarm", 
                         s=100, alpha=0.7)
    
    # Add character labels
    for i, character in enumerate(pca_df["Character"]):
        plt.annotate(character, 
                    (pca_df["PCA1"].iloc[i], pca_df["PCA2"].iloc[i]),
                    fontsize=9, alpha=0.8)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_by, fontsize=12)
    
    # Add labels and title
    plt.title("Character Similarity Map", fontsize=16)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.tight_layout()
    
    return plt

# 4. Character Development Over Seasons
def plot_character_arcs(character_arcs_df, metric="Sentiment", characters=None, limit=6):
    """
    Plot how characters develop over seasons based on specified metric.
    
    Args:
        character_arcs_df: DataFrame with character data by season
        metric: Metric to track (e.g., "Sentiment", "Word Count")
        characters: List of characters to plot (default: top characters by dialogue)
        limit: Maximum number of characters to plot
    """
    if characters is None:
        # Get top characters by total dialogue volume
        dialogue_counts = character_arcs_df.groupby("Character")["Word Count"].sum()
        characters = dialogue_counts.sort_values(ascending=False).head(limit).index.tolist()
    else:
        characters = characters[:limit]  # Limit to specified max
    
    plt.figure(figsize=(12, 8))
    
    for character in characters:
        char_data = character_arcs_df[character_arcs_df["Character"] == character]
        if len(char_data) > 0:
            plt.plot(char_data["Season"], char_data[metric], 
                    marker="o", linewidth=2, label=character)
    
    plt.title(f"Character {metric} Development Over Seasons", fontsize=16)
    plt.xlabel("Season", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Force x-axis to be integers (seasons)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    return plt

# 5. Sentiment Distribution Violinplot
def plot_sentiment_distribution(dialogues_df):
    """
    Create a violin plot showing the distribution of sentiment for top characters.
    
    Args:
        dialogues_df: DataFrame with character dialogue including sentiment
    """
    # Calculate sentiment for each line
    if "Sentiment" not in dialogues_df.columns:
        from textblob import TextBlob
        dialogues_df["Sentiment"] = dialogues_df["Sentence"].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
    
    # Get top 10 characters by dialogue volume
    top_chars = dialogues_df.groupby("Name").size().sort_values(ascending=False).head(10).index
    char_sentiments = dialogues_df[dialogues_df["Name"].isin(top_chars)]
    
    plt.figure(figsize=(14, 10))
    sns.violinplot(x="Name", y="Sentiment", data=char_sentiments, 
                  palette="viridis", inner="quartile")
    
    plt.title("Sentiment Distribution by Character", fontsize=16)
    plt.xlabel("Character", fontsize=12)
    plt.ylabel("Sentiment Score", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return plt

# 6. Character Interaction Network
def plot_character_network(dialogues_df, min_interactions=5):
    """
    Create a network graph of character interactions.
    Requires networkx package.
    
    Args:
        dialogues_df: DataFrame with dialogue including Scene column
        min_interactions: Minimum number of shared scenes for an edge
    """
    import networkx as nx
    
    # First, create a scene-to-character mapping
    scene_chars = {}
    for scene, group in dialogues_df.groupby("Scene"):
        characters = group["Name"].unique().tolist()
        if len(characters) > 1:  # Only include scenes with multiple characters
            scene_chars[scene] = characters
    
    # Create a character interaction graph
    G = nx.Graph()
    
    # Add nodes (characters)
    for character in dialogues_df["Name"].unique():
        G.add_node(character)
    
    # Add edges (interactions)
    char_interactions = {}
    for scene, characters in scene_chars.items():
        for i in range(len(characters)):
            for j in range(i+1, len(characters)):
                char1, char2 = characters[i], characters[j]
                key = tuple(sorted([char1, char2]))
                char_interactions[key] = char_interactions.get(key, 0) + 1
    
    # Add edges with weight above threshold
    for (char1, char2), count in char_interactions.items():
        if count >= min_interactions:
            G.add_edge(char1, char2, weight=count)
    
    # Create the visualization
    plt.figure(figsize=(16, 12))
    
    # Compute node sizes based on dialogue amount
    dialogue_counts = dialogues_df.groupby("Name").size()
    node_sizes = [dialogue_counts.get(char, 0) / 50 for char in G.nodes()]
    
    # Compute layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                         node_color="lightblue", alpha=0.8)
    edge_weights = [G[u][v]['weight'] / 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("Character Interaction Network", fontsize=16)
    plt.axis("off")
    
    return plt

# Example usage:
plot_character_comparison(profiles_df, "Lexical Diversity").savefig("got_lexical_diversity.png")
plot_character_spider(profiles_df, ["Jon Snow", "Tyrion Lannister", "Daenerys Targaryen"]).savefig("got_character_traits.png")
plot_character_pca(profiles_df).savefig("got_character_similarity.png")