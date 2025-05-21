"""
Game of Thrones Character Emotional Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from collections import Counter

# Define emotion word lists for basic emotion detection
emotion_words = {
    "anger": ["angry", "fury", "rage", "hate", "wrath", "furious", "mad", "anger", "outrage", "wrath"],
    "fear": ["fear", "afraid", "scared", "terror", "panic", "horror", "dread", "frighten", "terrified"],
    "joy": ["happy", "joy", "delight", "pleased", "glad", "happiness", "cheer", "smile", "ecstatic", "thrilled"],
    "sadness": ["sad", "sorrow", "grief", "unhappy", "misery", "depression", "mourning", "miserable"],
    "disgust": ["disgust", "revulsion", "contempt", "loathing", "distaste", "repulsed", "vile"],
    "surprise": ["surprise", "astonish", "amaze", "shock", "wonder", "startled", "shocked"],
    }

def load_got_data(file_path="Game_of_Thrones_Script.csv"):
    """Load the Game of Thrones dialogue data."""
    try:
        # Load the CSV file
        script_df = pd.read_csv(file_path)
        
        # Standardize character names
        script_df["Name"] = script_df["Name"].astype(str).str.title().str.strip()
        script_df = script_df[script_df["Name"] != "Nan"]
        
        # Define main characters (same as your original script)
        main_characters = {
            "Jon Snow", "Arya Stark", "Sansa Stark", "Bran Stark", "Robb Stark", "Eddard Stark",
            "Catelyn Stark", "Tyrion Lannister", "Jaime Lannister", "Cersei Lannister",
            "Tywin Lannister", "Joffrey Baratheon", "Robert Baratheon", "Stannis Baratheon",
            "Renly Baratheon", "Davos Seaworth", "Melisandre", "Brienne Of Tarth",
            "Sandor Clegane", "Gregor Clegane", "Daenerys Targaryen", "Viserys Targaryen",
            "Jorah Mormont", "Barristan Selmy", "Daario Naharis", "Theon Greyjoy", "Yara Greyjoy",
            "Balon Greyjoy", "Ramsay Bolton", "Roose Bolton", "Petyr Baelish", "Varys",
            "Samwell Tarly", "Gilly", "Tormund Giantsbane", "Margaery Tyrell", "Olenna Tyrell",
            "Loras Tyrell", "High Sparrow", "Gendry", "Ellaria Sand", "Oberyn Martell",
            "Euron Greyjoy", "Missandei", "Grey Worm", "Qyburn", "Beric Dondarrion",
            "Benjen Stark", "Meera Reed", "Jojen Reed", "Alliser Thorne"
        }
        
        # Filter only main characters
        main_dialogue_df = script_df[script_df["Name"].isin(main_characters)]
        
        print(f"✅ Loaded {main_dialogue_df.shape[0]} lines of dialogue for {main_dialogue_df['Name'].nunique()} main characters.")
        return main_dialogue_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_character_emotions(dialogues_df):
    """
    Analyze emotional patterns in character dialogue.
    """
    character_emotion_data = []
    
    for character, lines in dialogues_df.groupby("Name"):
        # Get all sentences for this character
        all_sentences = lines["Sentence"].tolist()
        full_text = " ".join(all_sentences)
        
        # Simple tokenization
        words = full_text.lower().split()
        # Remove punctuation
        words = [word.strip(".,!?:;\"'()[]{}") for word in words]
        words = [word for word in words if word]  # Remove empty strings
        
        # Basic sentiment analysis
        sentiment_scores = []
        for sentence in all_sentences:
            try:
                score = TextBlob(sentence).sentiment.polarity
                sentiment_scores.append(score)
            except:
                # Skip problematic sentences
                continue
        
        # Calculate sentiment statistics
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)  # Emotional volatility
            max_sentiment = max(sentiment_scores)
            min_sentiment = min(sentiment_scores)
            sentiment_range = max_sentiment - min_sentiment
        else:
            avg_sentiment = sentiment_std = max_sentiment = min_sentiment = sentiment_range = 0
        
        # Count emotional words by category
        emotion_counts = {emotion: 0 for emotion in emotion_words}
        total_emotion_words = 0
        
        for word in words:
            for emotion, word_list in emotion_words.items():
                if word in word_list:
                    emotion_counts[emotion] += 1
                    total_emotion_words += 1
        
        # Calculate emotion proportions
        emotion_proportions = {}
        for emotion, count in emotion_counts.items():
            emotion_proportions[f"{emotion}_ratio"] = count / len(words) if len(words) > 0 else 0
        
        # Calculate emotional diversity (number of different emotions expressed)
        emotions_expressed = sum(1 for emotion, count in emotion_counts.items() if count > 0)
        
        # Emotional intensity - exclamation marks per sentence
        exclamation_count = sum(1 for s in all_sentences if s.strip().endswith('!'))
        exclamation_ratio = exclamation_count / len(all_sentences) if all_sentences else 0
        
        # Store results
        character_data = {
            "Character": character,
            "Avg_Sentiment": avg_sentiment,
            "Emotional_Volatility": sentiment_std,
            "Sentiment_Range": sentiment_range,
            "Max_Positive": max_sentiment,
            "Max_Negative": min_sentiment,
            "Emotions_Expressed": emotions_expressed,
            "Exclamation_Ratio": exclamation_ratio,
            "Total_Emotion_Words": total_emotion_words
        }
        
        # Add emotion proportions
        character_data.update(emotion_proportions)
        
        character_emotion_data.append(character_data)
    
    return pd.DataFrame(character_emotion_data)

def visualize_character_emotions(emotion_df, output_dir="."):
    """Create visualizations of character emotions."""
    # 1. Top 10 characters by emotional volatility
    plt.figure(figsize=(12, 8))
    top_volatile = emotion_df.sort_values("Emotional_Volatility", ascending=False).head(10)
    sns.barplot(x="Emotional_Volatility", y="Character", data=top_volatile)
    plt.title("Top 10 Characters by Emotional Volatility", fontsize=16)
    plt.xlabel("Emotional Volatility (Standard Deviation of Sentiment)", fontsize=12)
    plt.tight_layout()
    plt.savefig("emotional_volatility.png")
    plt.close()
    
    # 2. Emotion profile for top 8 characters
    top_chars = emotion_df.sort_values("Total_Emotion_Words", ascending=False).head(8)["Character"].tolist()
    emotion_cols = [f"{emotion}_ratio" for emotion in emotion_words.keys()]
    
    # Reshape data for plotting
    plot_data = []
    for _, row in emotion_df[emotion_df["Character"].isin(top_chars)].iterrows():
        for emotion in emotion_words.keys():
            plot_data.append({
                "Character": row["Character"],
                "Emotion": emotion.capitalize(),
                "Ratio": row[f"{emotion}_ratio"]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Character", y="Ratio", hue="Emotion", data=plot_df)
    plt.title("Emotion Profiles of Main Characters", fontsize=16)
    plt.xlabel("Character", fontsize=12)
    plt.ylabel("Proportion of Dialogue", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Emotion")
    plt.tight_layout()
    plt.savefig("emotion_profiles.png")
    plt.close()
    
    # 3. Sentiment distribution
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Character", y="Avg_Sentiment", data=emotion_df.sort_values("Avg_Sentiment"))
    plt.title("Sentiment Distribution by Character", fontsize=16)
    plt.xlabel("Character", fontsize=12)
    plt.ylabel("Average Sentiment", fontsize=12)
    plt.xticks(rotation=90)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png")
    plt.close()
    
    print(f"✅ Created visualizations in the current directory")

def run_emotional_analysis(file_path="Game_of_Thrones_Script.csv", output_dir="."):
    """Run the emotional analysis pipeline and generate outputs."""
    print("Loading Game of Thrones dialogue data...")
    got_dialogue = load_got_data(file_path)
    
    if got_dialogue is None:
        print("❌ Error loading data. Analysis aborted.")
        return
    
    print("Analyzing character emotions...")
    emotion_profiles = analyze_character_emotions(got_dialogue)
    
    if emotion_profiles.empty:
        print("❌ No emotion data could be generated.")
        return
    
    # Save the data directly to current directory
    emotion_profiles.to_csv("GoT_Character_Emotions.csv", index=False)
    print(f"✅ Saved emotion data to GoT_Character_Emotions.csv")
    
    # Visualize the data to current directory
    print("Creating visualizations...")
    visualize_character_emotions(emotion_profiles, ".")
    
    # Display summary information
    print("\n----- EMOTION ANALYSIS SUMMARY -----")
    print(f"Analyzed {len(emotion_profiles)} characters")
    
    # Most emotional characters
    most_volatile = emotion_profiles.sort_values("Emotional_Volatility", ascending=False)["Character"].iloc[0]
    most_positive = emotion_profiles.sort_values("Avg_Sentiment", ascending=False)["Character"].iloc[0]
    most_negative = emotion_profiles.sort_values("Avg_Sentiment")["Character"].iloc[0]
    
    print(f"Most emotionally volatile character: {most_volatile}")
    print(f"Most positive character: {most_positive}")
    print(f"Most negative character: {most_negative}")
    
    # Return the data for further analysis
    return emotion_profiles

# Run the analysis if the script is executed directly
if __name__ == "__main__":
    run_emotional_analysis()
