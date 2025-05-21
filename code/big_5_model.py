"""
GOT_personality_model.py - Maps Game of Thrones character linguistic features to Big Five personality traits
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_enhanced_profiles(file_path="GOT_Enhanced_Character_Profiles.csv"):
    """
    Load the enhanced character profiles created by the analysis script.
    """
    try:
        profiles_df = pd.read_csv(file_path)
        print(f"✅ Loaded {profiles_df.shape[0]} character profiles")
        return profiles_df
    except Exception as e:
        print(f"❌ Error loading profiles: {e}")
        return None

def map_to_big_five(profiles_df):
    """
    Map linguistic features to Big Five personality traits.
    
    The Big Five traits are:
    - Openness: curious, creative, open to new experiences
    - Conscientiousness: organized, responsible, thorough
    - Extraversion: outgoing, energetic, sociable
    - Agreeableness: kind, cooperative, compassionate
    - Neuroticism: anxious, moody, emotionally unstable (opposite is Emotional Stability)
    """
    print("Mapping linguistic features to Big Five personality traits...")
    
    # Initialize trait columns with zeros
    big_five_df = pd.DataFrame(index=profiles_df.index)
    big_five_df['Character'] = profiles_df['Character']
    big_five_df['Openness'] = 0.0
    big_five_df['Conscientiousness'] = 0.0
    big_five_df['Extraversion'] = 0.0
    big_five_df['Agreeableness'] = 0.0
    big_five_df['Neuroticism'] = 0.0
    
    # mappings
    
    # --- Openness ---
    if 'Lexical_Diversity' in profiles_df.columns:
        big_five_df['Openness'] += profiles_df['Lexical_Diversity'] * 3.0
    
    if 'Long_Word_Ratio' in profiles_df.columns:
        big_five_df['Openness'] += profiles_df['Long_Word_Ratio'] * 2.0
    
    if 'avg_word_length' in profiles_df.columns:
        big_five_df['Openness'] += profiles_df['avg_word_length'] * 1.5
    
    # --- Conscientiousness ---
    if 'Function_Word_Ratio' in profiles_df.columns:
        big_five_df['Conscientiousness'] += profiles_df['Function_Word_Ratio'] * 1.5
    
    if 'Contraction_Ratio' in profiles_df.columns:
        big_five_df['Conscientiousness'] -= profiles_df['Contraction_Ratio'] * 2.0
    
    if 'avg_sentence_length' in profiles_df.columns:
        big_five_df['Conscientiousness'] += profiles_df['avg_sentence_length'] * 0.1
        
    if 'emotion_disgust' in profiles_df.columns:
        big_five_df['Conscientiousness'] += profiles_df['emotion_joy'] * 1.5
    
    # --- Extraversion ---
    if 'exclamation_ratio' in profiles_df.columns:
        big_five_df['Extraversion'] += profiles_df['exclamation_ratio'] * 3.0
    
    if 'question_ratio' in profiles_df.columns:
        big_five_df['Extraversion'] += profiles_df['question_ratio'] * 2.0
    
    if 'Second_Person_Ratio' in profiles_df.columns:
        big_five_df['Extraversion'] += profiles_df['Second_Person_Ratio'] * 3.0
    
    if 'emotion_ratio' in profiles_df.columns:
        big_five_df['Extraversion'] += profiles_df['emotion_ratio'] * 1.5
        
    if 'emotion_joy' in profiles_df.columns:
        big_five_df['Extraversion'] += profiles_df['emotion_joy'] * 2.0
    
    # --- Agreeableness ---
    if 'Avg_Sentiment' in profiles_df.columns:
        big_five_df['Agreeableness'] += profiles_df['Avg_Sentiment'] * 2.5
    
    if 'Command_Word_Ratio' in profiles_df.columns:
        big_five_df['Agreeableness'] -= profiles_df['Command_Word_Ratio'] * 2.0
    
    if 'Ego-centric Speech Ratio' in profiles_df.columns:
        big_five_df['Agreeableness'] -= profiles_df['Ego-centric Speech Ratio'] * 1.5
        
    if 'emotion_trust' in profiles_df.columns:
        big_five_df['Agreeableness'] += profiles_df['emotion_trust'] * 2.0
    
    # --- Neuroticism ---
    if 'emotion_fear' in profiles_df.columns:
        big_five_df['Neuroticism'] += profiles_df['emotion_fear'] * 3.0
    
    if 'emotion_sadness' in profiles_df.columns:
        big_five_df['Neuroticism'] += profiles_df['emotion_sadness'] * 2.0
    
    if 'emotion_anger' in profiles_df.columns:
        big_five_df['Neuroticism'] += profiles_df['emotion_anger'] * 2.0
    
    if 'Avg_Sentiment' in profiles_df.columns:
        big_five_df['Neuroticism'] -= profiles_df['Avg_Sentiment'] * 1.5
    
    if 'First_Person_Ratio' in profiles_df.columns:
        big_five_df['Neuroticism'] += profiles_df['First_Person_Ratio'] * 3.0
    
    # Normalize scores 
    scaler = MinMaxScaler()
    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    big_five_df[traits] = scaler.fit_transform(big_five_df[traits])
    
    # Save the results
    big_five_df.to_csv("GOT_Character_Big_Five_Traits.csv", index=False)
    
    print("✅ Big Five personality traits mapped and saved to 'GOT_Character_Big_Five_Traits.csv'")
    return big_five_df

def create_character_summaries(big_five_df):
    """
    Create natural language summaries of character personalities based on Big Five scores.
    """
    print("Creating character personality summaries...")
    
    summaries = []
    
    for _, row in big_five_df.iterrows():
        character = row['Character']
        
        # Get trait scores
        openness = row['Openness']
        conscientiousness = row['Conscientiousness']
        extraversion = row['Extraversion']
        agreeableness = row['Agreeableness']
        neuroticism = row['Neuroticism']
        
        # Create descriptive terms based on scores
        openness_desc = "very open to new experiences and intellectually curious" if openness > 0.75 else \
                       "somewhat open to new experiences" if openness > 0.5 else \
                       "somewhat conventional and traditional" if openness > 0.25 else \
                       "very conventional and practical"
        
        conscientiousness_desc = "extremely disciplined and organized" if conscientiousness > 0.75 else \
                                "generally reliable and organized" if conscientiousness > 0.5 else \
                                "somewhat spontaneous and flexible" if conscientiousness > 0.25 else \
                                "very spontaneous and careless"
        
        extraversion_desc = "highly outgoing and energetic" if extraversion > 0.75 else \
                          "somewhat social and active" if extraversion > 0.5 else \
                          "somewhat reserved and reflective" if extraversion > 0.25 else \
                          "very introverted and solitary"
        
        agreeableness_desc = "extremely cooperative and compassionate" if agreeableness > 0.75 else \
                            "generally kind and trusting" if agreeableness > 0.5 else \
                            "somewhat competitive and skeptical" if agreeableness > 0.25 else \
                            "very challenging and suspicious"
        
        neuroticism_desc = "highly anxious and emotionally reactive" if neuroticism > 0.75 else \
                          "somewhat prone to worry and stress" if neuroticism > 0.5 else \
                          "generally emotionally stable" if neuroticism > 0.25 else \
                          "extremely calm and emotionally resilient"
        
        # Create summary
        summary = f"{character} is {openness_desc}. They are {conscientiousness_desc}, and tend to be {extraversion_desc}. " \
                 f"In interpersonal relationships, they are {agreeableness_desc}. Under pressure, they are {neuroticism_desc}."
        
        summaries.append({
            'Character': character,
            'Personality_Summary': summary,
            'Openness': openness,
            'Conscientiousness': conscientiousness,
            'Extraversion': extraversion,
            'Agreeableness': agreeableness,
            'Neuroticism': neuroticism
        })
    
    # Create and save dataframe
    summaries_df = pd.DataFrame(summaries)
    summaries_df.to_csv("GOT_Character_Personality_Summaries.csv", index=False)
    
    print("✅ Character personality summaries created and saved to 'GOT_Character_Personality_Summaries.csv'")
    return summaries_df

def main():
    profiles_df = load_enhanced_profiles()
    
    if profiles_df is None:
        print("Cannot proceed without character profiles.")
        return
    
    # Map linguistic features to Big Five traits
    big_five_df = map_to_big_five(profiles_df)
    
    # Create character personality summaries
    summaries_df = create_character_summaries(big_five_df)
    
    print("\n✅ Personality analysis complete!")

if __name__ == "__main__":
    main()

