import pandas as pd
import numpy as np
import re
from collections import Counter
import spacy
from textblob import TextBlob
import joblib
from scipy.spatial.distance import pdist, squareform

# Import baseline model functions
from baseline_model import load_got_data

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    print("✅ spaCy model loaded successfully.")
except Exception as e:
    SPACY_AVAILABLE = False
    print(f"⚠️ spaCy model not available: {e}")
    print("Some advanced NLP features will be disabled.")

def extract_emotional_indicators(text):
    """
    Extract emotional indicators from text using predefined emotional lexicons.
    """
    # Define emotional lexicons
    emotion_lexicons = {
        'anger': ['angry', 'rage', 'fury', 'wrath', 'hostile', 'irritate', 'annoy', 'upset', 
                 'hate', 'outrage', 'fight', 'kill', 'destroy', 'attack', 'threaten', 'furious', 'mad', 'anger'],
        'fear': ['fear', 'afraid', 'scared', 'terrified', 'dread', 'horror', 'panic', 'worry', 
                'concern', 'anxious', 'nervous', 'timid', 'frightened', 'alarmed'],
        'joy': ['happy', 'joy', 'delight', 'glad', 'pleased', 'enjoy', 'cheer', 'content', 
               'smile', 'laugh', 'amuse', 'thrill', 'elated', 'euphoric', 'merry', 'cheer', 'happiness'],
        'sadness': ['sad', 'unhappy', 'miserable', 'depressed', 'gloomy', 'melancholy', 'sorrow', 
                   'grief', 'woe', 'despair', 'regret', 'mourn', 'weep', 'cry', 'tear', 'misery', 'depression', 
                   'mourning'],
        'trust': ['trust', 'faith', 'confidence', 'belief', 'assurance', 'honest', 'loyal', 
                 'reliable', 'faithful', 'dependable', 'honor', 'truth', 'ally', 'friend'],
        'disgust': ['disgust', 'revulsion', 'repel', 'loathe', 'hate', 'detest', 'abhor', 
                   'contempt', 'despise', 'nausea', 'sick', 'resentment', 'aversion', 'loathing'],
        'surprise': ['surprise', 'astonish', 'amaze', 'astound', 'wonder', 'shock', 'startle', 
                    'unexpected', 'sudden', 'abrupt', 'revelation', 'discovery', 'stun', 'startled'],
        'anticipation': ['anticipation', 'expect', 'await', 'foresee', 'predict', 'hope', 
                        'plan', 'prepare', 'ready', 'eager', 'keen', 'vigilant', 'watchful']
    }
    

    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Count occurrences of emotional words
    emotion_counts = {emotion: 0 for emotion in emotion_lexicons}
    for token in tokens:
        for emotion, lexicon in emotion_lexicons.items():
            if token in lexicon:
                emotion_counts[emotion] += 1
    
    # Calculate emotion proportions
    total_emotional_words = sum(emotion_counts.values())
    emotion_proportions = {
        f'emotion_{emotion}': count / len(tokens) if len(tokens) > 0 else 0
        for emotion, count in emotion_counts.items()
    }
    
    # Add overall emotion ratio
    emotion_proportions['emotion_ratio'] = total_emotional_words / len(tokens) if len(tokens) > 0 else 0
    
    return emotion_proportions

def analyze_dialogue_structure(sentences):
    """
    Analyze structural aspects of dialogue - length, complexity, etc.
    """
    # Basic measurements
    sentence_lengths = [len(s.split()) for s in sentences]
    word_lengths = [len(word) for s in sentences for word in s.split()]
    
    # Compute statistics
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    median_sentence_length = np.median(sentence_lengths) if sentence_lengths else 0
    avg_word_length = np.mean(word_lengths) if word_lengths else 0
    
    # Compute sentence type ratios
    question_sentences = sum(1 for s in sentences if s.strip().endswith('?'))
    exclamation_sentences = sum(1 for s in sentences if s.strip().endswith('!'))
    declarative_sentences = len(sentences) - question_sentences - exclamation_sentences
    
    # Calculate ratios
    question_ratio = question_sentences / len(sentences) if len(sentences) > 0 else 0
    exclamation_ratio = exclamation_sentences / len(sentences) if len(sentences) > 0 else 0
    declarative_ratio = declarative_sentences / len(sentences) if len(sentences) > 0 else 0
    
    return {
        'avg_sentence_length': avg_sentence_length,
        'median_sentence_length': median_sentence_length,
        'avg_word_length': avg_word_length,
        'question_ratio': question_ratio,
        'exclamation_ratio': exclamation_ratio,
        'declarative_ratio': declarative_ratio,
        'sentence_length_variance': np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
    }

def analyze_linguistic_complexity(text):
    """
    Analyze linguistic complexity using spaCy if available.
    """
    if not SPACY_AVAILABLE:
        return {
            'flesch_reading_ease': 0, 
            'num_named_entities': 0,   
            'avg_token_depth': 0,      
            'passive_voice_ratio': 0   
        }
    
    doc = nlp(text)
    
    # Named entity recognition
    named_entities = [ent.text for ent in doc.ents]
    
    # Dependency parsing - sentence complexity
    token_depths = []
    for token in doc:
        # Calculate token depth 
        depth = 0
        current = token
        while current.head != current: 
            depth += 1
            current = current.head
        token_depths.append(depth)
    
    avg_token_depth = np.mean(token_depths) if token_depths else 0
    
    # Detect passive voice constructions
    passive_constructions = 0
    for sent in doc.sents:
        if any(token.dep_ == "nsubjpass" for token in sent):
            passive_constructions += 1
    
    # Compute Flesch Reading Ease score 
    word_count = len([token for token in doc if not token.is_punct])
    sentence_count = len(list(doc.sents))
    syllable_count = sum([len([c for c in token.text if c.lower() in 'aeiouy']) for token in doc if not token.is_punct])
    
    if word_count > 0 and sentence_count > 0:
        flesch = 206.835 - (1.015 * (word_count / sentence_count)) - (84.6 * (syllable_count / word_count))
    else:
        flesch = None
    
    return {
        'flesch_reading_ease': flesch,
        'num_named_entities': len(named_entities),
        'avg_token_depth': avg_token_depth,
        'passive_voice_ratio': passive_constructions / sentence_count if sentence_count > 0 else 0
    }

def analyze_character_relationship_language(dialogues_df):
    """
    Analyze language used when characters talk about other characters.
    """
    print("Analyzing character relationship language...")
    
    # Get list of all character names 
    all_characters = list(dialogues_df['Name'].unique())
    
    # Initialize relationship sentiment matrix
    relationship_matrix = pd.DataFrame(0.0, index=all_characters, columns=all_characters)
    relationship_count_matrix = pd.DataFrame(0, index=all_characters, columns=all_characters)
    
    # For each character, analyze mentions of other characters
    for character, lines in dialogues_df.groupby('Name'):
        full_text = ' '.join(lines['Sentence'])
        
        # Look for mentions of other characters
        for other_character in all_characters:
            if character == other_character:
                continue
            
            # Split name by space "
            name_parts = other_character.split()
            first_name = name_parts[0]
            last_name = name_parts[-1] if len(name_parts) > 1 else None
            
            # Find sentences mentioning other character
            mentions = []
            for sentence in lines['Sentence']:
                if first_name in sentence or (last_name and last_name in sentence):
                    mentions.append(sentence)
            
            # analyze sentiment
            if mentions:
                mention_text = ' '.join(mentions)
                sentiment = TextBlob(mention_text).sentiment.polarity
                
                # Update relationship matrices
                relationship_matrix.loc[character, other_character] = sentiment
                relationship_count_matrix.loc[character, other_character] = len(mentions)
    
    # Save relationship matrices
    relationship_matrix.to_csv("GOT_Character_Relationship_Sentiment.csv")
    relationship_count_matrix.to_csv("GOT_Character_Relationship_Mentions.csv")
    
    print("✅ Character relationship analysis complete.")
    print("Results saved to 'GOT_Character_Relationship_Sentiment.csv' and 'GOT_Character_Relationship_Mentions.csv'")
    
    return relationship_matrix, relationship_count_matrix

def perform_enhanced_character_analysis(dialogues_df):
    """
    Perform enhanced character analysis with additional linguistic and emotional features.
    """
    print("Performing enhanced character analysis...")
    
    character_profiles = []
    
    # Process character dialogue
    for character, lines in dialogues_df.groupby('Name'):
        sentences = lines['Sentence'].tolist()
        full_text = ' '.join(sentences)
        
        # Basic dialogue stats
        num_lines = len(sentences)
        total_words = len(full_text.split())
        unique_words = len(set(word.lower() for word in full_text.split()))
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Extract emotional indicators
        emotion_features = extract_emotional_indicators(full_text)
        
        # Analyze dialogue structure
        structure_features = analyze_dialogue_structure(sentences)
        
        # Analyze linguistic complexity
        complexity_features = analyze_linguistic_complexity(full_text)
        
        # Create character profile
        profile = {
            'Character': character,
            'Dialogue_Lines': num_lines,
            'Total_Words': total_words,
            'Unique_Words': unique_words,
            'Lexical_Diversity': lexical_diversity,
            **emotion_features,
            **structure_features,
            **complexity_features
        }
        
        character_profiles.append(profile)
    
    # Create dataframe 
    enhanced_profiles_df = pd.DataFrame(character_profiles)
    
    # Save enhanced profiles
    enhanced_profiles_df.to_csv("GOT_Enhanced_Character_Profiles.csv", index=False)
    
    print(f"✅ Created enhanced profiles for {enhanced_profiles_df.shape[0]} characters.")
    print("Enhanced profiles saved to 'GOT_Enhanced_Character_Profiles.csv'")
    
    return enhanced_profiles_df

def compute_character_similarity(enhanced_profiles_df):
    """
    Compute character similarity based on linguistic and emotional features.
    """
    print("Computing character similarity...")
    
    # Select features for similarity calculation
    feature_cols = [col for col in enhanced_profiles_df.columns 
                   if col not in ['Character', 'Dialogue_Lines', 'Total_Words', 'Unique_Words']]
    
    # Filter out columns 
    valid_cols = []
    for col in feature_cols:
        if not enhanced_profiles_df[col].isnull().any() and enhanced_profiles_df[col].dtype in ['float64', 'int64']:
            valid_cols.append(col)
    
    if not valid_cols:
        print("❌ No valid numerical features found for similarity calculation.")
        characters = enhanced_profiles_df['Character'].tolist()
        n = len(characters)
        similarity_df = pd.DataFrame(np.eye(n), index=characters, columns=characters)
        similarity_df.to_csv("GOT_Character_Similarity_Matrix.csv")
        return similarity_df
    
    print(f"Using {len(valid_cols)} features for similarity calculation.")
    
    # Extract character names and features
    characters = enhanced_profiles_df['Character'].tolist()
    features = enhanced_profiles_df[valid_cols].values
    
    # Standardize features 
    std_values = np.std(features, axis=0)
    std_values[std_values == 0] = 1e-10  
    features_std = (features - np.mean(features, axis=0)) / std_values
    

    features_std = np.nan_to_num(features_std)
    
    # Compute distance matrix
    dist_matrix = squareform(pdist(features_std, metric='euclidean'))
    
    # Convert to similarity matrix (
    similarity_matrix = 1 / (1 + dist_matrix)
    
    # Create DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=characters, columns=characters)
    
    # Save similarity matrix
    similarity_df.to_csv("GOT_Character_Similarity_Matrix.csv")
    
    print("✅ Character similarity analysis complete.")
    print("Similarity matrix saved to 'GOT_Character_Similarity_Matrix.csv'")
    
    return similarity_df

def analyze_character_arcs(dialogues_df):
    """
    Analyze character arcs over time if episode/season data is available.
    """
    print("Analyzing character arcs over time...")
    
    time_col = None
    if 'Season' in dialogues_df.columns:
        time_col = 'Season'
    elif 'Episode' in dialogues_df.columns:
        time_col = 'Episode'
    
    
    # Initialize data structure 
    character_arcs = {}
    
    # analyze sentiment and complexity over time
    for character, lines in dialogues_df.groupby('Name'):
        # Group by time unit
        time_groups = lines.groupby(time_col)
        
        # Initialize lists for time series
        times = []
        sentiments = []
        question_ratios = []
        exclamation_ratios = []
        avg_sentence_lengths = []
        
        # Process each time unit
        for time_unit, time_lines in time_groups:
            time_sentences = time_lines['Sentence'].tolist()
            
            if not time_sentences:
                continue
                
            # Combine all dialogue
            time_text = ' '.join(time_sentences)
            
            # Calculate sentiment
            sentiment = TextBlob(time_text).sentiment.polarity
            
            # Calculate dialogue structure 
            structure = analyze_dialogue_structure(time_sentences)
            
            # Store values
            times.append(time_unit)
            sentiments.append(sentiment)
            question_ratios.append(structure['question_ratio'])
            exclamation_ratios.append(structure['exclamation_ratio'])
            avg_sentence_lengths.append(structure['avg_sentence_length'])
        
        # Store character arc data
        character_arcs[character] = {
            'times': times,
            'sentiments': sentiments,
            'question_ratios': question_ratios,
            'exclamation_ratios': exclamation_ratios,
            'avg_sentence_lengths': avg_sentence_lengths
        }
    
    # Save character arcs data
    joblib.dump(character_arcs, 'GOT_Character_Arcs.pkl')
    

    arc_rows = []
    for character, arc_data in character_arcs.items():
        for i, time_unit in enumerate(arc_data['times']):
            arc_rows.append({
                'Character': character,
                time_col: time_unit,
                'Sentiment': arc_data['sentiments'][i],
                'Question_Ratio': arc_data['question_ratios'][i],
                'Exclamation_Ratio': arc_data['exclamation_ratios'][i],
                'Avg_Sentence_Length': arc_data['avg_sentence_lengths'][i]
            })
    
    arc_df = pd.DataFrame(arc_rows)
    arc_df.to_csv(f"GOT_Character_Arcs_by_{time_col}.csv", index=False)
    
    print("✅ Character arc analysis complete.")
    print(f"Results saved to 'GOT_Character_Arcs_by_{time_col}.csv' and 'GOT_Character_Arcs.pkl'")
    
    return character_arcs

def main():
    dialogues_df = load_got_data()
    
    if dialogues_df is None:
        print("❌ Error: Could not load data. Analysis aborted.")
        return
    
    # Perform enhanced character analysis
    enhanced_profiles = perform_enhanced_character_analysis(dialogues_df)
    
    # Compute character similarity
    similarity_matrix = compute_character_similarity(enhanced_profiles)
    
    # Analyze character relationships
    relationship_matrix, mention_matrix = analyze_character_relationship_language(dialogues_df)
    
    # Analyze character arcs over time
    character_arcs = analyze_character_arcs(dialogues_df)
    
    print("\n✅ Enhanced character analysis complete!")
    print("Summary of files generated:")
    print("- GOT_Enhanced_Character_Profiles.csv: Detailed character profiles")
    print("- GOT_Character_Similarity_Matrix.csv: Character similarity based on linguistic features")
    print("- GOT_Character_Relationship_Sentiment.csv: Sentiment when characters mention others")
    print("- GOT_Character_Relationship_Mentions.csv: Frequency of character mentions")
    if character_arcs:
        print("- GOT_Character_Arcs_by_*.csv: Character development over time")
        print("- GOT_Character_Arcs.pkl: Raw data for character arcs")

if __name__ == "__main__":
    main()