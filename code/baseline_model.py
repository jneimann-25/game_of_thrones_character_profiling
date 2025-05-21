import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
from numpy import asarray

# Load the Game of Thrones script CSV file directly
def load_got_data(file_path="Game_of_Thrones_Script.csv"):
    """
    Load and preprocess the Game of Thrones script data.
    """
    try:
        # Load the CSV file
        script_df = pd.read_csv(file_path)
        
        # Standardize character names
        script_df["Name"] = script_df["Name"].astype(str).str.title().str.strip()
        script_df = script_df[script_df["Name"] != "Nan"]
        
        # Define main characters
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

# split sentences
def simple_tokenize_sentences(text):
    """Splits text into sentences using regex."""
    return re.split(r'(?<=[.!?]) +', text)  

# Basic character dialogue analysis function
def analyze_character_dialogue(dialogues):
    """
    Extracts basic linguistic, sentiment, and psychological traits from character dialogue.
    """
    character_stats = []

    for character, lines in dialogues.groupby("Name"):
        full_text = " ".join(lines["Sentence"])  

        # Word and Sentence Complexity
        words = full_text.split()  
        sentences = simple_tokenize_sentences(full_text)  
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Lexical Diversity (unique words / total words)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / word_count if word_count > 0 else 0

        # Pronoun Usage (ego-centric vs. collective speech)
        pronoun_counts = Counter(words)
        first_person_singular = sum([pronoun_counts[p] for p in ["I", "me", "my", "mine"] if p in pronoun_counts])
        first_person_plural = sum([pronoun_counts[p] for p in ["we", "us", "our", "ours"] if p in pronoun_counts])
        pronoun_ratio = first_person_singular / (first_person_plural + 1) 


        try:
            sentiment_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentences if sentence.strip()]
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        except Exception as e:
            print(f"TextBlob sentiment analysis failed: {e}")
            print("Using fallback sentiment analysis method...")
            
            # Fallback sentiment method
            positive_words = ['good', 'great', 'happy', 'joy', 'love', 'beautiful', 'kind', 'honor', 
                             'brave', 'loyal', 'friend', 'trust', 'hope', 'peace', 'victory', 'smile']
            negative_words = ['bad', 'terrible', 'sad', 'hate', 'angry', 'fear', 'death', 'kill', 
                             'cruel', 'betray', 'enemy', 'war', 'dead', 'pain', 'suffer', 'evil']
            
            word_list = [word.lower() for word in words]
            positive_count = sum(1 for word in word_list if word in positive_words)
            negative_count = sum(1 for word in word_list if word in negative_words)
            total_sentiment_words = positive_count + negative_count
            avg_sentiment = (positive_count - negative_count) / total_sentiment_words if total_sentiment_words > 0 else 0

        # Store metrics
        character_stats.append({
            "Character": character,
            "Word Count": word_count,
            "Sentence Count": sentence_count,
            "Avg Sentence Length": avg_sentence_length,
            "Lexical Diversity": lexical_diversity,
            "Ego-centric Speech Ratio": pronoun_ratio,
            "Avg Sentiment": avg_sentiment
        })

    return pd.DataFrame(character_stats)

# Enhanced linguistic analysis function
def expand_linguistic_analysis(dialogues_df):
    """
    Expands character analysis with additional linguistic features.
    """
    character_linguistic_traits = []
    
    for character, lines in dialogues_df.groupby("Name"):
        # Get all sentences for this character
        all_sentences = lines["Sentence"].tolist()
        full_text = " ".join(all_sentences)
        
        # Simple tokenization
        words = full_text.lower().split()

        words = [word.strip(".,!?:;\"'()[]{}") for word in words]
        words = [word for word in words if word]  # Remove empty strings
        
        # word counting
        word_count = len(words)
        unique_words = set(words)
        
        # Simple question and exclamation detection
        question_count = sum(1 for s in all_sentences if s.strip().endswith('?'))
        exclamation_count = sum(1 for s in all_sentences if s.strip().endswith('!'))
        sentence_count = len(all_sentences)
        
        # Calculate ratios
        question_ratio = question_count / sentence_count if sentence_count > 0 else 0
        exclamation_ratio = exclamation_count / sentence_count if sentence_count > 0 else 0
        
        # Word length features
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        long_words = sum(1 for word in words if len(word) > 6)
        long_word_ratio = long_words / word_count if word_count > 0 else 0
        
        # Function word analysis 
        function_words = ['the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was', 'for', 
                        'it', 'with', 'as', 'his', 'on', 'be', 'at', 'by', 'he', 'this']
        function_word_count = sum(1 for word in words if word in function_words)
        function_word_ratio = function_word_count / word_count if word_count > 0 else 0
        
        # Contraction usage (
        contractions = ["'s", "'t", "'ve", "'ll", "'re", "'d", "'m"]
        contraction_count = sum(1 for word in words if any(c in word for c in contractions))
        contraction_ratio = contraction_count / word_count if word_count > 0 else 0
        
        # Pronoun usage 
        first_person = ['i', 'me', 'my', 'mine', 'myself']
        second_person = ['you', 'your', 'yours', 'yourself']
        third_person = ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
                      'they', 'them', 'their', 'theirs', 'themselves']
        
        first_person_count = sum(1 for word in words if word in first_person)
        second_person_count = sum(1 for word in words if word in second_person)
        third_person_count = sum(1 for word in words if word in third_person)
        
        first_person_ratio = first_person_count / word_count if word_count > 0 else 0
        second_person_ratio = second_person_count / word_count if word_count > 0 else 0
        third_person_ratio = third_person_count / word_count if word_count > 0 else 0
        
        # Command words 
        command_words = ['must', 'should', 'will', 'shall', 'need', 'never', 'always', 'do', 'don\'t']
        command_count = sum(1 for word in words if word in command_words)
        command_ratio = command_count / word_count if word_count > 0 else 0
        
      
        character_linguistic_traits.append({
            "Character": character,
            "Question Ratio": question_ratio,
            "Exclamation Ratio": exclamation_ratio,
            "Avg Word Length": avg_word_length,
            "Long Word Ratio": long_word_ratio,
            "Function Word Ratio": function_word_ratio,
            "Contraction Ratio": contraction_ratio,
            "First Person Ratio": first_person_ratio,
            "Second Person Ratio": second_person_ratio,
            "Third Person Ratio": third_person_ratio,
            "Command Word Ratio": command_ratio
        })
    
    return pd.DataFrame(character_linguistic_traits)

# Main execution function
def run_analysis(file_path="Game_of_Thrones_Script.csv"):
    """
    Run the complete character profile analysis pipeline.
    """
    # Load data
    got_main_dialogue_df = load_got_data(file_path)
    
    if got_main_dialogue_df is None:
        print("❌ Error: Could not load data. Analysis aborted.")
        return
    
    try:
        # Run the basic character analysis
        print("Running basic character dialogue analysis...")
        got_character_profiles = analyze_character_dialogue(got_main_dialogue_df)
        
        # Run the expanded linguistic analysis
        print("Running enhanced linguistic analysis...")
        got_linguistic_traits = expand_linguistic_analysis(got_main_dialogue_df)
        
        # Merge the results
        print("Merging results...")
        got_enhanced_profiles = got_character_profiles.merge(got_linguistic_traits, on="Character", how="left")
        
        # Save the enhanced profile dataset
        print("Saving results...")
        got_enhanced_profiles.to_csv("Game_of_Thrones_Enhanced_Character_Profiles.csv", index=False)
        
        # Display summary
        print(f"✅ Processed {got_enhanced_profiles.shape[0]} characters with enhanced linguistic features.")
        print(f"Enhanced dataset saved as 'Game_of_Thrones_Enhanced_Character_Profiles.csv'.")
        
        # Return the dataframe for further analysis
        return got_enhanced_profiles
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

# Execute the analysis when script is run directly
if __name__ == "__main__":
    run_analysis()
