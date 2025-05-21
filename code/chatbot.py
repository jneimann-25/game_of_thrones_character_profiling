
import pandas as pd
import numpy as np
import re
import random
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
import string
import joblib
import json
from difflib import get_close_matches
from typing import Dict, List, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Try to download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    print("⚠️ NLTK download failed. Using existing resources if available.")

class GOTCharacterChatbot:
    def __init__(self):
        self.character_dialogues = {}
        self.character_personalities = {}
        self.character_emotions = {}
        self.active_character = None
        self.conversation_history = []
        self.loaded = False
        self.generation_models = {}  
        self.character_states = {}  
        self.character_knowledge = {}
        self.world_knowledge = {}
        
        # Dialogue style adjustments 
        self.dialogue_style_adjustments = {
            'Tyrion Lannister': {'wit': 0.9, 'sarcasm': 0.8, 'intelligence': 0.95},
            'Jon Snow': {'directness': 0.9, 'honor': 0.95, 'simplicity': 0.85},
            'Daenerys Targaryen': {'authority': 0.95, 'visionary': 0.9, 'determination': 0.95},
            'Cersei Lannister': {'cunning': 0.95, 'ruthlessness': 0.9, 'pride': 0.95},
            'Arya Stark': {'determination': 0.9, 'directness': 0.85, 'vengefulness': 0.8},
            'Sansa Stark': {'diplomacy': 0.9, 'caution': 0.85, 'resilience': 0.9},
            'Jaime Lannister': {'confidence': 0.9, 'sarcasm': 0.7, 'honor': 0.6},
            'Petyr Baelish': {'cunning': 0.95, 'manipulation': 0.9, 'ambition': 0.95},
            'Varys': {'intelligence': 0.95, 'subtlety': 0.9, 'loyalty': 0.8},
            'Theon Greyjoy': {'insecurity': 0.85, 'regret': 0.8, 'loyalty': 0.7},
            'Brienne of Tarth': {'honor': 0.95, 'loyalty': 0.95, 'directness': 0.9},
            'Samwell Tarly': {'intelligence': 0.9, 'kindness': 0.9, 'humility': 0.85}
        }
    
    def load_data(self, 
                 dialogues_path="Game_of_Thrones_Main_Characters_Dialogue.csv", 
                 personality_path="GOT_Character_Personality_Summaries.csv",
                 enhanced_profiles_path="GOT_Enhanced_Character_Profiles.csv",
                 knowledge_graph_path="got_knowledge_graph.json"):
        """Load character dialogues, personality data, and knowledge graph."""
        try:
            print(f"Loading dialogues from {dialogues_path}...")
            dialogues_df = pd.read_csv(dialogues_path)
            print(f"Dialogue data shape: {dialogues_df.shape}")
            print(f"Dialogue columns: {dialogues_df.columns.tolist()}")
            
            if 'Name' not in dialogues_df.columns:
                print("ERROR: 'Name' column not found in dialogues file!")
                print(f"Available columns: {dialogues_df.columns.tolist()}")
                for col in dialogues_df.columns:
                    unique_vals = dialogues_df[col].unique()
                    if len(unique_vals) < 20:  
                        print(f"Possible character column: {col}")
                        print(f"Values: {unique_vals}")
            
            # Process character dialogues
            print("Processing character dialogues...")
            for character, group in dialogues_df.groupby('Name'):
                print(f"Found character: {character} with {len(group)} dialogues")
                self.character_dialogues[character] = group['Sentence'].tolist()
            
            # load personality data
            print(f"Loading personality data from {personality_path}...")
            try:
                personality_df = pd.read_csv(personality_path)
                print(f"Personality data shape: {personality_df.shape}")
                
                # Process personality data
                for _, row in personality_df.iterrows():
                    character = row['Character']
                    self.character_personalities[character] = {
                        'summary': row['Personality_Summary'],
                        'openness': row['Openness'],
                        'conscientiousness': row['Conscientiousness'],
                        'extraversion': row['Extraversion'],
                        'agreeableness': row['Agreeableness'],
                        'neuroticism': row['Neuroticism']
                    }
            except Exception as e:
                print(f"⚠️ Warning: Could not load personality data: {e}")
                # Create dummy personality data for testing
                for character in self.character_dialogues:
                    self.character_personalities[character] = {
                        'summary': f"{character}'s personality",
                        'openness': 0.5,
                        'conscientiousness': 0.5,
                        'extraversion': 0.5,
                        'agreeableness': 0.5,
                        'neuroticism': 0.5
                    }
            
            # Load enhanced profiles
            print(f"Loading emotional profiles from {enhanced_profiles_path}...")
            try:
                profiles_df = pd.read_csv(enhanced_profiles_path)
                
                # Process emotional data
                emotion_cols = [col for col in profiles_df.columns if col.startswith('emotion_') and col != 'emotion_ratio']
                for _, row in profiles_df.iterrows():
                    character = row['Character']
                    emotions = {}
                    
                    for col in emotion_cols:
                        emotion_name = col.replace('emotion_', '')
                        emotions[emotion_name] = row[col] if col in row else 0.0
                    
                    self.character_emotions[character] = emotions
            except Exception as e:
                print(f"⚠️ Warning: Could not load emotional profiles: {e}")
                # Create dummy 
                for character in self.character_dialogues:
                    self.character_emotions[character] = {
                        'joy': 0.3,
                        'anger': 0.3,
                        'sadness': 0.3,
                        'fear': 0.3
                    }
            
            try:
                self.load_knowledge_graph(knowledge_graph_path)
            except Exception as e:
                print(f"⚠️ Warning: Could not load knowledge graph: {e}")
                

            print("Initializing dialogue embeddings (this may take a while)...")
            self.initialize_embeddings()
            
            print(f"✅ Loaded data for {len(self.character_dialogues)} characters")
            self.loaded = True
            return True
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("Attempting to create test data instead...")
            return self.create_test_data()
        
    def load_knowledge_graph(self, path="got_knowledge_graph.json"):
        """Improved JSON loading with error handling"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.knowledge = json.load(f)
            print(f"✅ Successfully loaded knowledge graph from {path}")
        except FileNotFoundError:
            print(f"❌ Error: File {path} not found in {os.getcwd()}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in {path}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error loading {path}: {e}")
    
    def get_character_list(self):
        """Return a list of available characters."""
        return list(self.character_dialogues.keys())
    
    def set_active_character(self, character_name):
        """Set the active character for the chatbot."""
        characters = self.get_character_list()
        
        if character_name in characters:
            self.active_character = character_name
            return True
        else:
            matches = get_close_matches(character_name, characters, n=1, cutoff=0.6)
            if matches:
                self.active_character = matches[0]
                return True
            else:
                return False
    
    def get_character_info(self, character_name=None):
        """Get information about a character."""
        if character_name is None:
            character_name = self.active_character
        
        if character_name not in self.character_personalities:
            return None
        
        # Get personality and emotion data
        personality = self.character_personalities[character_name]
        emotions = self.character_emotions.get(character_name, {})
        
        # Get dominant emotions
        dominant_emotions = []
        if emotions:
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            dominant_emotions = [emotion for emotion, score in sorted_emotions[:3] if score > 0.1]
        
        # Create character info
        info = {
            'name': character_name,
            'personality_summary': personality['summary'],
            'dominant_emotions': dominant_emotions,
            'personality_traits': {
                'openness': personality['openness'],
                'conscientiousness': personality['conscientiousness'],
                'extraversion': personality['extraversion'],
                'agreeableness': personality['agreeableness'],
                'neuroticism': personality['neuroticism']
            },
            'style_adjustments': self.dialogue_style_adjustments.get(character_name, {})
        }
        
        return info
    
    def get_response_sentiment(self, text):
        """Analyze sentiment of input text."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def extract_keywords(self, text):
        """Extract important keywords from text."""
        try:
            tokens = word_tokenize(text.lower())
            
            # Remove punctuation and stopwords
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                        'be', 'been', 'being', 'to', 'of', 'for', 'with', 'by', 'about', 
                        'against', 'between', 'into', 'through', 'during', 'before', 'after',
                        'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
                        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                        'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'will', 'just',
                        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'couldn', 'didn', 'doesn',
                        'hadn', 'hasn', 'haven', 'isn', 'shouldn', 'wasn', 'weren', 'won', 
                        'wouldn', 'this', 'that', 'these', 'those', 'me', 'my', 'mine', 'you',
                        'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its',
                        'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs', 'what'}
            
            tokens = [token for token in tokens if token not in stopwords and token not in string.punctuation]
            
            # Return top keywords
            return tokens
        except:
            return []
    
    def find_relevant_dialogues(self, query, character=None, max_results=5):
        """Find relevant character dialogues based on query keywords."""
        if character is None:
            character = self.active_character
        
        if character not in self.character_dialogues:
            return []
        
        dialogues = self.character_dialogues[character]
        keywords = self.extract_keywords(query)
        
        if not keywords:
            return random.sample(dialogues, min(max_results, len(dialogues)))
        
        # Score dialogues based on keyword matches
        scored_dialogues = []
        for dialogue in dialogues:
            dialogue_keywords = self.extract_keywords(dialogue)
            matches = sum(1 for kw in keywords if kw in dialogue_keywords)
            if matches > 0:
                scored_dialogues.append((dialogue, matches))
        
        # Sort by matches and return top results
        scored_dialogues.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in scored_dialogues[:max_results]]
    
    def initialize_generation_model(self, character):
        """
        Load a fine-tuned model for the character if available,
        otherwise load and cache a base fallback model.
        """
        model_path = f"./models/{character.replace(' ', '_')}"

        print(f"\nAttempting to load generation model for {character}...")
        print(f"Checking path: {os.path.abspath(model_path)}") 

        # --- Try loading the fine-tuned character model ---
        try:
            if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, 'pytorch_model.bin')):
                print(f"Found specific model directory for {character}. Loading from {model_path}...")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)

                device_id = 0 if torch.cuda.is_available() else -1
                device_name = "GPU" if device_id == 0 else "CPU"
                print(f"--> Using device: {device_name}")

                # Create and cache the pipeline for the fine-tuned model
                self.generation_models[character] = pipeline(
                    'text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device=device_id
                )
                print(f"✅ Successfully loaded FINE-TUNED model for {character}.")
                return self.generation_models[character]
            else:
                print(f"INFO: No specific model directory found at {model_path} or it's incomplete.")
                raise FileNotFoundError("Specific model not found, proceeding to fallback.")

        except Exception as e:
            print(f"⚠️ Could not load specific model for {character}. Reason: {e}")

            # --- Fallback to a base model ---
            base_fallback_model_name = "gpt2" 
            print(f"\nAttempting to load FALLBACK base model: {base_fallback_model_name}")


            cache_key = f"__base_{base_fallback_model_name}"
            if cache_key in self.generation_models:
                print(f"--> Using cached base model pipeline: {base_fallback_model_name}")
                return self.generation_models[cache_key]
            else:
                try:
                    print(f"Loading base model {base_fallback_model_name} for the first time...")
                    base_tokenizer = AutoTokenizer.from_pretrained(base_fallback_model_name)
                    base_model = AutoModelForCausalLM.from_pretrained(base_fallback_model_name)

                    device_id = 0 if torch.cuda.is_available() else -1
                    device_name = "GPU" if device_id == 0 else "CPU"
                    print(f"--> Using device: {device_name} for fallback model")


                    base_pipeline = pipeline(
                        'text-generation',
                        model=base_model,
                        tokenizer=base_tokenizer,
                        device=device_id
                    )
                    self.generation_models[cache_key] = base_pipeline 
                    print(f"✅ Successfully loaded and cached base fallback model: {base_fallback_model_name}")
                    return base_pipeline
                except Exception as final_e:
                    print(f"❌ FATAL: Could not load base fallback model {base_fallback_model_name}: {final_e}")
                    print("❌ Chatbot generation will likely fail.")
                    return None
    
    def generate_original_response(self, prompt, character):
        """Generate original dialogue in character using max_new_tokens"""
        if character not in self.generation_models:

            pipeline_obj = self.initialize_generation_model(character)
            if pipeline_obj is None: 
                 return "I seem to be having trouble thinking right now."
            self.generation_models[character] = pipeline_obj
        else:
            pipeline_obj = self.generation_models[character]


        if pipeline_obj is None:
             return "My thoughts are unclear at the moment."

        # Get character style adjustments
        style = self.dialogue_style_adjustments.get(character, {})


        simple_prompt = prompt + f"\n{character}:" # Use the 'prompt' (contextual_query)

        print(f"DEBUG: Sending prompt to pipeline (first 100 chars): '{simple_prompt[:100]}...'")

        try:
            response = pipeline_obj(
                simple_prompt,
                max_new_tokens=60,  
                num_return_sequences=1,
                temperature=0.6, 
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=pipeline_obj.tokenizer.eos_token_id 
            )

            # --- Better Extraction ---
            generated_text = response[0]['generated_text']
            output_text = generated_text[len(simple_prompt):].strip()
            output_text = output_text.split('<|endoftext|>')[0].strip()
            if '.' in output_text:
                 output_text = output_text.rsplit('.', 1)[0] + '.'
            elif '?' in output_text:
                 output_text = output_text.rsplit('?', 1)[0] + '?'
            elif '!' in output_text:
                 output_text = output_text.rsplit('!', 1)[0] + '!'


            if not output_text:
                 print("DEBUG: Pipeline generated empty text after processing.")
                 return "..." 

            print(f"DEBUG: Raw generated response from pipeline: '{output_text}'") 

        except Exception as gen_e:
            print(f"❌ Error during pipeline generation: {gen_e}")
            return "I... I cannot find the words." 
    
    def adapt_response_to_personality(self, response_candidates, query):
        """Adapt response selection based on character personality."""
        if not self.active_character or not response_candidates:
            return random.choice(response_candidates) if response_candidates else "..."
        
        # Get personality traits
        personality = self.character_personalities.get(self.active_character, {})
        if not personality:
            return random.choice(response_candidates)
        
        # Analyze query sentiment
        query_sentiment = self.get_response_sentiment(query)
        
        # Get trait scores
        extraversion = personality.get('extraversion', 0.5)
        agreeableness = personality.get('agreeableness', 0.5)
        neuroticism = personality.get('neuroticism', 0.5)
        
        # Score responses based on personality match
        scored_responses = []
        for response in response_candidates:
            score = 0
            response_sentiment = self.get_response_sentiment(response)
            response_length = len(response.split())
            
            # Extraverts give longer responses
            if extraversion > 0.6 and response_length > 10:
                score += extraversion * 2
            elif extraversion < 0.4 and response_length < 8:
                score += (1 - extraversion) * 2
            
            # Agreeable characters match sentiment
            sentiment_match = 1 - abs(query_sentiment - response_sentiment)
            if agreeableness > 0.6:
                score += agreeableness * sentiment_match * 3
            
            # Neurotic characters givecnegative responses
            if neuroticism > 0.6 and response_sentiment < 0:
                score += neuroticism * 2
            elif neuroticism < 0.4 and response_sentiment > 0:
                score += (1 - neuroticism) * 2
            
            scored_responses.append((response, score))
        
        # Sort by score and select top response 
        scored_responses.sort(key=lambda x: x[1], reverse=True)
        top_responses = scored_responses[:max(1, len(scored_responses)//3)]
        return random.choice([r[0] for r in top_responses])
    
    def update_emotional_state(self, character, user_input, chatbot_response):
        """Update character's emotional state based on conversation"""
        if character not in self.character_states:
            self.character_states[character] = {
                'baseline_emotions': self.character_emotions.get(character, {}),
                'current_emotions': self.character_emotions.get(character, {}).copy(),
                'last_update': 0
            }
        
        # Analyze sentiment of both user input and response
        input_sentiment = self.get_response_sentiment(user_input)
        response_sentiment = self.get_response_sentiment(chatbot_response)
        
        # Modify emotional state 
        state = self.character_states[character]['current_emotions']
        
        # Example emotional adjustments
        if input_sentiment < -0.5:  # Very negative input
            state['anger'] = min(1.0, state.get('anger', 0) + 0.2)
            state['disgust'] = min(1.0, state.get('disgust', 0) + 0.1)
        elif input_sentiment > 0.5:  # Very positive input
            state['joy'] = min(1.0, state.get('joy', 0) + 0.2)
        
        # Gradual decay towards baseline
        for emotion in state:
            baseline = self.character_states[character]['baseline_emotions'].get(emotion, 0)
            state[emotion] = baseline + (state[emotion] - baseline) * 0.9  # 10% decay
    
    def get_character_opinion(self, about_character: str) -> str:
        """Get the active character's opinion about another character"""
        if not self.active_character or about_character not in self.character_knowledge:
            return ""
            
        opinions = self.character_knowledge[self.active_character].get('opinions', {})
        return opinions.get(about_character, "I don't have much to say about them.")
    
    def generate_response(self, query):
        """Generate a character-appropriate response to user input with robust error handling."""
        if not self.loaded or not self.active_character:
            return "Chatbot not properly initialized. Please load data and set an active character."
        
        try:
            print("Starting response generation...")
            self.conversation_history.append(("user", query))
            

            recent_history = self.conversation_history[-3:]  # Use more history
            context = "\n".join([f"{speaker}: {text}" for speaker, text in recent_history])
            contextual_query = f"Context:\n{context}\nUser's latest query: {query}"
            
            print("Checking for mentioned characters...")
            # Check for references to other characters
            mentioned_characters = []
            character_list = list(self.character_dialogues.keys())
            for char in character_list:
                if char.lower() in query.lower() and char != self.active_character:
                    mentioned_characters.append(char)
            
            print("Finding relevant dialogues...")
            # Find relevant dialogues 
            relevant_dialogues = []
            try:
                if hasattr(self, 'embedding_model') and self.embedding_model is not None and self.active_character in self.dialogue_embeddings:
                    print("Using semantic search...")
                    relevant_dialogues = self.find_relevant_dialogues_semantic(contextual_query, max_results=5)
                else:
                    print("Using keyword search...")
                    relevant_dialogues = self.find_relevant_dialogues(contextual_query, max_results=5)
            except Exception as e:
                print(f"Error in dialogue retrieval: {e}")
                relevant_dialogues = []
            
            print(f"Found {len(relevant_dialogues)} relevant dialogues")
            
            print("Generating original response...")
            # Generate original response
            try:
                original_response = self.generate_original_response(contextual_query, self.active_character)
            except Exception as e:
                print(f"Error in original response generation: {e}")
                original_response = "I'm not sure how to respond to that."
            
            # Default response 
            response = f"As {self.active_character}, I acknowledge your message."
            
            try:
                print("Determining response strategy...")
                if relevant_dialogues and len(relevant_dialogues) > 0:
                    relevance_score = 0
                    try:
                        relevance_score = self.calculate_relevance_score(relevant_dialogues[0], query)
                    except Exception as e:
                        print(f"Error calculating relevance: {e}")
                    
                    print(f"Top relevance score: {relevance_score}")
                    
                    if relevance_score > 0.7:
                        # High relevance - favor retrieval but still blend
                        if random.random() < 0.5:  # 50% chance to use retrieved response as base
                            retrieved_response = self.adapt_response_to_personality(relevant_dialogues, contextual_query)
                            # Blend with generated content 
                            response = self.blend_responses(retrieved_response, original_response, ratio=0.7)
                        else:
                            response = original_response
                    else:
                        # Low relevance - favor generation
                        if random.random() < 0.8:  # 80% generation when retrieval relevance is low
                            response = original_response
                        else:
                            retrieved_response = self.adapt_response_to_personality(relevant_dialogues, contextual_query)
                            response = self.blend_responses(retrieved_response, original_response, ratio=0.3)
                else:
                    # No relevant dialogues - use original response
                    response = original_response
                    
                print("Handling mentioned characters...")
                if mentioned_characters:
                    knowledge_response = ""
                    for char in mentioned_characters:
                        opinion = self.get_character_opinion(char)
                        if opinion:
                            knowledge_response += f" Regarding {char}, {opinion} "
                            
                    if knowledge_response:
                        response = self.blend_responses(response, knowledge_response, ratio=0.8)
                
                print("Post-processing response...")
                response = self.post_process_response(response, query)
            except Exception as e:
                print(f"Error in response selection/blending: {e}")
                # Fallback to a simple response
                response = f"I am {self.active_character}. I need to think about what you've said."
            
            try:
                print("Updating emotional state...")
                # Update emotional state after response generation
                self.update_emotional_state(self.active_character, query, response)
            except Exception as e:
                print(f"Error updating emotional state: {e}")
            
            # Add to conversation history
            self.conversation_history.append((self.active_character, response))
            
            print("Response generation complete")
            return response
            
        except Exception as e:
            print(f"Unexpected error in generate_response: {e}")
            import traceback
            traceback.print_exc()
            return f"I am {self.active_character}, but I seem to be having trouble formulating a response right now."
    
    def get_character_summary(self, character_name=None):
        """Get a summary of a character's personality and emotional traits."""
        if character_name is None:
            character_name = self.active_character
        
        if not character_name or character_name not in self.character_personalities:
            return "Character information not available."
        
        info = self.get_character_info(character_name)
        if not info:
            return "Character information not available."
        
        # Create summary text
        summary = f"Character Profile: {character_name}\n\n"
        summary += f"Personality: {info['personality_summary']}\n\n"
        
        if info['dominant_emotions']:
            summary += f"Dominant Emotions: {', '.join(info['dominant_emotions'])}\n\n"
        
        traits = info['personality_traits']
        summary += "Big Five Traits (0-1 scale):\n"
        summary += f"- Openness: {traits['openness']:.2f}\n"
        summary += f"- Conscientiousness: {traits['conscientiousness']:.2f}\n"
        summary += f"- Extraversion: {traits['extraversion']:.2f}\n"
        summary += f"- Agreeableness: {traits['agreeableness']:.2f}\n"
        summary += f"- Neuroticism: {traits['neuroticism']:.2f}\n\n"
        
        # Add current emotional state if available
        if character_name in self.character_states:
            emotions = self.character_states[character_name]['current_emotions']
            summary += "Current Emotional State:\n"
            for emotion, value in emotions.items():
                if value > 0.1:
                    summary += f"- {emotion}: {value:.2f}\n"
        
        return summary
    
    def calculate_relevance_score(self, dialogue, query):
        """Calculate how relevant a dialogue is to the query"""
        # Simple implementation - you can enhance this with embeddings
        dialogue_words = set(self.extract_keywords(dialogue))
        query_words = set(self.extract_keywords(query))
        
        if not query_words:
            return 0.3  # Default medium-low score
        
        # Calculate Jaccard similarity
        intersection = len(dialogue_words.intersection(query_words))
        union = len(dialogue_words.union(query_words))
        
        return intersection / union if union > 0 else 0
    
    def blend_responses(self, response1, response2, ratio=0.5):
        """Blend two responses together with the given ratio"""

        if random.random() < 0.3:
            transitions = [". Also, ", ". Furthermore, ", ". I should add that ", ". And ", ". Remember, "]
            return response1 + random.choice(transitions) + response2
        else:
            if random.random() < ratio:
                main_response = response1
                secondary = response2
            else:
                main_response = response2
                secondary = response1
                

            words = secondary.split()
            if len(words) > 5:
                phrase_length = random.randint(3, min(6, len(words)-1))
                start_idx = random.randint(0, len(words)-phrase_length)
                phrase = " ".join(words[start_idx:start_idx+phrase_length])
                

                main_words = main_response.split()
                if len(main_words) > 5:
                    insert_idx = random.randint(2, len(main_words)-2)
                    result = " ".join(main_words[:insert_idx]) + " " + phrase + " " + " ".join(main_words[insert_idx:])
                    return result
                
            return main_response
    
    def post_process_response(self, response, query):
        """Post-process the response to improve quality"""
        # Get character style adjustments
        style = self.dialogue_style_adjustments.get(self.active_character, {})
        
        # Add characteristic phrases
        character_phrases = {
            'Tyrion Lannister': ["I drink and I know things.", "A Lannister always pays his debts.", "That's what I do."],
            'Jon Snow': ["Winter is coming.", "The North remembers.", "I know nothing."],
            'Daenerys Targaryen': ["I will take what is mine with fire and blood.", "Dracarys.", "I am Daenerys Stormborn."],
            # Add more for other characters
        }
        
        # Sometimes add a characteristic phrase
        if self.active_character in character_phrases and random.random() < 0.15:
            phrase = random.choice(character_phrases[self.active_character])
            if phrase.lower() not in response.lower():
                if random.random() < 0.5:
                    response = phrase + " " + response
                else:
                    response = response + " " + phrase
        
        # Ensure response isn't too short
        if len(response.split()) < 3:
            base_responses = {
                'Tyrion Lannister': "Let me think about that for a moment.",
                'Jon Snow': "I'm not sure what to say to that.",
                'Daenerys Targaryen': "You speak to a queen. Choose your words carefully.",
                # Add more defaults for other characters
            }
            default = base_responses.get(self.active_character, "I'm considering how to respond to that.")
            response = default
        
        return response
    
    def initialize_embeddings(self):
        """Initialize embeddings for all dialogues with better progress tracking"""
        try:
            print("Loading SentenceTransformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model loaded successfully")
            self.dialogue_embeddings = {}
            
            # Count total dialogues for overall progress
            total_characters = len(self.character_dialogues)
            completed_characters = 0
            
            for character, dialogues in self.character_dialogues.items():
                completed_characters += 1
                print(f"Processing character {completed_characters}/{total_characters}: {character} ({len(dialogues)} dialogues)")
                
                # Process in smaller batches with manual progress tracking
                batch_size = 16
                total_batches = (len(dialogues) + batch_size - 1) // batch_size
                all_embeddings = []
                
                for i in range(0, len(dialogues), batch_size):
                    batch = dialogues[i:i+batch_size]
                    current_batch = i//batch_size + 1
                    
                    # Clear batch progress information
                    print(f"  Batch {current_batch}/{total_batches}", end="\r")
                    
                    # Process 
                    batch_embeddings = self.embedding_model.encode(
                        batch, 
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    all_embeddings.extend(batch_embeddings)
                
                print(f"  Completed {len(dialogues)} dialogues for {character}")
                
                # Store embeddings
                self.dialogue_embeddings[character] = {
                    'texts': dialogues,
                    'embeddings': np.array(all_embeddings)
                }
            
            print(f"✅ Completed embeddings for all {total_characters} characters")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing embeddings: {e}")
            import traceback
            traceback.print_exc()
            self.embedding_model = None
            return False
    
    def find_relevant_dialogues_semantic(self, query, character=None, max_results=5):
        """Find relevant dialogues using semantic similarity"""
        if character is None:
            character = self.active_character
        
        if not hasattr(self, 'embedding_model') or self.embedding_model is None:
            return self.find_relevant_dialogues(query, character, max_results)
        
        if character not in self.dialogue_embeddings:
            return self.find_relevant_dialogues(query, character, max_results)
        
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode(query)
            
            # Get character embeddings and dialogues
            character_embeddings = self.dialogue_embeddings[character]['embeddings']
            character_dialogues = self.dialogue_embeddings[character]['texts']
            
            # Calculate cosine similarity
            similarities = cosine_similarity([query_embedding], character_embeddings)[0]
            
            # Get indices of top matches
            top_indices = np.argsort(similarities)[-max_results:][::-1]
            
            # Return corresponding dialogues and their similarity scores
            results = [(character_dialogues[i], similarities[i]) for i in top_indices]
            
            return [dialogue for dialogue, _ in results]
        except Exception as e:
            print(f"❌ Error in semantic search: {e}")
            # Fall back to keyword-based search
            return self.find_relevant_dialogues(query, character, max_results)
    
    def run_interactive(self):
        """Run an interactive console for the chatbot."""
        if not self.loaded:
            success = self.load_data()
            if not success:
                print("Failed to load required data. Exiting.")
                return
        
        print("Welcome to the Game of Thrones Character Chatbot!")
        print("Available characters:", ", ".join(self.get_character_list()))
        
        while True:
            if not self.active_character:
                character = input("\nChoose a character to talk to: ")
                if self.set_active_character(character):
                    print(f"\nNow talking to {self.active_character}")
                    print(self.get_character_summary())
                    print("\nStart chatting! Commands:")
                    print("- 'exit': Quit")
                    print("- 'switch': Change character")
                    print("- 'info': Character info")
                    print("- 'memory': Show conversation history")
                    print("- 'emotions': Show character's current emotional state")
                else:
                    print(f"Character '{character}' not found. Please choose from the available list.")
                    continue
            
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'switch':
                self.active_character = None
                continue
            elif user_input.lower() == 'info':
                print(self.get_character_summary())
                continue
            elif user_input.lower() == 'memory':
                print("\nConversation history:")
                for i, (speaker, text) in enumerate(self.conversation_history[-5:]):
                    print(f"{i+1}. {speaker}: {text[:60]}{'...' if len(text) > 60 else ''}")
                continue
            elif user_input.lower() == 'emotions':
                if self.active_character in self.character_states:
                    emotions = self.character_states[self.active_character]['current_emotions']
                    print("\nCurrent emotional state:")
                    for emotion, value in emotions.items():
                        if value > 0.1:
                            print(f"- {emotion}: {value:.2f}")
                else:
                    print("No emotional state data available.")
                continue
            
            response = self.generate_response(user_input)
            print(f"\n{self.active_character}: {response}")

if __name__ == "__main__":
    chatbot = GOTCharacterChatbot()
    chatbot.run_interactive()
