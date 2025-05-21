#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:29:02 2025

@author: jneimann
"""

import pandas as pd

# Load the Game of Thrones script CSV file
got_file_path = "Game_of_Thrones_Script.csv"
got_script_df = pd.read_csv(got_file_path)

# Standardize character names
got_script_df["Name"] = got_script_df["Name"].astype(str).str.title().str.strip()
got_script_df = got_script_df[got_script_df["Name"] != "Nan"]

# Define main characters
got_main_characters = {
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
got_main_dialogue_df = got_script_df[got_script_df["Name"].isin(got_main_characters)]

# Save the cleaned dataset
got_main_dialogue_df.to_csv("Game_of_Thrones_Main_Characters_Dialogue.csv", index=False)

# Prevent auto-execution when importing
if __name__ == "__main__":
    print(f"✅ Extracted {got_main_dialogue_df.shape[0]} lines of dialogue for {got_main_dialogue_df['Name'].nunique()} main characters.")
    print("Cleaned dataset saved as 'Game_of_Thrones_Main_Characters_Dialogue.csv'.")

