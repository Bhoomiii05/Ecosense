import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from rapidfuzz import process, fuzz

# --- 1. Page Configuration ---
# Set config at the very top
st.set_page_config(page_title="EcoScnce | Predictor", page_icon="🌿", layout="centered")

# --- 2. Custom CSS ---
st.markdown("""
<style>
    body { 
        background: linear-gradient(135deg, #f8fbff 0%, #e6f0ff 100%);
        font-family: 'Segoe UI', sans-serif; 
    }
    .header { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        background: linear-gradient(135deg, #004080 0%, #0078d7 100%); 
        padding: 20px 40px; 
        border-radius: 16px; 
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 30px;
    }
    .header h1 { 
        color: white; 
        margin: 0; 
        font-size: 28px;
        font-weight: bold;
    }
    .logo { 
        width: 55px; 
        height: 55px; 
        border-radius: 50%; 
    }
    .main-box { 
        background-color: white; 
        border-radius: 20px; 
        padding: 40px; 
        margin-top: 20px; 
        box-shadow: 0px 6px 16px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .input-label { 
        font-size: 20px !important; 
        font-weight: bold !important; 
        color: #004080 !important; 
        margin-bottom: 10px;
    }
    .stTextInput input {
        font-size: 18px !important;
        padding: 15px !important;
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
    }
    .stButton button {
        font-size: 18px !important;
        font-weight: bold;
        border-radius: 12px !important; 
        padding: 12px 30px !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    /* Style for individual correction buttons (selecting word) */
    .stButton .correction-button {
        background-color: #e6f0ff;
        color: #004080;
        border: 2px solid #0078d7;
        font-size: 16px !important;
        font-weight: normal !important;
        padding: 8px 12px !important;
    }
    .stButton .correction-button:hover {
        background-color: #0078d7;
        color: white;
    }
    /* Style for 'Yes'/'Go Back' buttons */
    .stButton .confirm-button-yes {
        background-color: #e8fbe8;
        color: #008000;
        border: 2px solid #00cc44;
        font-size: 16px !important;
        font-weight: bold !important;
        padding: 8px 12px !important;
    }
    .stButton .confirm-button-yes:hover {
        background-color: #00cc44;
        color: white;
    }
    .stButton .confirm-button-back {
        background-color: #fff;
        color: #555;
        border: 2px solid #ccc;
        font-size: 16px !important;
        font-weight: normal !important;
        padding: 8px 12px !important;
    }
    .stButton .confirm-button-back:hover {
        background-color: #f0f0f0;
        border-color: #999;
    }

    /* Style for cancel/keep buttons */
    .stButton .control-button-keep {
        background-color: #e0e0e0;
        color: #333;
        border: 2px solid #999;
        font-size: 16px !important;
        font-weight: normal !important;
    }
    .stButton .control-button-keep:hover {
        background-color: #ccc;
    }
    .stButton .control-button-cancel {
        background-color: #ffe6e6;
        color: #cc0000;
        border: 2px solid #ff3333;
        font-size: 16px !important;
        font-weight: normal !important;
    }
    .stButton .control-button-cancel:hover {
        background-color: #ff3333;
        color: white;
    }
    
    .suggestions-box {
        background-color: #f8fbff;
        border-radius: 12px;
        padding: 15px 20px;
        margin: 15px 0;
        border-left: 5px solid #0078d7;
    }
    .success-box { 
        background-color: #e8fbe8; 
        border-left: 6px solid #00cc44; 
        padding: 25px; 
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
    }
    .error-box { 
        background-color: #ffe6e6; 
        border-left: 6px solid #ff3333; 
        padding: 25px; 
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. Load Assets & Build Vocab ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("sustainable_Dataset.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'sustainable_Dataset.csv' not found. Make sure it's in the same folder.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def build_master_vocab(_df):
    """Builds a master vocabulary from all text columns for spell checking."""
    if _df is None:
        return ["plastic", "bottle", "sustainable"] # Fallback
        
    vocab = set()
    text_columns = ["Name", "Components Used", "Packaging"]
    
    for col in text_columns:
        if col in _df.columns:
            for cell in _df[col].dropna():
                for t in re.findall(r"[a-zA-Z]+", str(cell).lower()):
                    if len(t) > 3: 
                        vocab.add(t)
    
    vocab.update(["plastic", "sustainable", "environment", "recyclable", 
                    "biodegradable", "metal", "glass", "paper", "wood",
                    "rubber", "cotton", "silk", "bamboo", "ceramic",
                    "bottle", "container", "package", "bag"])
    
    return list(vocab)

@st.cache_data
def get_product_categories(_df):
    """Extract significant category-like words from product names"""
    if _df is None:
        return []
        
    categories = set()
    ignore_words = {"a", "an", "the", "with", "for", "in", "of", "and", "or", "new", "eco"}
    for name in _df["Name"].dropna():
        words = re.findall(r"[a-zA-Z]+", str(name).lower())
        for word in words:
            if len(word) > 3 and word not in ignore_words:
                categories.add(word)
    return list(categories)

# --- Load data on script run ---
df = load_data()
if df is not None:
    MASTER_VOCAB = build_master_vocab(df)
    CATEGORIES = get_product_categories(df)
else:
    MASTER_VOCAB = ["plastic", "bottle", "sustainable"]
    CATEGORIES = ["bottle"]


# --- 4. Helper Functions ---

# --- NEW: Keyword definitions for overriding logic ---
PENALTY_WORDS = {'plastic', 'nylon', 'polystyrene', 'non-recyclable', 'disposable', 'single-use', 'pvc'}
BONUS_WORDS = {'reusable', 'bamboo', 'glass', 'metal', 'recycled', 'compostable', 'biodegradable', 'wood', 'cotton', 'stainless-steel', 'cork'}


def tokenize_text(s):
    return [t for t in re.findall(r"[a-zA-Z]+", str(s).lower()) if len(t) >= 2]

def enhanced_spell_check(text, master_vocab_list, category_list):
    """
    Finds typos and category suggestions.
    Returns: (correction_map, category_suggestions, note)
    - correction_map: List of tuples [('original', ['sugg1', 'sugg2']), ...]
    - category_suggestions: List of strings ["'word' (did you mean: ...)", ...]
    - note: String "Auto-corrected | Category suggestions"
    """
    original_text = text.strip().lower()
    # Use regex to find all words, including those with apostrophes
    words = re.findall(r"[a-zA-Z']+", original_text) 
    
    correction_map = [] # List of ('original', [list_of_suggestions])
    category_suggestions = [] # List of strings
    note_parts = []
    
    master_vocab_set = set(master_vocab_list)
    # Keep track of words we've already offered a correction for
    corrected_originals = set()

    for word in words:
        if word in master_vocab_set or len(word) < 3 or word in corrected_originals:
            continue 

        # 1. Find multiple direct typo corrections
        # Use process.extract to get a few good suggestions
        suggestions = process.extract(
            word, 
            master_vocab_list, 
            scorer=fuzz.WRatio, 
            limit=3, # Get top 3 suggestions
            score_cutoff=80 # Lowered cutoff to get more options
        )
        
        # suggestions is a list of tuples: [(match, score, index)]
        # We only want the matches, and only if they are different from the word
        valid_suggestions = [match for match, score, _ in suggestions if match != word]
        
        if valid_suggestions:
            # Found one or more suggestions
            correction_map.append((word, valid_suggestions))
            corrected_originals.add(word) 
            if "Auto-corrected" not in note_parts:
                note_parts.append("Auto-corrected")
        else:
            # 2. No typo found, check for category hints
            category_matches = process.extract(
                word, 
                category_list, 
                scorer=fuzz.partial_ratio, 
                limit=2,
                score_cutoff=80
            )
            
            cat_suggestions = []
            for cat_match, cat_score, _ in category_matches:
                if cat_match != word and cat_match not in word and word not in cat_match:
                    cat_suggestions.append(cat_match)
                        
            if cat_suggestions:
                unique_suggestions = list(set(cat_suggestions))
                category_suggestions.append(f"'{word}' (did you mean: {', '.join(unique_suggestions)}?)")
                if "Category suggestions" not in note_parts:
                        note_parts.append("Category suggestions")

    note = " | ".join(note_parts)
    
    # Return map of corrections, list of category hints, and note
    return correction_map, category_suggestions, note


def find_best_match(user_text, df):
    """Find best matching product with category awareness"""
    if df is None:
        return None
        
    choices = df["Name"].dropna().tolist()
    if not choices:
        return None
    
    match, score, _ = process.extractOne(user_text, choices, scorer=fuzz.partial_ratio)
    
    if score > 60:
        matched_product = df[df["Name"] == match].iloc[0]
        
        user_words = set(user_text.lower().split())
        product_words = set(match.lower().split())
        
        common_categories = {'plastic', 'metal', 'glass', 'paper', 'wood', 
                           'rubber', 'cotton', 'silk', 'bamboo', 'ceramic'}
        
        user_categories = user_words & common_categories
        product_categories = product_words & common_categories
        
        if user_categories and product_categories and not (user_categories & product_categories):
            if score < 80:
                return None
        
        return matched_product
    else:
        return None

# --- NEW: Function to apply override logic ---
def get_prediction_details(user_text, best_match_row):
    """
    Analyzes the user's text and the matched product to give a final
    prediction, including overrides.
    
    Returns: (final_level, final_alternative, match_name, reason)
    """
    
    # Get the base prediction from the matched product
    original_level = best_match_row["Sustainability_Level"]
    final_alternative = best_match_row["Sustainable_Alternative"]
    match_name = best_match_row["Name"]
    
    # Start with the base level
    final_level = original_level
    
    # Analyze user's text for overrides
    user_tokens = set(tokenize_text(user_text))
    
    has_penalty_word = bool(user_tokens & PENALTY_WORDS)
    has_bonus_word = bool(user_tokens & BONUS_WORDS)
    
    # Default reason
    reason = f"Based on similarity to: **{match_name}**"
    
    # --- Override Logic ---
    current_level_low = str(original_level).strip().lower()
    
    if current_level_low == "high" or current_level_low == "medium":
        # Check if we should DOWNGRADE
        if has_penalty_word and not has_bonus_word: # e.g., "plastic bottle"
            penalty_word_found = (user_tokens & PENALTY_WORDS).pop()
            final_level = "Low" # Downgrade!
            reason = f"Matched '{match_name}', but **downgraded to 'Low'** because you specified a material like **'{penalty_word_found}'**."
            # Suggest a new alternative
            if 'plastic' in user_tokens:
                 final_alternative = "Consider a reusable metal or glass bottle instead."
            elif 'nylon' in user_tokens:
                 final_alternative = "Consider bags made from cotton or recycled materials."

    elif current_level_low == "low" or current_level_low == "medium":
        # Check if we should UPGRADE
        if has_bonus_word and not has_penalty_word: # e.g., "reusable metal bottle"
            bonus_word_found = (user_tokens & BONUS_WORDS).pop()
            final_level = "High" # Upgrade!
            reason = f"Matched '{match_name}' (which is normally {original_level}), but **upgraded to 'High'** because you specified **'{bonus_word_found}'**."

    return final_level, final_alternative, match_name, reason


# --- 5. Streamlit App UI ---

# --- Header Section ---
col1, col2 = st.columns([8, 1])
with col1:
    st.markdown('<div class="header"><h1>🌿 EcoScnce | Sustainability Predictor</h1></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="header"><span style="font-size: 55px;">🌿</span></div>', unsafe_allow_html=True)

st.markdown(
    "Enter a product description to predict if it's sustainable based on similar products in our database."
)

# --- Input Section ---
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# --- Initialize Session State ---
if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False
if "skip_suggestions" not in st.session_state:
    st.session_state.skip_suggestions = False
if "user_description" not in st.session_state:
    st.session_state.user_description = "a plastc botle" # Default
if "word_to_correct" not in st.session_state:
    # This new state stores the word the user has selected
    st.session_state.word_to_correct = None 
if "correction_map" not in st.session_state:
    st.session_state.correction_map = []

# --- FIX: Apply pending text update BEFORE rendering the widget ---
# Check if a correction was just applied in the previous run
if "new_text" in st.session_state:
    st.session_state.user_description = st.session_state.new_text # Set the main state
    del st.session_state.new_text # Clean up the temporary variable

# User Input
st.markdown('<p class="input-label">Product Description:</p>', unsafe_allow_html=True)
# The text_input widget's value is now controlled by session_state
# This will now correctly use the updated st.session_state.user_description
user_description = st.text_input(
    " ", 
    key="user_description", 
    label_visibility="collapsed"
)

# --- Predict Button ---
if st.button("🔍 Predict", key="predict", use_container_width=True):
    st.session_state.run_prediction = True
    st.session_state.skip_suggestions = False # Reset skip flag
    st.session_state.word_to_correct = None # Reset word selection
    st.rerun() 

# --- Main Prediction Logic Block ---
if st.session_state.run_prediction:
    
    if df is None:
        st.error("Application is not ready. Data file could not be loaded.")
        st.session_state.run_prediction = False
    elif not user_description.strip():
        st.warning("Please enter a product description.")
        st.session_state.run_prediction = False
    else:
        # 1. Run Spell Check ONLY if we don't have a map already
        if not st.session_state.word_to_correct:
            correction_map, category_suggestions, spell_note = enhanced_spell_check(
                user_description, MASTER_VOCAB, CATEGORIES
            )
            st.session_state.correction_map = correction_map
        else:
            # We are in the middle of a correction, use the stored map
            correction_map = st.session_state.correction_map
            category_suggestions = [] # Don't show category hints while confirming a word
        
        # We show suggestions if there are corrections to be made AND
        # the user has not clicked "Keep Original"
        show_suggestion_box = bool(correction_map) and not st.session_state.skip_suggestions
        
        if show_suggestion_box:
            # --- DISPLAY SUGGESTION BOX ---
            st.markdown('<div class="suggestions-box">', unsafe_allow_html=True)
            
            # Check if user has selected a word to correct
            if st.session_state.word_to_correct is None:
                # --- Step 1: Show list of misspelled words ---
                st.subheader("💡 Select a word to correct:")
                
                num_cols = min(len(correction_map), 3) # Max 3 buttons per row
                if num_cols > 0:
                    cols = st.columns(num_cols)
                    for i, (original, corrected_list) in enumerate(correction_map):
                        col = cols[i % num_cols]
                        with col:
                            button_key = f"select_{original}"
                            # Show only the original word on the button
                            if st.button(original, key=button_key, use_container_width=True):
                                # User selected this word. Store it and rerun.
                                st.session_state.word_to_correct = original
                                st.rerun()
                
                # Show category suggestions as text (if any)
                if category_suggestions:
                    st.write("**Other suggestions:**")
                    for suggestion in category_suggestions:
                        st.write(f"• {suggestion}")

            else:
                # --- Step 2: Show suggestions for the selected word ---
                selected_original = st.session_state.word_to_correct
                # Find the correction list from the map
                suggestion_list = []
                for o, s_list in st.session_state.correction_map: # Use stored map
                    if o == selected_original:
                        suggestion_list = s_list
                        break
                
                if suggestion_list:
                    st.subheader(f"Select a correction for '{selected_original}':")
                    
                    # Create buttons for each suggestion
                    num_cols = min(len(suggestion_list), 3)
                    if num_cols > 0:
                        cols = st.columns(num_cols)
                        for i, corrected_word in enumerate(suggestion_list):
                            col = cols[i % num_cols]
                            with col:
                                if st.button(corrected_word, key=f"apply_{corrected_word}", use_container_width=True):
                                    # --- Apply this correction ---
                                    current_text = st.session_state.user_description
                                    new_text = re.sub(
                                        r'\b' + re.escape(selected_original) + r'\b', 
                                        corrected_word, 
                                        current_text, 
                                        count=1, 
                                        flags=re.IGNORECASE
                                    )
                                    
                                    # --- FIX: Use temp state var ---
                                    st.session_state.new_text = new_text 
                                    st.session_state.word_to_correct = None # Reset selection
                                    st.session_state.correction_map = [] # Clear map to force re-check
                                    st.rerun()
                    
                    st.markdown("---") # Divider
                    # Add a 'Go Back' button
                    if st.button("⬅️ Go Back (Select different word)", key="confirm_back", use_container_width=True):
                        st.session_state.word_to_correct = None # Reset selection
                        st.rerun()
                else:
                    # Failsafe in case state gets weird
                    st.session_state.word_to_correct = None
                    st.rerun()

            st.markdown("---") # Divider

            # Add control buttons (Keep Original / Cancel)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("➡️ Keep Original and Predict", key="keep_original_predict", use_container_width=True):
                    st.session_state.skip_suggestions = True
                    st.session_state.word_to_correct = None
                    st.rerun()
            with col2:
                if st.button("❌ Cancel Correction", key="cancel_suggestions", use_container_width=True):
                    st.session_state.run_prediction = False
                    st.session_state.skip_suggestions = False
                    st.session_state.word_to_correct = None
                    # --- FIX: Use temp state var ---
                    st.session_state.new_text = "a plastc botle" # Reset to default
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # --- PROCEED WITH PREDICTION ---
            # This runs if:
            # 1. No corrections were found (correction_map is empty)
            # 2. User clicked "Keep Original and Predict"
            
            use_text = st.session_state.user_description 
            st.info(f"**Analyzing:** '{use_text}'")
            
            best_match = find_best_match(use_text, df)

            if best_match is None:
                st.error(
                    "❌ Sorry, I couldn't find a similar product in my database.\n\n"
                    "**Tips:**\n"
                    "• Be specific about materials (plastic, metal, glass, etc.)\n"
                    "• Use common product names\n"
                )
            else:
                level_col = "Sustainability_Level"
                alternative_col = "Sustainable_Alternative"
                name_col = "Name"

                if (level_col not in best_match.index or 
                    alternative_col not in best_match.index or 
                    name_col not in best_match.index):
                    st.error("Error: Missing required columns in the dataset.")
                else:
                    # --- MODIFIED: Use new logic function ---
                    level, alternative, match_name, reason = get_prediction_details(use_text, best_match)

                    st.markdown("---")
                    st.subheader("🎯 Prediction Result:")

                    if str(level).strip().lower() == "high":
                        st.balloons()
                        st.markdown('<div class="success-box">✅ Yes, likely sustainable!</div>', unsafe_allow_html=True)
                        st.markdown(reason) # Use the new dynamic reason
                        st.markdown(f"**Original Match Level:** {best_match[level_col]}")
                        
                        st.write("### 🌍 Eco Score (Estimated)")
                        st.progress(0.9)
                        st.success("High Sustainability Score: 90%")
                    else:
                        st.markdown('<div class="error-box">❌ No, likely not sustainable.</div>', unsafe_allow_html=True)
                        st.markdown(reason) # Use the new dynamic reason
                        st.markdown(f"**Original Match Level:** {best_match[level_col]}")
                        
                        st.write("### 🌍 Eco Score (Estimated)")
                        st.progress(0.4)
                        st.error("Low Sustainability Score: 40%")

                        st.markdown("---")
                        st.subheader("💡 Suggested Sustainable Alternative:")
                        st.info(f"**{alternative}**")

            # Reset flags so the prediction doesn't run in a loop
            st.session_state.run_prediction = False
            st.session_state.skip_suggestions = False
            st.session_state.word_to_correct = None

st.markdown("</div>", unsafe_allow_html=True) # Close main-box

# --- Refresh Button ---
if st.button("🔄 Start Over", key="refresh", use_container_width=True):
    st.session_state.run_prediction = False
    st.session_state.skip_suggestions = False
    st.session_state.word_to_correct = None
    # --- FIX: Use temp state var ---
    st.session_state.new_text = "a plastc botle" # Reset to default
    st.rerun()

