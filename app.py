import streamlit as st
import pandas as pd
import os
import uuid
from PIL import Image
import pytesseract
from translator import process_input, generate_audio
import random
import datetime

DB_FILE = 'vocab_db.csv'

HIRAGANA_DICT = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "wo", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po"
}

HIRAGANA_GROUPS = {
    "A-row (a, i, u, e, o)": ["あ", "い", "う", "え", "お"],
    "K-row (ka, ki, ... )": ["か", "き", "く", "け", "こ"],
    "S-row (sa, shi, ... )": ["さ", "し", "す", "せ", "そ"],
    "T-row (ta, chi, ... )": ["た", "ち", "つ", "て", "と"],
    "N-row (na, ni, ... )": ["な", "に", "ぬ", "ね", "の"],
    "H-row (ha, hi, ... )": ["は", "ひ", "ふ", "へ", "ほ"],
    "M-row (ma, mi, ... )": ["ま", "み", "む", "め", "も"],
    "Y-row (ya, yu, yo)": ["や", "ゆ", "よ"],
    "R-row (ra, ri, ... )": ["ら", "り", "る", "れ", "ろ"],
    "W/N-row (wa, wo, n)": ["わ", "を", "ん"],
    "G-row (ga, gi, ...)": ["が", "ぎ", "ぐ", "げ", "ご"],
    "Z-row (za, ji, ...)": ["ざ", "じ", "ず", "ぜ", "ぞ"],
    "D-row (da, ji, ...)": ["だ", "ぢ", "づ", "で", "ど"],
    "B-row (ba, bi, ...)": ["ば", "び", "ぶ", "べ", "ぼ"],
    "P-row (pa, pi, ...)": ["ぱ", "ぴ", "ぷ", "ぺ", "ぽ"]
}

KATAKANA_DICT = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヲ": "wo", "ン": "n",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po"
}

KATAKANA_GROUPS = {
    "A-row (a, i, u, e, o)": ["ア", "イ", "ウ", "エ", "オ"],
    "K-row (ka, ki, ... )": ["カ", "キ", "ク", "ケ", "コ"],
    "S-row (sa, shi, ... )": ["サ", "シ", "ス", "セ", "ソ"],
    "T-row (ta, chi, ... )": ["タ", "チ", "ツ", "テ", "ト"],
    "N-row (na, ni, ... )": ["ナ", "ニ", "ヌ", "ネ", "ノ"],
    "H-row (ha, hi, ... )": ["ハ", "ヒ", "フ", "ヘ", "ホ"],
    "M-row (ma, mi, ... )": ["マ", "ミ", "ム", "メ", "モ"],
    "Y-row (ya, yu, yo)": ["ヤ", "ユ", "ヨ"],
    "R-row (ra, ri, ... )": ["ラ", "リ", "ル", "レ", "ロ"],
    "W/N-row (wa, wo, n)": ["ワ", "ヲ", "ン"],
    "G-row (ga, gi, ...)": ["ガ", "ギ", "グ", "ゲ", "ゴ"],
    "Z-row (za, ji, ...)": ["ザ", "ジ", "ズ", "ゼ", "ゾ"],
    "D-row (da, ji, ...)": ["ダ", "ヂ", "ヅ", "デ", "ド"],
    "B-row (ba, bi, ...)": ["バ", "ビ", "ブ", "ベ", "ボ"],
    "P-row (pa, pi, ...)": ["パ", "ピ", "プ", "ペ", "ポ"]
}

def init_db():
    """Initialize the vocabulary database if it doesn't exist."""
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=[
            'id', 
            'english', 
            'kanji', 
            'kana', 
            'romaji', 
            'review_score',
            'next_review_date',
            'interval',
            'ease_factor'
        ])
        df.to_csv(DB_FILE, index=False)

def load_data():
    """Load vocabulary data into a Pandas DataFrame."""
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        changed = False
        if 'next_review_date' not in df.columns:
            df['next_review_date'] = datetime.date.today().isoformat()
            changed = True
        if 'interval' not in df.columns:
            df['interval'] = 0
            changed = True
        if 'ease_factor' not in df.columns:
            df['ease_factor'] = 2.5
            changed = True
        if changed:
            df.to_csv(DB_FILE, index=False)
        return df
    return pd.DataFrame()

def save_entry(english, kanji, kana, romaji):
    """Save a new vocabulary entry to the database."""
    df = load_data()
    new_row = pd.DataFrame([{
        'id': str(uuid.uuid4()),
        'english': english,
        'kanji': kanji,
        'kana': kana,
        'romaji': romaji,
        'review_score': 0,
        'next_review_date': datetime.date.today().isoformat(),
        'interval': 0,
        'ease_factor': 2.5
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

def update_score(entry_id, score_change):
    """Update the review score for a specific vocabulary entry."""
    df = load_data()
    mask = df['id'] == entry_id
    if mask.any():
        current_score = df.loc[mask, 'review_score'].values[0]
        df.loc[mask, 'review_score'] = current_score + score_change
        df.to_csv(DB_FILE, index=False)

def update_sm2(entry_id, quality):
    """
    SM-2 Spaced Repetition Algorithm.
    quality: 1 (Again), 3 (Hard), 4 (Good), 5 (Easy)
    """
    df = load_data()
    mask = df['id'] == entry_id
    if mask.any():
        interval = df.loc[mask, 'interval'].values[0]
        ease_factor = df.loc[mask, 'ease_factor'].values[0]
        review_score = df.loc[mask, 'review_score'].values[0]
        
        if quality < 3: # failed
            interval = 0
            ease_factor = max(1.3, ease_factor - 0.2)
        else:
            if interval == 0:
                interval = 1
            elif interval == 1:
                interval = 6
            else:
                interval = int(round(interval * ease_factor))
                
            ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            ease_factor = max(1.3, ease_factor)
            
        next_review = datetime.date.today() + datetime.timedelta(days=interval)
        
        df.loc[mask, 'interval'] = interval
        df.loc[mask, 'ease_factor'] = ease_factor
        df.loc[mask, 'next_review_date'] = next_review.isoformat()
        
        # legacy review_score for dashboard stats
        if quality >= 3:
            df.loc[mask, 'review_score'] = review_score + 1
        elif review_score > 0:
            df.loc[mask, 'review_score'] = review_score - 1
            
        df.to_csv(DB_FILE, index=False)

def render_practice_section(title, char_dict, char_groups, session_prefix):
    st.write(f"### {title}")
    
    deck_key = f"{session_prefix}_deck"
    show_ans_key = f"{session_prefix}_show_ans"
    quiz_msg_key = f"{session_prefix}_quiz_msg"
    ans_input_key = f"{session_prefix}_ans_input"
    
    with st.expander("Practice Settings", expanded=True):
        group_keys = list(char_groups.keys())
        col1, col2 = st.columns(2)
        with col1:
            basic_groups = st.multiselect(
                "Basic Characters:", 
                group_keys[:10], 
                default=[group_keys[0]]
            )
        with col2:
            dakuten_groups = st.multiselect(
                "Dakuten & Handakuten:", 
                group_keys[10:], 
                default=[]
            )
        selected_groups = basic_groups + dakuten_groups
        mode = st.radio("Practice Mode:", ["Type Answer (Strict)", "Self-Graded (Standard Flashcards)"], key=f"{session_prefix}_mode")
        
        if st.button("Start / Reset Deck", key=f"{session_prefix}_reset"):
            chars = []
            for g in selected_groups:
                chars.extend(char_groups[g])
            st.session_state[deck_key] = chars
            random.shuffle(st.session_state[deck_key])
            st.session_state[show_ans_key] = False
            st.session_state[quiz_msg_key] = ""
            st.session_state[f"{session_prefix}_balloons_shown"] = False
    
    if deck_key in st.session_state:
        deck = st.session_state[deck_key]
        if not deck:
            st.success("You've mastered all the characters in this deck! Great job!")
            if not st.session_state.get(f"{session_prefix}_balloons_shown", False):
                st.balloons()
                st.session_state[f"{session_prefix}_balloons_shown"] = True
        else:
            current_char = deck[0]
            correct_romaji = char_dict[current_char]
            
            st.write(f"**Cards remaining in deck:** {len(deck)}")
            st.markdown(f"<h1 class='flashcard-char'>{current_char}</h1>", unsafe_allow_html=True)
            
            if mode == "Type Answer (Strict)":
                def check_ans():
                    user_ans = st.session_state[ans_input_key].strip().lower()
                    if user_ans == correct_romaji:
                        st.session_state[quiz_msg_key] = "Correct!"
                        st.session_state[deck_key].pop(0) # Remove from deck
                    else:
                        st.session_state[quiz_msg_key] = f"Incorrect! {current_char} is '{correct_romaji}'. Moved to back of the deck."
                        # Move to back of deck
                        st.session_state[deck_key].append(st.session_state[deck_key].pop(0))
                    st.session_state[ans_input_key] = "" # Clear input
                
                if st.session_state.get(quiz_msg_key):
                    if "Correct" in st.session_state[quiz_msg_key]:
                        st.success(st.session_state[quiz_msg_key])
                    else:
                        st.error(st.session_state[quiz_msg_key])
                    st.session_state[quiz_msg_key] = ""
                    
                st.text_input("Type the Romaji:", key=ans_input_key, on_change=check_ans)
                st.button("Submit", on_click=check_ans, use_container_width=True)
                
                # JavaScript injection to keep the keyboard open (autofocus the input)
                st.components.v1.html(
                    """
                    <script>
                        const input = window.parent.document.querySelector('input[type="text"]');
                        if (input) {
                            input.focus();
                        }
                    </script>
                    """,
                    height=0,
                    width=0,
                )
            
            else: # Self-Graded
                if not st.session_state.get(show_ans_key, False):
                    col1, col2, col3 = st.columns([1,1,1])
                    with col2:
                        if st.button("Reveal Answer", use_container_width=True, key=f"{session_prefix}_reveal"):
                            st.session_state[show_ans_key] = True
                            st.rerun()
                else:
                    st.markdown(f"<h2 style='text-align: center; color: #4B4B4B;'>{correct_romaji}</h2>", unsafe_allow_html=True)
                    st.write("### Did you know it?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes (Remove from deck)", use_container_width=True, key=f"{session_prefix}_yes"):
                            st.session_state[deck_key].pop(0)
                            st.session_state[show_ans_key] = False
                            st.rerun()
                    with col2:
                        if st.button("No (Keep in deck)", use_container_width=True, key=f"{session_prefix}_no"):
                            st.session_state[deck_key].append(st.session_state[deck_key].pop(0))
                            st.session_state[show_ans_key] = False
                            st.rerun()

def main():
    st.set_page_config(page_title="Nihongo Learner", page_icon="🎌", layout="centered")
    
    # Inject CSS for mobile optimization (hide header/footer, add padding)
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Make flashcard font size responsive on small screens */
        .flashcard-char {
            text-align: center; 
            font-size: min(80px, 25vw) !important;
        }
        
        /* Add some padding for mobile viewing */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize DB on first run
    init_db()
    
    # Initialize session state for navigation if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard / Home"
        
    st.sidebar.title("🎌 Nihongo Learner")
    
    # Sidebar navigation
    pages = ["Dashboard / Home", "Automated Entry", "Image Upload & OCR", "Study & Review", "Hiragana Practice", "Katakana Practice"]
    selection = st.sidebar.radio("Navigation", pages)
    
    st.session_state.current_page = selection

    st.title(f"{selection}")

    if selection == "Dashboard / Home":
        st.write("## Welcome to Nihongo Learner! 🎌")
        st.write("Track your Japanese learning progress below.")
        
        df = load_data()
        
        if df.empty:
            st.info("Your database is empty! Head over to 'Automated Entry' to add your first words.")
        else:
            total_words = len(df)
            reviewed_words = len(df[df['review_score'] != 0])
            mastered_words = len(df[df['review_score'] >= 3])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Vocabulary", total_words)
            col2.metric("Words Reviewed", reviewed_words)
            col3.metric("Words Mastered", mastered_words)
            
            st.markdown("### Mastery Progress")
            progress = mastered_words / total_words if total_words > 0 else 0
            st.progress(progress)
            st.write(f"{int(progress * 100)}% of your vocabulary is mastered (Score >= 3).")
            
            st.markdown("---")
            st.dataframe(df.tail(5)[['english', 'kanji', 'kana', 'romaji', 'review_score']])

            st.markdown("---")
            st.write("### Export to Anki")
            st.write("You can export your vocabulary database to a CSV format that is ready to be imported into Anki.")
            
            # Create a clean dataframe for export
            anki_df = df[['english', 'kanji', 'kana', 'romaji']].copy()
            csv_export = anki_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Anki Deck (CSV)",
                data=csv_export,
                file_name="nihongo_anki_export.csv",
                mime="text/csv",
            )

    elif selection == "Automated Entry":
        st.write("Enter an English or Romaji word/sentence, and we'll automatically generate the rest!")
        
        with st.form("auto_entry_form"):
            input_text = st.text_input("Input text (English or Romaji)")
            input_type = st.radio("What type of input is this?", ["English", "Romaji (or Japanese)"])
            
            submit_button = st.form_submit_button("Generate")
            
        if submit_button and input_text:
            with st.spinner("Processing..."):
                # Pass "English" or "Romaji" to our backend
                type_for_backend = "english" if input_type == "English" else "romaji"
                result = process_input(input_text, type_for_backend)
                
                # Store the result in session state so we don't lose it when clicking "Save"
                st.session_state.preview_data = result
                st.success("Generation complete! Check the preview below.")

        # If we have preview data in session state, show it and offer to save
        if 'preview_data' in st.session_state and st.session_state.preview_data:
            st.markdown("### Preview")
            preview = st.session_state.preview_data
            
            st.markdown(f"**English:** {preview['english']}")
            st.markdown(f"**Kana:** {preview['kana']}")
            st.markdown(f"**Romaji:** {preview['romaji']}")
            
            audio_text = preview['kanji'] if preview['kanji'] else preview['kana']
            if audio_text:
                audio_fp = generate_audio(audio_text)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3')
            
            if st.button("Save to Database"):
                save_entry(preview["english"], preview["kanji"], preview["kana"], preview["romaji"])
                st.success(f"Saved '{preview['english']}' to database!")
                # Clear preview data after saving
                st.session_state.preview_data = None
                st.rerun()
        

    elif selection == "Image Upload & OCR":
        st.write("Upload an image containing Japanese text, and we'll extract it using OCR.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            
            if st.button("Extract Text"):
                with st.spinner("Extracting Japanese text..."):
                    try:
                        # Assuming 'jpn' language pack is installed in Tesseract
                        extracted_text = pytesseract.image_to_string(image, lang='jpn')
                        st.session_state.extracted_text = extracted_text
                    except Exception as e:
                        st.error(f"Error extracting text. Ensure Tesseract is installed with 'jpn' language. Details: {e}")
                        
        if 'extracted_text' in st.session_state:
            st.markdown("### Extracted Text")
            edited_text = st.text_area("Edit text or select specific words/sentences:", 
                                       value=st.session_state.extracted_text, 
                                       height=150)
            
            st.markdown("---")
            st.write("Send to Automated Vocabulary Entry:")
            
            with st.form("ocr_to_auto_form"):
                # Use the edited text from the text area as the default
                text_to_process = st.text_input("Text to process", value=edited_text)
                
                submit_ocr = st.form_submit_button("Generate Entry")
                
            if submit_ocr and text_to_process:
                with st.spinner("Processing..."):
                    # Extracted text is Japanese
                    result = process_input(text_to_process, "japanese")
                    st.session_state.ocr_preview_data = result
                    st.success("Generation complete! Check the preview below.")

        if 'ocr_preview_data' in st.session_state and st.session_state.ocr_preview_data:
            st.markdown("### Preview")
            preview = st.session_state.ocr_preview_data
            
            st.markdown(f"**English:** {preview['english']}")
            st.markdown(f"**Kana:** {preview['kana']}")
            st.markdown(f"**Romaji:** {preview['romaji']}")
            
            audio_text = preview['kanji'] if preview['kanji'] else preview['kana']
            if audio_text:
                audio_fp = generate_audio(audio_text)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3')
            
            if st.button("Save OCR Entry to Database"):
                save_entry(preview["english"], preview["kanji"], preview["kana"], preview["romaji"])
                st.success(f"Saved '{preview['english']}' to database!")
                st.session_state.ocr_preview_data = None
                st.rerun()
        
    elif selection == "Study & Review":
        st.write("### Flashcard Study Mode")
        
        df = load_data()
        
        if df.empty:
            st.info("Your database is empty. Please add some vocabulary first using 'Automated Entry' or 'Image Upload & OCR'.")
        else:
            # Initialize study session state
            if 'study_df' not in st.session_state or st.button("Restart Session"):
                today = datetime.date.today().isoformat()
                due_cards = df[df['next_review_date'] <= today]
                st.session_state.study_df = due_cards.reset_index(drop=True)
                st.session_state.current_card_index = 0
                st.session_state.show_answer = False
                
            study_df = st.session_state.study_df
            
            if len(study_df) == 0:
                st.success("You have no cards due for review today! Come back tomorrow.")
            elif st.session_state.current_card_index >= len(study_df):
                st.success("You have reviewed all due cards in this session!")
                # Button to restart is already at the top
            else:
                current_card = study_df.iloc[st.session_state.current_card_index]
                
                # Progress indicator
                st.progress((st.session_state.current_card_index) / len(study_df))
                st.write(f"Card {st.session_state.current_card_index + 1} of {len(study_df)}")
                
                # Card UI
                st.markdown("---")
                st.markdown(f"<h1 class='flashcard-char'>{current_card['english']}</h1>", unsafe_allow_html=True)
                st.markdown("---")
                
                if not st.session_state.show_answer:
                    # Centered button
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        if st.button("Reveal Answer", use_container_width=True):
                            st.session_state.show_answer = True
                            st.rerun()
                else:
                    st.markdown(f"**Kana:** {current_card['kana']}")
                    st.markdown(f"**Romaji:** {current_card['romaji']}")
                    st.markdown(f"**Current Review Score:** {current_card['review_score']}")
                    
                    audio_text = current_card['kanji'] if pd.notna(current_card['kanji']) and current_card['kanji'] else current_card['kana']
                    if audio_text:
                        audio_fp = generate_audio(audio_text)
                        if audio_fp:
                            st.audio(audio_fp, format='audio/mp3')
                    
                    st.write("### How did you do?")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("Again (Fail)", use_container_width=True):
                            update_sm2(current_card['id'], 1)
                            st.session_state.current_card_index += 1
                            st.session_state.show_answer = False
                            st.rerun()
                            
                    with col2:
                        if st.button("Hard", use_container_width=True):
                            update_sm2(current_card['id'], 3)
                            st.session_state.current_card_index += 1
                            st.session_state.show_answer = False
                            st.rerun()
                            
                    with col3:
                        if st.button("Good", use_container_width=True):
                            update_sm2(current_card['id'], 4)
                            st.session_state.current_card_index += 1
                            st.session_state.show_answer = False
                            st.rerun()
                            
                    with col4:
                        if st.button("Easy", use_container_width=True):
                            update_sm2(current_card['id'], 5)
                            st.session_state.current_card_index += 1
                            st.session_state.show_answer = False
                            st.rerun()
                            
        st.markdown("---")
        with st.expander("View Full Database"):
            st.dataframe(load_data())
            
    elif selection == "Hiragana Practice":
        render_practice_section("Hiragana Practice", HIRAGANA_DICT, HIRAGANA_GROUPS, "hira")
        
    elif selection == "Katakana Practice":
        render_practice_section("Katakana Practice", KATAKANA_DICT, KATAKANA_GROUPS, "kata")

if __name__ == "__main__":
    main()
