import pandas as pd
import numpy as np
import time
import hashlib
import concurrent.futures
from functools import lru_cache
from deep_translator import GoogleTranslator
from tqdm.auto import tqdm
import os
import signal
import sys
from collections import defaultdict
from langdetect import detect, LangDetectException
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')

def is_english(text, debug=True):
    """Check if text is in English with debugging"""
    try:
        if not text or text.strip() == "":
            if debug:
                print("Empty text")
            return False
        
        # Check for common Japanese characters
        japanese_chars = set('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん')
        text_chars = set(text)
        if japanese_chars.intersection(text_chars):
            if debug:
                print("Text contains Japanese characters")
                print(f"Text sample: {text[:100]}...")
            return False
            
        # Check for other non-Latin scripts that might cause issues
        non_latin_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
                             '一二三四五六七八九十百千万亿元年月日时分秒'
                             'ضصثقفغعهخحجشسيبلاتنمكطظزوةىرؤءذد')
        if non_latin_chars.intersection(text_chars):
            if debug:
                print("Text contains non-Latin characters")
                print(f"Text sample: {text[:100]}...")
            return False
            
        # Use langdetect as a final check
        lang = detect(text)
        if debug and lang != 'en':
            print(f"Detected language: {lang}")
            print(f"Text sample: {text[:100]}...")
        return lang == 'en'
    except LangDetectException as e:
        if debug:
            print(f"Language detection failed: {e}")
            print(f"Text sample: {text[:100]}...")
        return False

def calculate_similarity(original, translated, debug=True):
    """Calculate BLEU score between original and translated text with debugging"""
    try:
        reference = [word_tokenize(original.lower())]
        candidate = word_tokenize(translated.lower())
        bleu_score = sentence_bleu(reference, candidate)
        
        if debug and bleu_score < 0.1:  # Reduced threshold for debugging
            print(f"Low BLEU score: {bleu_score}")
            print(f"Original: {original[:100]}...")
            print(f"Translated: {translated[:100]}...")
            
        return bleu_score
    except Exception as e:
        if debug:
            print(f"BLEU calculation failed: {e}")
        return 0

def validate_translation(original, translated, min_bleu=0.1, debug=True):
    """Validate that the translation is acceptable with more lenient criteria"""
    if not translated or pd.isna(translated):
        if debug:
            print("Empty or NaN translation")
        return False
    
    # Basic length check
    if len(translated.strip()) < 5:
        if debug:
            print("Translation too short")
        return False
    
    # Strict check to ensure result is in English
    if not is_english(translated, debug):
        if debug:
            print("Non-English translation")
        return False
    
    # Check if translation is too different from original
    similarity = calculate_similarity(original, translated, debug)
    if similarity < min_bleu:
        if debug:
            print(f"Translation too different (BLEU score: {similarity})")
        return False
    
    # Check if translation is not just the same as original
    if translated.strip().lower() == original.strip().lower():
        if debug:
            print("Translation identical to original")
        return False
    
    # Additional check for suspicious content (very short words or repetitive patterns)
    words = translated.split()
    if len(words) > 3:
        avg_word_len = sum(len(word) for word in words) / len(words)
        if avg_word_len < 2:
            if debug:
                print("Translation has suspiciously short words")
            return False
    
    return True

# Create a cache directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'translation_cache'), exist_ok=True)

# Global flag to indicate if the script should stop
should_stop = False

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global should_stop
    print("\n\nGraceful shutdown initiated. Saving progress...")
    should_stop = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Load the undersampled dataset
df = pd.read_csv('data/merged_data/undersampled_dataset.csv')

categories = ['toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit',
              'identity_attack', 'insult', 'threat']

# Assuming the text column is named 'comment_text' - adjust if needed
text_column = 'comment_text'

# CONFIGURATION PARAMETERS
# Number of least represented categories to balance
NUM_CATEGORIES_TO_BALANCE = 4
# Target multiplication factor (2 = double the samples)
TARGET_FACTOR = 10
# Languages to use for back-translation (languages that translate better to English)
INTERMEDIATE_LANGUAGES = ['fr', 'de', 'es', 'it']  # French, German, Spanish, Italian
# Number of workers for parallel processing - reduced to avoid rate limiting
MAX_WORKERS = 3  # Reduced from 10 to 3 to avoid hitting API rate limits
# Maximum chunk size for translation
MAX_CHUNK_SIZE = 4000
# Minimum delay between API calls (seconds) - increased to avoid rate limiting
MIN_DELAY = 0.5  # Increased from 0.1 to 0.5 seconds
# Maximum delay for exponential backoff
MAX_DELAY = 10.0

# Create a translation cache
translation_cache = {}

def get_cache_key(text, source_lang, target_lang):
    """Generate a unique key for the translation cache"""
    key = f"{text[:100]}_{source_lang}_{target_lang}"
    return hashlib.md5(key.encode()).hexdigest()

def save_to_cache(text, translated_text, source_lang, target_lang):
    """Save a translation to the cache"""
    key = get_cache_key(text, source_lang, target_lang)
    translation_cache[key] = translated_text
    
    # Also save to disk for persistence
    cache_file = os.path.join(os.path.dirname(__file__), 'translation_cache', f"{key}.txt")
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(translated_text)

def load_from_cache(text, source_lang, target_lang):
    """Load a translation from the cache if it exists"""
    key = get_cache_key(text, source_lang, target_lang)
    
    # Check memory cache first
    if key in translation_cache:
        return translation_cache[key]
    
    # Then check disk cache
    cache_file = os.path.join(os.path.dirname(__file__), 'translation_cache', f"{key}.txt")
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            translation = f.read()
            translation_cache[key] = translation
            return translation
    
    return None

def translate_text(text, source_lang, target_lang):
    """Translate text with caching, error handling, and validation"""
    if not text or pd.isna(text) or text.strip() == "":
        return ""
    
    # Check cache first
    cached = load_from_cache(text, source_lang, target_lang)
    if cached:
        # For cached English translations, still validate they're actually English
        if target_lang == 'en' and not is_english(cached, debug=False):
            print(f"Cached translation is not in English. Retranslating...")
            # Continue to translation instead of returning cached result
        else:
            return cached
    
    # If not in cache, translate
    max_retries = 5  # Increased retries for better success rate
    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translation = translator.translate(text)
            
            # For back-translation to English, validate the result
            if target_lang == 'en':
                if not is_english(translation, debug=True):
                    print(f"Translation not in English (attempt {attempt+1}/{max_retries})")
                    if attempt == max_retries - 1:
                        raise ValueError("Translation not in English after all attempts")
                    time.sleep((2 ** attempt) * MIN_DELAY)  # Exponential backoff
                    continue
                
                if not validate_translation(text, translation):
                    if attempt == max_retries - 1:
                        raise ValueError("Translation validation failed")
                    continue
            
            # Save to cache
            if translation:
                save_to_cache(text, translation, source_lang, target_lang)
            
            # Add a small delay to avoid rate limiting
            time.sleep(MIN_DELAY)
            return translation
        except Exception as e:
            if attempt < max_retries - 1:
                # Check if it's a rate limiting error
                if "too many requests" in str(e).lower():
                    # Use a longer delay for rate limiting errors
                    sleep_time = min((2 ** (attempt + 2)) * MIN_DELAY, MAX_DELAY)
                    print(f"Rate limit exceeded. Waiting longer: {sleep_time:.2f}s...")
                else:
                    # Normal exponential backoff for other errors
                    sleep_time = min((2 ** attempt) * MIN_DELAY, MAX_DELAY)
                    print(f"Translation error: {e}. Retrying in {sleep_time:.2f}s...")
                
                time.sleep(sleep_time)
            else:
                # If we've used all retries, cache this failure to avoid repeated attempts
                if "too many requests" in str(e).lower():
                    print(f"Rate limit consistently exceeded. Consider running the script later.")
                    # Return original text instead of failing completely
                    return text
                else:
                    raise Exception(f"Translation failed after {max_retries} attempts: {e}")

def back_translate_chunk(chunk, source_lang='en', intermediate_lang='fr'):
    """Back-translate with improved validation"""
    try:
        # Forward translation
        intermediate = translate_text(chunk, source_lang, intermediate_lang)
        if not intermediate or intermediate.strip() == "":
            return chunk
            
        # Add delay between translations
        time.sleep(MIN_DELAY)
        
        # Back translation
        back = translate_text(intermediate, intermediate_lang, source_lang)
        if not back or back.strip() == "":
            return chunk
            
        # 1. Must be in English
        if not is_english(back, debug=False):
            print(f"Back-translation result not in English, returning original")
            return chunk
            
        # 2. Basic length check
        if len(back.split()) < 3:  # Too short
            print(f"Back-translation too short, returning original")
            return chunk
            
        # 3. Check similarity with original
        similarity = calculate_similarity(chunk, back, debug=False)
        if similarity < 0.05:  # Very low similarity threshold
            print(f"Back-translation too different (BLEU: {similarity}), returning original")
            return chunk
            
        return back
    except Exception as e:
        print(f"Back-translation failed: {str(e)}")
        return chunk

def split_into_chunks(text, max_length=MAX_CHUNK_SIZE):
    """Split text into chunks of maximum length, trying to split at sentence boundaries"""
    if not text or len(text) <= max_length:
        return [text]
    
    # Try to split at sentence boundaries (., !, ?)
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ['.', '!', '?'] and len(current) > max_length/2:
            sentences.append(current)
            current = ""
    
    if current:
        sentences.append(current)
    
    # If sentences are still too long, split them further
    chunks = []
    for sentence in sentences:
        if len(sentence) <= max_length:
            chunks.append(sentence)
        else:
            # Split at word boundaries
            words = sentence.split()
            current_chunk = ""
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_length:
                    current_chunk += " " + word if current_chunk else word
                else:
                    chunks.append(current_chunk)
                    current_chunk = word
            if current_chunk:
                chunks.append(current_chunk)
    
    return chunks

def back_translate_parallel(text, source_lang='en', intermediate_lang='fr'):
    """Handle back-translation of text using parallel processing for chunks"""
    if not text or pd.isna(text) or len(text) == 0:
        return text
    
    # Check if the entire text is in cache
    cached = load_from_cache(text, source_lang, intermediate_lang)
    if cached:
        # Verify cached translation is in English
        if not is_english(cached, debug=False):
            print(f"Cached translation is not in English. Retranslating...")
        else:
            return cached
    
    # Split text into chunks
    chunks = split_into_chunks(text)
    
    # If only one chunk, process directly
    if len(chunks) == 1:
        return back_translate_chunk(chunks[0], source_lang, intermediate_lang)
    
    # Process chunks in parallel
    translated_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as executor:
        future_to_chunk = {
            executor.submit(back_translate_chunk, chunk, source_lang, intermediate_lang): i 
            for i, chunk in enumerate(chunks)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                translated_chunk = future.result()
                translated_chunks.append((chunk_idx, translated_chunk))
            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {e}")
                translated_chunks.append((chunk_idx, chunks[chunk_idx]))
    
    # Reassemble chunks in correct order
    translated_chunks.sort(key=lambda x: x[0])
    result = " ".join(chunk for _, chunk in translated_chunks)
    
    # Final verification that result is in English
    if not is_english(result, debug=False):
        print(f"Final assembled translation is not in English. Returning original.")
        return text
    
    # Cache the complete translation
    save_to_cache(text, result, source_lang, intermediate_lang)
    
    return result

def process_batch(batch_df, intermediate_lang):
    """Process a batch of texts with the same intermediate language"""
    results = []
    for idx, row in batch_df.iterrows():
        text = row[text_column]
        translated = back_translate_parallel(text, 'en', intermediate_lang)
        results.append((idx, translated))
    return results

# Check class distribution for each category
print("Original class distribution:")
class_counts = {}
for category in categories:
    counts = df[category].value_counts()
    class_counts[category] = counts
    print(f"\n{category}:")
    print(counts)

# Identify the categories based on positive class counts (threshold >= 0.5)
category_positive_counts = {}
for cat in categories:
    # Count samples where category value is >= 0.5 (not just exactly 1)
    positive_count = len(df[df[cat] >= 0.5])
    category_positive_counts[cat] = positive_count

# Print all category counts for debugging
print("\nPositive sample counts for each category (threshold >= 0.5):")
for cat, count in category_positive_counts.items():
    print(f"{cat}: {count} positive samples")

# Make sure 'sexual_explicit' is always included
least_represented_categories = ['sexual_explicit']  # Always include sexual_explicit

# Add other categories based on representation (excluding sexual_explicit)
other_categories = sorted(
    [(cat, count) for cat, count in category_positive_counts.items() if cat != 'sexual_explicit'],
    key=lambda x: x[1]
)[:NUM_CATEGORIES_TO_BALANCE-1]

least_represented_categories.extend([cat for cat, _ in other_categories])
print(f"\nCategories to balance: {least_represented_categories}")

# Dictionary to store all new samples for each category
new_samples_by_category = {}
# For each underrepresented category
for cat in least_represented_categories:
    # Get all positive samples of this category (threshold >= 0.5)
    category_samples = df[df[cat] >= 0.5].copy()
    current_count = len(category_samples)
    
    # Print detailed information for debugging
    print(f"\nProcessing category: {cat}")
    print(f"Total positive samples found: {current_count}")
    
    # Calculate how many samples to add
    samples_to_add = current_count * (TARGET_FACTOR - 1)
    print(f"Category {cat}: Current count = {current_count}, Need to add {samples_to_add} samples")
    
    # Verify we have enough source samples
    if current_count == 0:
        print(f"Warning: No positive samples found for {cat}, skipping...")
        continue
    
    # Initialize list to store new samples for this category
    new_samples_by_category[cat] = []
    
    # Calculate how many samples to generate per language
    samples_per_language = int(np.ceil(samples_to_add / len(INTERMEDIATE_LANGUAGES)))
    print(f"Will generate {samples_per_language} samples per language")
    
    # Group similar length texts together for more efficient processing
    category_samples['text_length'] = category_samples[text_column].str.len()
    category_samples = category_samples.sort_values('text_length')
    
    # Keep track of successfully translated samples
    successful_translations = 0
    
    # For each intermediate language
    for lang_idx, lang in enumerate(INTERMEDIATE_LANGUAGES):
        if successful_translations >= samples_to_add:
            print(f"Reached target for {cat}, moving to next category")
            break
            
        # Calculate remaining samples needed
        remaining_samples = samples_to_add - successful_translations
        
        # Calculate samples for this language
        if lang_idx == len(INTERMEDIATE_LANGUAGES) - 1:
            samples_for_this_lang = remaining_samples
        else:
            samples_for_this_lang = min(samples_per_language, remaining_samples)
        
        print(f"\nTranslating {samples_for_this_lang} samples using {lang}...")
        print(f"Progress: {successful_translations}/{samples_to_add} total samples generated")
        
        # Sample with replacement if needed
        need_replacement = samples_for_this_lang > len(category_samples)
        if need_replacement:
            print(f"Using replacement sampling (need {samples_for_this_lang}, have {len(category_samples)})")
        
        lang_samples = category_samples.sample(
            n=int(samples_for_this_lang),
            replace=need_replacement
        ).copy()
        
        # Process in smaller batches
        batch_size = 20
        num_batches = (len(lang_samples) + batch_size - 1) // batch_size
        
        all_results = []
        batch_successful = 0
        
        for i in tqdm(range(num_batches), desc=f"Processing batches via {lang}"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(lang_samples))
            batch = lang_samples.iloc[start_idx:end_idx]
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_idx = {}
                for idx, row in batch.iterrows():
                    text = row[text_column]
                    if pd.isna(text) or text.strip() == "":
                        continue
                    future = executor.submit(back_translate_parallel, text, 'en', lang)
                    future_to_idx[future] = idx
                
                # Collect results
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_idx),
                    total=len(future_to_idx),
                    desc=f"Batch {i+1}/{num_batches}",
                    leave=False
                ):
                    idx = future_to_idx[future]
                    try:
                        translated_text = future.result()
                        # Enhanced validation to ensure only English text
                        if translated_text.strip() != "":
                            # Double-check language detection
                            if is_english(translated_text, debug=True):
                                # Additional quality check
                                original_text = row[text_column]
                                similarity = calculate_similarity(original_text, translated_text, debug=False)
                                if similarity >= 0.05:  # Very low threshold just to catch completely unrelated text
                                    all_results.append((idx, translated_text))
                                    batch_successful += 1
                                else:
                                    print(f"Skipping translation with too low similarity at index {idx} (BLEU: {similarity})")
                            else:
                                print(f"Skipping non-English translation at index {idx}")
                        else:
                            print(f"Skipping empty translation at index {idx}")
                    except Exception as e:
                        print(f"Error processing text at index {idx}: {e}")
                        # Don't append failed translations
            
            print(f"Batch {i+1}: Successfully translated {batch_successful} samples")
        
        # Create new dataframe only with successful translations
        if all_results:
            successful_df = lang_samples.copy()
            successful_df = successful_df.loc[[idx for idx, _ in all_results]]
            for idx, translated_text in all_results:
                successful_df.loc[idx, text_column] = translated_text
            
            # Add only successful translations to our collection
            new_samples_by_category[cat].append(successful_df)
            successful_translations += len(successful_df)
            
            print(f"\nLanguage {lang}: Added {len(successful_df)} successful translations")
            print(f"Total progress: {successful_translations}/{samples_to_add} samples")
        else:
            print(f"\nWarning: No successful translations for {lang}")
        
        if successful_translations >= samples_to_add:
            print(f"\nReached target for {cat}, moving to next category")
            break
        elif lang_idx == len(INTERMEDIATE_LANGUAGES) - 1:
            print(f"\nWarning: Could only generate {successful_translations}/{samples_to_add} samples for {cat}")
            
# Combine all additional samples across all categories
additional_samples = []
for cat, sample_dfs in new_samples_by_category.items():
    if sample_dfs:
        cat_samples = pd.concat(sample_dfs)
        additional_samples.append(cat_samples)
        print(f"Category {cat}: Added {len(cat_samples)} new samples")

# Combine all additional samples
if additional_samples:
    additional_df = pd.concat(additional_samples)
    
    # Add the additional samples to the original dataframe
    balanced_df = pd.concat([df, additional_df])
    
    # Shuffle the dataframe
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    
    # Check the new class distribution
    print("\nNew class distribution:")
    for category in categories:
        print(f"\n{category}:")
        new_counts = balanced_df[category].value_counts()
        print(new_counts)
        
        # Calculate the increase for positive samples
        original_positive = class_counts[category].get(1, 0)
        new_positive = new_counts.get(1, 0)
        increase_factor = new_positive / original_positive if original_positive > 0 else 0
        print(f"Increase factor for {category}: {increase_factor:.2f}x")
    
    # Final check to ensure all comments are in English
    non_english_count = 0
    non_english_indices = []
    
    print("\nPerforming final check to ensure all comments are in English...")
    for idx, row in tqdm(balanced_df.iterrows(), total=len(balanced_df), desc="Verifying English-only dataset"):
        text = row[text_column]
        if not is_english(text, debug=False):
            non_english_count += 1
            non_english_indices.append(idx)
            if non_english_count <= 10:  # Show only first 10 examples
                print(f"Found non-English text at index {idx}: {text[:100]}...")
    
    if non_english_count > 0:
        print(f"\nWarning: Found {non_english_count} non-English comments ({non_english_count/len(balanced_df):.2%} of dataset)")
        print("Removing non-English comments from the dataset...")
        balanced_df = balanced_df.drop(non_english_indices)
        print(f"Dataset size after removing non-English comments: {len(balanced_df)}")
    else:
        print("\nAll comments are in English!")
    
    # Save the balanced dataset
    balanced_df.to_csv('data/merged_data/backtranslated_dataset.csv', index=False)
    print(f"Saved balanced dataset with {len(balanced_df)} samples")
else:
    print("No additional samples were added")
