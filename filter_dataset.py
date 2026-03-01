import json
import re
import os
import langid

def is_gibberish(text):
    """
    Detect gibberish transcriptions that should be filtered out.
    Returns True if the text appears to be gibberish.
    """
    if not text or len(text.strip()) < 10:
        return True
    
    # Check ASCII ratio - gibberish often has many non-ASCII characters
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_chars / len(text) if len(text) > 0 else 0
    if ascii_ratio < 0.6:
        return True
    
    # Check letter ratio - gibberish has low letter-to-total ratio
    letters = sum(1 for c in text if c.isalpha())
    letter_ratio = letters / len(text) if len(text) > 0 else 0
    if letter_ratio < 0.5:
        return True
    
    # Check for excessive special characters or numbers
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars > len(text) * 0.3:
        return True
    
    return False

def is_english(text):
    """
    Detect if the text is in English using langid.
    Returns True if detected as English or has common English words.
    """
    if not text or len(text.strip()) < 10:
        return True  # Be conservative - keep short messages
    
    # Check for common English words
    text_lower = text.lower()
    common_english_words = [
        'the', 'and', 'for', 'you', 'that', 'this', 'with', 'from', 'have', 'are', 'was', 'were',
        'let', 'get', 'go', 'okay', 'copy', 'yes', 'yeah', 'we', 'suggest', 'it', 'its', "it's",
        'can', 'will', 'would', 'should', 'could', 'do', 'does', 'did', 'not', 'is', 'am', 'be',
        'take', 'look', 'good', 'nice', 'job', 'fine', 'super', 'well', 'push', 'full', 'work',
        'more', 'energy', 'strat', 'mate', 'guys', 'thank', 'thanks', 'all', 'my', 'i', "i'm",
        "let's", "we'll", "i'll", "you'll", "that's", "what's", "here's", "there's"
    ]
    has_common_words = sum(1 for word in common_english_words if f' {word} ' in f' {text_lower} ' or text_lower.startswith(word + ' ') or text_lower.endswith(' ' + word))
    
    # If has common English words, definitely English
    if has_common_words >= 1:
        return True
    
    # Use langid for detection
    lang, confidence = langid.classify(text)
    
    # If detected as English, keep it
    if lang == 'en':
        return True
    
    # If not detected as English but confidence is low (uncertain), be conservative and keep it
    # Low confidence means the detector isn't sure, so we don't want to remove potentially English text
    if confidence > -50:  # langid uses negative log probabilities, higher (less negative) = less confident
        return True
    
    # High confidence non-English detection - remove it
    return False

def is_purely_conversational(text):
    """
    Detect purely conversational messages without technical content.
    Returns True if the message is purely conversational.
    """
    text_lower = text.lower().strip()
    
    # If text contains any numbers (positions, laps, settings, times), keep it
    if any(c.isdigit() for c in text):
        return False
    
    # Technical keywords that indicate the message has value
    technical_keywords = [
        'engine', 'strat', 'mode', 'position', 'p1', 'p2', 'p3', 'p4', 'p5',
        'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15',
        'p16', 'p17', 'p18', 'p19', 'p20',
        'tire', 'tyre', 'deg', 'gap', 'delta', 'lap', 'laps', 'box', 'pit', 'stop',
        'fuel', 'temp', 'temperature', 'brake', 'brakes', 'diff', 'energy',
        'drs', 'overtake', 'defend', 'attack', 'pace', 'sector', 'speed',
        'debris', 'yellow', 'flag', 'flags', 'safety', 'vsc', 'damage', 'front', 'rear',
        'balance', 'understeer', 'oversteer', 'downforce', 'ers', 'battery',
        'charge', 'deploy', 'harvest', 'rpm', 'throttle', 'steering',
        'suspension', 'ride', 'height', 'wing', 'wings', 'setting', 'settings', 'switch', 'turn',
        'flat', 'checkered', 'check', 'rain', 'wet', 'dry', 'slicks', 'inters', 'intermediate',
        'softs', 'mediums', 'hards', 'compound', 'degradation', 'graining',
        'lock', 'lockup', 'spin', 'slide', 'grip', 'traction', 'vibration',
        'cool', 'cooling', 'overheat', 'pressure', 'pressures', 'window',
        'lift', 'coast', 'saving', 'manage', 'target', 'margin',
        'fastest', 'quickest', 'slower', 'quicker', 'losing', 'gaining',
        'behind', 'ahead', 'catching', 'dropping', 'closing',
        'purple', 'green', 'personal', 'best', 'time',
        'vset', 'bias', 'offset', 'bbal', 'brake balance',
        # Additional technical terms from analysis
        'struggling', 'struggle', 'bouncing', 'bounce', 'pulling',
        'wind', 'gusts', 'track', 'conditions', 'formation', 'grid',
        'car', 'exit', 'entry', 'straight', 'line', 'corner',
        'clutch', 'drop', 'gear', 'gears', 'second', 'third', 'fourth',
        'rev', 'revs', 'drink', 'visor', 'radio', 'data',
        'video', 'telemetry', 'issues', 'issue', 'problem',
        'contact', 'incident', 'penalty', 'stewards',
        'backing', 'formation', 'procedure', 'start',
        'push', 'pushing', 'lifting', 'saving', 'managing',
        'left', 'right', 'side', 'sides', 'bottom',
        'hot', 'cold', 'warm', 'warmup', 'warm up',
        # Edge cases from validation
        'sticks', 'focus', 'clump', 'clumping', 'click', 'smooths',
        'braking', 'brake point', 'lockup', 'locking', 'break'
    ]
    
    # If text contains technical keywords, keep it even if conversational
    for keyword in technical_keywords:
        if keyword in text_lower:
            return False
    
    # Conversational words/phrases to check for (anywhere in message)
    conversational_words = [
        'okay', 'ok', 'copy', 'copied', 'roger', 'affirm', 'affirmative',
        'yes', 'yep', 'yeah', 'no', 'nope',
        'thanks', 'thank you', 'cheers', 'appreciate it',
        'good job', 'well done', 'nice', 'great', 'excellent', 'perfect', 'brilliant',
        'sorry', 'my bad', 'apologies',
        "let's go", 'lets go', 'come on', 'push', 'keep going', 'stay focused',
        'copy that', 'understood', 'got it', 'all good', 'all clear',
        'see you', 'talk later', 'catch you',
        'nice one', 'good work', 'keep pushing', 'stay calm', 'focus',
        'push now', 'good stuff', 'keep it up', 'great job',
        'mate', 'guys', 'lads', 'buddy'
    ]
    
    # Check if message contains conversational words
    has_conversational = any(word in text_lower for word in conversational_words)
    
    # If it has conversational content but no technical content, remove it
    if has_conversational:
        return True
    
    return False

def filter_dataset(input_file, output_file, removed_file):
    """
    Filter a JSONL dataset file to remove gibberish, non-English, and purely conversational messages.
    Also saves removed entries to a separate file for review.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return
    
    kept_count = 0
    gibberish_count = 0
    conversational_count = 0
    non_english_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(removed_file, 'w', encoding='utf-8') as removedfile:
        
        for line in infile:
            total_count += 1
            try:
                entry = json.loads(line.strip())
                completion = entry.get('completion', '').strip()
                
                removed = False
                removal_reason = None
                
                if is_gibberish(completion):
                    gibberish_count += 1
                    removed = True
                    removal_reason = "gibberish"
                elif not is_english(completion):
                    non_english_count += 1
                    removed = True
                    removal_reason = "non-english"
                elif is_purely_conversational(completion):
                    conversational_count += 1
                    removed = True
                    removal_reason = "conversational"
                
                if removed:
                    # Add removal reason to the entry
                    entry['removal_reason'] = removal_reason
                    removedfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                else:
                    # Keep this entry
                    outfile.write(line)
                    kept_count += 1
                
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line[:50]}...")
                continue
    
    print(f"\nFiltering complete for {input_file}:")
    print(f"  Total entries: {total_count}")
    print(f"  Kept: {kept_count}")
    print(f"  Removed gibberish: {gibberish_count}")
    print(f"  Removed non-English: {non_english_count}")
    print(f"  Removed conversational: {conversational_count}")
    print(f"  Output saved to: {output_file}")
    print(f"  Removed entries saved to: {removed_file}")
    
    return kept_count, gibberish_count, non_english_count, conversational_count

def main():
    """
    Process 2023-2025 dataset files.
    """
    print("F1 Dataset Filter - Removing Gibberish, Non-English, and Purely Conversational Messages")
    print("=" * 80)

    # Filter 2025 dataset
    if os.path.exists('f1_dataset_2025.jsonl'):
        print("\nProcessing 2025 dataset...")
        filter_dataset('f1_dataset_2025.jsonl', 
                      'f1_dataset_2025_filtered.jsonl',
                      'f1_dataset_2025_removed.jsonl')
    else:
        print("\nWarning: f1_dataset_2025.jsonl not found, skipping...")
    
    # # Filter 2024 dataset
    # if os.path.exists('f1_dataset_2024.jsonl'):
    #     print("\nProcessing 2024 dataset...")
    #     filter_dataset('f1_dataset_2024.jsonl', 
    #                   'f1_dataset_2024_filtered.jsonl',
    #                   'f1_dataset_2024_removed.jsonl')
    # else:
    #     print("\nWarning: f1_dataset_2024.jsonl not found, skipping...")
    
    # # Filter 2023 dataset
    # if os.path.exists('f1_dataset_2023.jsonl'):
    #     print("\nProcessing 2023 dataset...")
    #     filter_dataset('f1_dataset_2023.jsonl', 
    #                   'f1_dataset_2023_filtered.jsonl',
    #                   'f1_dataset_2023_removed.jsonl')
    # else:
    #     print("\nWarning: f1_dataset_2023.jsonl not found, skipping...")
    
    # Combine filtered datasets if all exist
    if os.path.exists('f1_dataset_2025_filtered.jsonl') and os.path.exists('f1_dataset_2024_filtered.jsonl') and os.path.exists('f1_dataset_2023_filtered.jsonl'):
        print("\n" + "=" * 80)
        print("Combining filtered datasets...")
        
        combined_count = 0
        with open('f1_dataset_combined_filtered.jsonl', 'w', encoding='utf-8') as outfile:
            # Add 2025 data
            with open('f1_dataset_2025_filtered.jsonl', 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
                    combined_count += 1

            # Add 2024 data
            with open('f1_dataset_2024_filtered.jsonl', 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
                    combined_count += 1
            
            # Add 2023 data
            with open('f1_dataset_2023_filtered.jsonl', 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
                    combined_count += 1
        
        print(f"Combined filtered dataset created: f1_dataset_combined_filtered.jsonl")
        print(f"Total entries in combined dataset: {combined_count}")
    
    print("\n" + "=" * 80)
    print("Filtering complete!")

if __name__ == "__main__":
    main()
