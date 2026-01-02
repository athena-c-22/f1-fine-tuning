import json
import re
import os

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

def is_purely_conversational(text):
    """
    Detect purely conversational messages without technical content.
    Returns True if the message is purely conversational.
    """
    text_lower = text.lower().strip()
    
    # Technical keywords that indicate the message has value
    technical_keywords = [
        'engine', 'strat', 'mode', 'position', 'p1', 'p2', 'p3', 'p4', 'p5',
        'p6', 'p7', 'p8', 'p9', 'p10', 'tire', 'tyre', 'deg', 'gap', 'delta',
        'lap', 'box', 'pit', 'stop', 'fuel', 'temp', 'brake', 'diff', 'energy',
        'drs', 'overtake', 'defend', 'attack', 'pace', 'sector', 'speed',
        'debris', 'yellow', 'safety', 'vsc', 'damage', 'front', 'rear',
        'balance', 'understeer', 'oversteer', 'downforce', 'ers', 'battery',
        'charge', 'deploy', 'harvest', 'rpm', 'throttle', 'steering',
        'suspension', 'ride', 'height', 'wing', 'setting', 'switch', 'turn'
    ]
    
    # If text contains technical keywords, keep it even if conversational
    for keyword in technical_keywords:
        if keyword in text_lower:
            return False
    
    # Purely conversational patterns (no technical content)
    conversational_patterns = [
        r'^(ok|okay|copy|copied|roger|affirm|affirmative|yes|yep|yeah|no|nope)\.?\s*$',
        r'^(thanks?|thank you|cheers|appreciate it)\.?\s*$',
        r'^(good job|well done|nice|great|excellent|perfect|brilliant)\.?\s*$',
        r'^(sorry|my bad|apologies)\.?\s*$',
        r'^(let\'s go|come on|push|keep going|stay focused)\.?\s*$',
        r'^(copy that|understood|got it|all good|all clear)\.?\s*$',
        r'^(see you|talk later|catch you)\.?\s*$',
    ]
    
    for pattern in conversational_patterns:
        if re.match(pattern, text_lower):
            return True
    
    # Short messages with only encouragement words (no numbers/positions)
    if len(text_lower.split()) <= 5:
        encouragement_only = [
            'well done', 'good job', 'nice one', 'good work', 'keep pushing',
            'stay calm', 'focus', 'push now', 'come on', 'lets go', 
            'good stuff', 'keep it up', 'great job'
        ]
        for phrase in encouragement_only:
            if phrase in text_lower and not any(c.isdigit() for c in text):
                return True
    
    return False

def filter_dataset(input_file, output_file, removed_file):
    """
    Filter a JSONL dataset file to remove gibberish and purely conversational messages.
    Also saves removed entries to a separate file for review.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return
    
    kept_count = 0
    gibberish_count = 0
    conversational_count = 0
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
    print(f"  Removed conversational: {conversational_count}")
    print(f"  Output saved to: {output_file}")
    print(f"  Removed entries saved to: {removed_file}")
    
    return kept_count, gibberish_count, conversational_count

def main():
    """
    Process both 2023 and 2024 dataset files.
    """
    print("F1 Dataset Filter - Removing Gibberish and Purely Conversational Messages")
    print("=" * 70)
    
    # Filter 2024 dataset
    if os.path.exists('f1_dataset_2024.jsonl'):
        print("\nProcessing 2024 dataset...")
        filter_dataset('f1_dataset_2024.jsonl', 
                      'f1_dataset_2024_filtered.jsonl',
                      'f1_dataset_2024_removed.jsonl')
    else:
        print("\nWarning: f1_dataset_2024.jsonl not found, skipping...")
    
    # Filter 2023 dataset
    if os.path.exists('f1_dataset_2023.jsonl'):
        print("\nProcessing 2023 dataset...")
        filter_dataset('f1_dataset_2023.jsonl', 
                      'f1_dataset_2023_filtered.jsonl',
                      'f1_dataset_2023_removed.jsonl')
    else:
        print("\nWarning: f1_dataset_2023.jsonl not found, skipping...")
    
    # Combine filtered datasets if both exist
    if os.path.exists('f1_dataset_2024_filtered.jsonl') and os.path.exists('f1_dataset_2023_filtered.jsonl'):
        print("\n" + "=" * 70)
        print("Combining filtered datasets...")
        
        combined_count = 0
        with open('f1_dataset_combined_filtered.jsonl', 'w', encoding='utf-8') as outfile:
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
    
    print("\n" + "=" * 70)
    print("Filtering complete!")

if __name__ == "__main__":
    main()
