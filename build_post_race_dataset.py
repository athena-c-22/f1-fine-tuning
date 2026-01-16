"""
F1 Post-Race Analysis Dataset Builder

Enhanced version that adds:
- FastF1 race events (safety cars, flags, etc.)
- IBM Granite post-race analysis summaries
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

import requests
import json
import time
from pathlib import Path
import whisper
import fastf1
from filter_dataset import is_english, is_gibberish, is_purely_conversational

try:
    from ibm_watsonx_ai.foundation_models import Model
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
except ImportError:
    print("Warning: ibm-watsonx-ai not installed")

# Configuration
IBM_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID")
GRANITE_MODEL_ID = "ibm/granite-3b-code-instruct"
WHISPER_MODEL = "base" 
OUTPUT_FILE = "post_race_training_data.jsonl"

# Rate limiting settings
REQUEST_DELAY = 0.5  # seconds between API requests
BATCH_SIZE = 10      # download 10 MP3s, then pause
BATCH_DELAY = 1.0    # seconds between batches

# All 2024 F1 races
RACES_2024 = [
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Australian Grand Prix",
    "Japanese Grand Prix",
    "Chinese Grand Prix",
    "Miami Grand Prix",
    "Emilia Romagna Grand Prix",
    "Monaco Grand Prix",
    "Canadian Grand Prix",
    "Spanish Grand Prix",
    "Austrian Grand Prix",
    "British Grand Prix",
    "Hungarian Grand Prix",
    "Belgian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "United States Grand Prix",
    "Mexico City Grand Prix",
    "São Paulo Grand Prix",
    "Las Vegas Grand Prix",
    "Qatar Grand Prix",
    "Abu Dhabi Grand Prix",
]


def get_session_key(year, gp_name):
    """Get OpenF1 session key for a race"""
    print(f"  🔍 Looking up session key for {gp_name}...")
    
    url = f"https://api.openf1.org/v1/meetings?year={year}"
    print(f"     Rate limit: waiting {REQUEST_DELAY}s...")
    time.sleep(REQUEST_DELAY)
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"  ❌ Failed to fetch meetings")
        return None
    
    meetings = response.json()
    for meeting in meetings:
        if gp_name.lower() in meeting.get('meeting_official_name', '').lower():
            meeting_key = meeting.get('meeting_key')
            
            # Get race session
            sessions_url = f"https://api.openf1.org/v1/sessions?meeting_key={meeting_key}&session_name=Race"
            print(f"     Rate limit: waiting {REQUEST_DELAY}s...")
            time.sleep(REQUEST_DELAY)
            sessions_response = requests.get(sessions_url)
            
            if sessions_response.status_code == 200:
                sessions = sessions_response.json()
                if sessions:
                    session_key = sessions[0].get('session_key')
                    print(f"  ✓ Found session key: {session_key}")
                    return session_key
    
    print(f"  ❌ No session found")
    return None


def download_and_transcribe_radio(session_key, model):
    """Download MP3s and transcribe for a session"""
    print(f"  📥 Getting radio data...")
    
    url = f"https://api.openf1.org/v1/team_radio?session_key={session_key}"
    print(f"     Rate limit: waiting {REQUEST_DELAY}s...")
    time.sleep(REQUEST_DELAY)
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"  ❌ Failed to get radio data")
        return []
    
    radio_data = response.json()
    print(f"  ✓ Found {len(radio_data)} radio messages")
    
    # Download and transcribe
    transcripts = []
    removed_non_english = 0
    removed_gibberish = 0
    removed_conversational = 0
    
    mp3_dir = Path(f"radio_mp3s/{session_key}")
    mp3_dir.mkdir(parents=True, exist_ok=True)
    
    for i, msg in enumerate(radio_data):
        # Rate limiting: pause after each batch
        if i > 0 and i % BATCH_SIZE == 0:
            print(f"  ⏸  Batch pause ({i}/{len(radio_data)}): waiting {BATCH_DELAY}s...")
            time.sleep(BATCH_DELAY)
        
        recording_url = msg.get('recording_url')
        if not recording_url:
            continue
        
        mp3_file = mp3_dir / f"{session_key}_{i}.mp3"
        
        try:
            print(f"     Rate limit: waiting {REQUEST_DELAY}s...")
            time.sleep(REQUEST_DELAY)  # Rate limit each download
            response = requests.get(recording_url)
            mp3_file.write_bytes(response.content)
            
            # Transcribe
            result = model.transcribe(str(mp3_file))
            transcript = result['text'].strip()
            
            # Filter 1: English only
            if not is_english(transcript):
                removed_non_english += 1
                continue
            
            # Filter 2: No gibberish
            if is_gibberish(transcript):
                removed_gibberish += 1
                continue
            
            # Filter 3: No purely conversational messages
            if is_purely_conversational(transcript):
                removed_conversational += 1
                continue
            
            # Passed all filters
            transcripts.append({
                'driver_number': msg.get('driver_number'),
                'transcript': transcript
            })
            
            if (i + 1) % 5 == 0:
                print(f"  ✓ Processed {i + 1}/{len(radio_data)} ({len(transcripts)} kept)")
        
        except Exception as e:
            print(f"  ⚠️  Error on message {i}: {e}")
    
    print(f"  ✅ Got {len(transcripts)}/{len(radio_data)} transcripts after filtering")
    print(f"     Filtered out: {removed_non_english} non-English, {removed_gibberish} gibberish, {removed_conversational} conversational")
    return transcripts


def get_fastf1_events(year, gp_name):
    """Get race events from FastF1"""
    print(f"  📊 Loading FastF1 events...")
    print(f"     Rate limit: waiting 0.5s...")
    time.sleep(0.5)
    
    # Strip "Grand Prix" from name for FastF1
    fastf1_name = gp_name.replace(" Grand Prix", "").strip()
    
    try:
        # Create cache directory if it doesn't exist
        cache_dir = Path('cache')
        cache_dir.mkdir(exist_ok=True)
        
        fastf1.Cache.enable_cache('cache')
        
        print(f"     Looking for '{fastf1_name}' in FastF1...")
        session = fastf1.get_session(year, fastf1_name, 'R')
        session.load()
        
        events = []
        if hasattr(session, 'race_control_messages') and session.race_control_messages is not None:
            for _, msg in session.race_control_messages.iterrows():
                events.append({
                    "lap": msg.get('Lap', ''),
                    "message": msg.get('Message', ''),
                    "category": msg.get('Category', ''),
                })
        
        print(f"  ✓ Got {len(events)} race events")
        return events
    
    except Exception as e:
        print(f"  ⚠️  FastF1 error: {e}")
        print(f"     Tried name: '{fastf1_name}'")
        return []


def generate_with_granite(transcripts, events, gp_name, year):
    """Generate post-race analysis with Granite"""
    print(f"  🤖 Generating analysis with Granite...")
    
    # Format inputs
    radio_text = "\n".join([f"- {t['transcript']}" for t in transcripts[:30]])
    events_text = "\n".join([f"Lap {e['lap']}: {e['message']}" for e in events[:20]])
    
    prompt = f"""Analyze {year} {gp_name}:

**Race Events:**
{events_text}

**Team Radio:**
{radio_text}

Provide analysis in JSON:
{{
  "race_summary": "2-3 sentences",
  "key_decisions": ["decision 1", "decision 2"],
  "critical_moments": ["moment 1", "moment 2"],
  "recommendations": ["recommendation 1", "recommendation 2"]
}}"""
    
    if not IBM_API_KEY or not IBM_PROJECT_ID:
        print(f"  ⚠️  IBM credentials not set, using mock data")
        return '{"race_summary": "MOCK - Set IBM_CLOUD_API_KEY and IBM_PROJECT_ID", "key_decisions": [], "critical_moments": [], "recommendations": []}'
    
    try:
        model = Model(
            model_id=GRANITE_MODEL_ID,
            params={GenParams.MAX_NEW_TOKENS: 1000, GenParams.TEMPERATURE: 0.7},
            credentials={"apikey": IBM_API_KEY, "url": "https://eu-gb.ml.cloud.ibm.com"},  # London region
            project_id=IBM_PROJECT_ID
        )
        
        completion = model.generate_text(prompt=prompt)
        print(f"  ✓ Analysis generated ({len(completion)} chars)")
        return completion
    
    except Exception as e:
        print(f"  ❌ Granite error: {e}")
        return f'{{"error": "{str(e)}"}}'


def process_race(year, gp_name, whisper_model):
    """Process one race - generates per-driver training examples"""
    print(f"\n{'='*60}")
    print(f"{year} {gp_name}")
    print(f"{'='*60}")
    
    # Get session key
    session_key = get_session_key(year, gp_name)
    if not session_key:
        print("⚠️  Skipping - no session key found\n")
        return []
    
    # Download and transcribe
    transcripts = download_and_transcribe_radio(session_key, whisper_model)
    if not transcripts:
        print("⚠️  Skipping - no transcripts\n")
        return []
    
    # Get FastF1 events
    events = get_fastf1_events(year, gp_name)
    
    # Group transcripts by driver
    driver_transcripts = {}
    for t in transcripts:
        driver_num = t.get('driver_number')
        if driver_num:
            if driver_num not in driver_transcripts:
                driver_transcripts[driver_num] = []
            driver_transcripts[driver_num].append(t)
    
    print(f"  📊 Found {len(driver_transcripts)} drivers with transcripts")
    
    # Generate one training example per driver
    examples = []
    for driver_num, driver_msgs in driver_transcripts.items():
        # Skip drivers with too few messages
        if len(driver_msgs) < 3:
            print(f"  ⚠️  Driver {driver_num}: Only {len(driver_msgs)} messages, skipping")
            continue
        
        print(f"  🤖 Processing Driver {driver_num} ({len(driver_msgs)} messages)...")
        
        # Generate driver-specific summary
        radio_sample = "\n".join([f"- {t['transcript']}" for t in driver_msgs[:15]])
        events_sample = "\n".join([f"Lap {e['lap']}: {e['message']}" for e in events[:15]])
        
        prompt = f"""Analyze Driver {driver_num} performance in {year} {gp_name}:

**Race Events:**
{events_sample}

**Driver {driver_num} Radio:**
{radio_sample}

Provide strategic analysis:"""
        
        completion = generate_with_granite(driver_msgs, events, gp_name, year)
        
        examples.append({
            "prompt": prompt,
            "completion": completion,
            "metadata": {
                "year": year,
                "gp": gp_name,
                "driver_number": driver_num,
                "session_key": session_key,
                "num_transcripts": len(driver_msgs),
                "num_events": len(events)
            }
        })
        
        print(f"  ✓ Driver {driver_num} complete")
    
    return examples


def main():
    print("="*60)
    print("F1 Post-Race Dataset Builder")
    print("="*60)
    
    # Load Whisper
    print(f"\n📥 Loading Whisper {WHISPER_MODEL} model...")
    model = whisper.load_model(WHISPER_MODEL)
    print("✓ Whisper loaded")
    
    # Process races
    training_examples = []
    for gp in RACES_2024:
        examples = process_race(2024, gp, model)
        
        if examples:
            training_examples.extend(examples)
            
            # Save incrementally
            for example in examples:
                with open(OUTPUT_FILE, 'a') as f:
                    f.write(json.dumps(example) + '\n')
            
            print(f"✅ Saved {len(examples)} driver examples to {OUTPUT_FILE}\n")
    
    print(f"\n{'='*60}")
    print(f"✅ Complete! Generated {len(training_examples)} examples")
    print(f"   (~{len(training_examples) / len(RACES_2024):.1f} drivers per race average)")
    print(f"   Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
