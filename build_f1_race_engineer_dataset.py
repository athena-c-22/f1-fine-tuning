"""
F1 Race Engineer Dataset Builder

This script creates a fine-tuning dataset by:
1. Fetching F1 telemetry data from OpenF1 API
2. Downloading and transcribing team radio messages
3. Pairing telemetry context with radio transcripts
4. Saving as JSONL format for model fine-tuning
"""

import requests
import pandas as pd
import json
import datetime
import time
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional
import whisper


# Configuration
SESSION_KEY = 9161  # Change this to the session you want
DRIVER_NUMBER = 44  # Change this to the driver you want
YEAR = 2024
TELEMETRY_WINDOW_SECONDS = 30  # Seconds before radio message to include telemetry
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
OUTPUT_FILE = "f1_dataset.jsonl"
CLEANUP_AUDIO_FILES = False  # Set to True to delete audio files after transcription


def fetch_api_data(url: str, description: str) -> Optional[List[Dict]]:
    """Fetch data from OpenF1 API with error handling."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data:
            print(f"Warning: No {description} found.")
            return None
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {description}: {e}")
        return None


def get_telemetry_data(session_key: int, driver_number: int) -> Optional[pd.DataFrame]:
    """Fetch and process telemetry data."""
    url = f"https://api.openf1.org/v1/car_data?session_key={session_key}&driver_number={driver_number}"
    data = fetch_api_data(url, "telemetry data")
    
    if data is None:
        return None
    
    df = pd.DataFrame(data)
    if df.empty:
        print("Warning: Telemetry DataFrame is empty.")
        return None
    
    # Convert date column to datetime
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], format='ISO8601')
        except Exception as e:
            print(f"Error converting date column: {e}")
            return None
    else:
        print("Warning: Telemetry data missing 'date' column.")
        return None
    
    return df


def get_radio_data(session_key: int, driver_number: int) -> Optional[List[Dict]]:
    """Fetch team radio data."""
    url = f"https://api.openf1.org/v1/team_radio?session_key={session_key}&driver_number={driver_number}"
    return fetch_api_data(url, "radio data")


def sanitize_filename(date_str: str) -> str:
    """Create a safe filename from ISO date string."""
    try:
        # Handle Z timezone indicator
        date_str = date_str.replace('Z', '+00:00')
        date_obj = datetime.datetime.fromisoformat(date_str)
        return date_obj.strftime('%Y%m%d_%H%M%S')
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not parse date '{date_str}': {e}")
        # Fallback: use hash of date string
        import hashlib
        return hashlib.md5(date_str.encode()).hexdigest()[:12]


def download_audio(url: str, filename: str) -> bool:
    """Download audio file with error handling."""
    try:
        urllib.request.urlretrieve(url, filename)
        time.sleep(0.1)  # Small delay to ensure file is written
        return True
    except urllib.error.URLError as e:
        print(f"Error downloading audio from {url}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading audio: {e}")
        return False


def transcribe_audio(audio_path: str, model) -> Optional[str]:
    """Transcribe audio file using Whisper."""
    try:
        # Verify file exists and has content
        path = Path(audio_path)
        if not path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            return None
        
        if path.stat().st_size == 0:
            print(f"Error: Audio file is empty: {audio_path}")
            return None
        
        # Use relative path for Whisper (better compatibility on Windows)
        result = model.transcribe(str(path))
        return result.get('text', '').strip()
    except FileNotFoundError as e:
        print(f"File not found error during transcription: {e}")
        return None
    except Exception as e:
        print(f"Error transcribing audio: {type(e).__name__}: {e}")
        return None


def process_radio_messages(radio_data: List[Dict], model) -> List[Dict]:
    """Download and transcribe all radio messages."""
    radio_list = []
    
    if not radio_data:
        return radio_list
    
    print(f"Processing {len(radio_data)} radio messages...")
    
    for idx, item in enumerate(radio_data, 1):
        try:
            # Validate required fields
            if 'date' not in item or 'recording_url' not in item:
                print(f"Warning: Radio item {idx} missing required fields, skipping.")
                continue
            
            # Create safe filename
            safe_date = sanitize_filename(item['date'])
            audio_filename = f"radio_{safe_date}.mp3"
            
            print(f"[{idx}/{len(radio_data)}] Processing: {audio_filename}")
            
            # Download audio
            if not download_audio(item['recording_url'], audio_filename):
                continue
            
            # Transcribe
            transcript = transcribe_audio(audio_filename, model)
            if transcript is None or not transcript:
                print(f"Warning: No transcript generated for {audio_filename}")
                if CLEANUP_AUDIO_FILES:
                    Path(audio_filename).unlink(missing_ok=True)
                continue
            
            # Store result
            radio_list.append({
                'timestamp': item['date'],
                'transcript': transcript
            })
            print(f"  ✓ Transcribed: {len(transcript)} characters")
            
            # Cleanup if requested
            if CLEANUP_AUDIO_FILES:
                Path(audio_filename).unlink(missing_ok=True)
                
        except Exception as e:
            print(f"Error processing radio item {idx}: {type(e).__name__}: {e}")
            continue
    
    print(f"Successfully processed {len(radio_list)}/{len(radio_data)} radio messages.")
    return radio_list


def create_training_pairs(radio_list: List[Dict], telemetry_df: pd.DataFrame, 
                          window_seconds: int) -> List[Dict]:
    """Create training pairs by matching telemetry with radio transcripts."""
    dataset = []
    
    if not radio_list:
        print("Warning: No radio transcripts to pair with telemetry.")
        return dataset
    
    if telemetry_df is None or telemetry_df.empty:
        print("Warning: No telemetry data available for pairing.")
        return dataset
    
    if 'date' not in telemetry_df.columns:
        print("Warning: Telemetry data missing 'date' column.")
        return dataset
    
    print(f"Creating training pairs from {len(radio_list)} radio messages...")
    
    for radio in radio_list:
        try:
            # Parse radio timestamp
            radio_time = datetime.datetime.fromisoformat(
                radio['timestamp'].replace('Z', '+00:00')
            )
            window_start = radio_time - datetime.timedelta(seconds=window_seconds)
            
            # Filter telemetry in time window
            mask = (telemetry_df['date'] >= window_start) & (telemetry_df['date'] < radio_time)
            context_data = telemetry_df[mask]
            
            if context_data.empty:
                continue  # Skip if no telemetry in window
            
            # Aggregate telemetry metrics
            metrics = {}
            if 'speed' in context_data.columns:
                metrics['avg_speed'] = context_data['speed'].mean()
            if 'rpm' in context_data.columns:
                metrics['avg_rpm'] = context_data['rpm'].mean()
            if 'throttle' in context_data.columns:
                metrics['avg_throttle'] = context_data['throttle'].mean()
            if 'brake' in context_data.columns:
                metrics['avg_brake'] = context_data['brake'].mean()
            
            # Build prompt
            metric_strs = [f"{k.replace('avg_', '')} {v:.1f}" for k, v in metrics.items()]
            prompt = f"Telemetry: {', '.join(metric_strs) if metric_strs else 'No metrics available'}. Advice:"
            completion = radio['transcript']
            
            dataset.append({
                "prompt": prompt,
                "completion": completion
            })
            
        except Exception as e:
            print(f"Error creating training pair: {type(e).__name__}: {e}")
            continue
    
    print(f"Created {len(dataset)} training pairs.")
    return dataset


def save_dataset(dataset: List[Dict], filename: str):
    """Save dataset as JSONL file."""
    if not dataset:
        print("Warning: No data to save.")
        return
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(dataset)} training examples to {filename}")
    except Exception as e:
        print(f"Error saving dataset: {e}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("F1 Race Engineer Dataset Builder")
    print("=" * 60)
    print(f"Session: {SESSION_KEY}, Driver: {DRIVER_NUMBER}, Year: {YEAR}")
    print()
    
    # Load Whisper model
    print("Loading Whisper model...")
    try:
        model = whisper.load_model(WHISPER_MODEL)
        print(f"✓ Loaded Whisper model: {WHISPER_MODEL}")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return
    
    print()
    
    # Fetch telemetry data
    print("Fetching telemetry data...")
    telemetry_df = get_telemetry_data(SESSION_KEY, DRIVER_NUMBER)
    if telemetry_df is not None:
        print(f"✓ Loaded {len(telemetry_df)} telemetry records")
    else:
        print("✗ Failed to load telemetry data")
        return
    
    print()
    
    # Fetch radio data
    print("Fetching radio data...")
    radio_data = get_radio_data(SESSION_KEY, DRIVER_NUMBER)
    if radio_data is None:
        print("✗ Failed to load radio data")
        return
    print(f"✓ Found {len(radio_data)} radio messages")
    
    print()
    
    # Process radio messages (download and transcribe)
    radio_list = process_radio_messages(radio_data, model)
    
    if not radio_list:
        print("No radio transcripts available. Exiting.")
        return
    
    print()
    
    # Create training pairs
    dataset = create_training_pairs(radio_list, telemetry_df, TELEMETRY_WINDOW_SECONDS)
    
    if not dataset:
        print("No training pairs created. Exiting.")
        return
    
    print()
    
    # Save dataset
    save_dataset(dataset, OUTPUT_FILE)
    
    print()
    print("=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
