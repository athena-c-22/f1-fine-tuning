"""
F1 Race Engineer Dataset Builder

This script creates a fine-tuning dataset by:
1. Fetching F1 telemetry data from OpenF1 API
2. Downloading and transcribing team radio messages
3. Pairing telemetry context with radio transcripts
4. Saving as JSONL format for model fine-tuning
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['MKL_THREADING_LAYER'] = 'sequential'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import requests
import pandas as pd
import json
import datetime
import time
import urllib.request
import urllib.error
import subprocess
import shlex
from pathlib import Path
from typing import List, Dict, Optional
import whisper
import whisper.audio as whisper_audio


# Configuration
YEARS = [2023]  # Years to process
SESSION_TYPE = "Race"  # Options: "Race", "Practice", "Qualifying", etc. or None for all
TELEMETRY_WINDOW_SECONDS = 30  # Seconds before radio message to include telemetry
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
OUTPUT_FILE = "f1_dataset.jsonl"
CLEANUP_AUDIO_FILES = False  # Set to True to delete audio files after transcription

# Filter options (set to None to process all)
SPECIFIC_SESSIONS = None  # e.g., [9161, 9162] or None for all sessions
SPECIFIC_DRIVERS = None  # e.g., [44, 11] or None for all drivers with radio data
MAX_SESSIONS = None  # Limit number of sessions to process (None = all)
MAX_DRIVERS_PER_SESSION = None  # Limit drivers per session (None = all)


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


def get_sessions(year: int, session_type: Optional[str] = None) -> Optional[List[Dict]]:
    """Get all sessions for a year, optionally filtered by type."""
    if session_type:
        url = f"https://api.openf1.org/v1/sessions?year={year}&session_type={session_type}"
    else:
        url = f"https://api.openf1.org/v1/sessions?year={year}"
    
    return fetch_api_data(url, "sessions")


def get_drivers_with_radio(session_key: int) -> List[int]:
    """Get list of driver numbers that have radio data for a session."""
    url = f"https://api.openf1.org/v1/team_radio?session_key={session_key}"
    radio_data = fetch_api_data(url, f"radio data for session {session_key}")
    
    if not radio_data:
        return []
    
    # Get unique driver numbers from radio data
    driver_numbers = set()
    for item in radio_data:
        if 'driver_number' in item:
            driver_numbers.add(item['driver_number'])
    
    return sorted(list(driver_numbers))


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
        print(f"  Downloading from: {url}")
        urllib.request.urlretrieve(url, filename)
        time.sleep(0.2)  # Small delay to ensure file is written
        
        # Verify download succeeded
        path = Path(filename)
        if not path.exists():
            print(f"  ✗ Error: File not found after download: {filename}")
            return False
        
        file_size = path.stat().st_size
        if file_size == 0:
            print(f"  ✗ Error: Downloaded file is empty: {filename}")
            path.unlink(missing_ok=True)
            return False
        
        print(f"  ✓ Downloaded successfully: {file_size} bytes")
        return True
    except urllib.error.URLError as e:
        print(f"  ✗ Error downloading audio from {url}: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error downloading audio: {e}")
        return False


def transcribe_audio(audio_path: str, model) -> Optional[str]:
    """Transcribe audio file using Whisper."""
    try:
        # Resolve to absolute path
        path = Path(audio_path).resolve()
        
        print(f"  Verifying file: {path}")
        if not path.exists():
            print(f"  ✗ Error: Audio file not found: {path}")
            print(f"    Current directory: {os.getcwd()}")
            return None
        
        file_size = path.stat().st_size
        if file_size == 0:
            print(f"  ✗ Error: Audio file is empty: {path}")
            return None
        
        print(f"  ✓ File verified: {file_size} bytes")
        
        # Get absolute path as string
        abs_path = str(path.absolute())
        print(f"  Transcribing with path: {abs_path}")
        
        # Verify file is readable
        if not os.access(abs_path, os.R_OK):
            print(f"  ✗ Error: File is not readable: {abs_path}")
            return None
        
        # Try to open the file to ensure it's accessible
        try:
            with open(abs_path, 'rb') as test_file:
                test_file.read(1)
        except Exception as e:
            print(f"  ✗ Error: Cannot open file for reading: {e}")
            return None
        
        print(f"  Starting Whisper transcription...")
        
        # Debug: Print the exact path being used
        print(f"    Debug - Absolute path: {repr(abs_path)}")
        print(f"    Debug - Path exists: {Path(abs_path).exists()}")
        print(f"    Debug - Current working directory: {os.getcwd()}")
        
        # Strategy 1: Try loading audio directly (bypasses ffmpeg path issues)
        try:
            print(f"    Trying direct audio loading...")
            # Load audio using Whisper's audio loader (handles ffmpeg internally)
            audio = whisper_audio.load_audio(abs_path)
            print(f"    ✓ Audio loaded directly: {len(audio)} samples")
            result = model.transcribe(audio)
            transcript = result.get('text', '').strip()
            if transcript:
                print(f"  ✓ Transcription successful: {len(transcript)} characters")
                return transcript
        except Exception as e:
            print(f"    ✗ Direct audio loading failed: {type(e).__name__}: {e}")
            # Continue to try file path method
        
        # Strategy 2: Try multiple path formats for file path method
        # Whisper/ffmpeg on Windows can be picky about path formats
        transcription_paths = [
            (abs_path, "absolute path (backslashes)"),
            (abs_path.replace('\\', '/'), "absolute path (forward slashes)"),
            (os.path.relpath(abs_path, os.getcwd()), "relative path"),
        ]
        
        last_error = None
        for transcribe_path, path_type in transcription_paths:
            try:
                print(f"    Trying {path_type}: {repr(transcribe_path)}")
                
                # Verify the path still exists before trying
                test_path = transcribe_path
                if not Path(test_path).exists():
                    print(f"      ⚠ Path doesn't exist, skipping...")
                    continue
                
                result = model.transcribe(transcribe_path)
                transcript = result.get('text', '').strip()
                
                if transcript:
                    print(f"  ✓ Transcription successful: {len(transcript)} characters")
                    return transcript
                else:
                    print(f"    ⚠ Empty transcript with {path_type}, trying next...")
                    continue
            except FileNotFoundError as e:
                last_error = e
                print(f"    ✗ FileNotFoundError with {path_type}: {e}")
                print(f"      Error details: {repr(str(e))}")
                continue
            except Exception as e:
                error_str = str(e).lower()
                error_type = type(e).__name__
                
                # Check for file/path related errors
                if ("cannot find the file" in error_str or 
                    "winerror 2" in error_str or 
                    "filenotfound" in error_str or
                    "CreateProcess" in str(e) or
                    "subprocess" in error_str):
                    last_error = e
                    print(f"    ✗ File/process error with {path_type}: {error_type}: {e}")
                    continue
                
                # For other errors, log but don't treat as fatal - might be transcription issue
                print(f"    ⚠ Error with {path_type}: {error_type}: {e}")
                # If it's a subprocess error, it's likely ffmpeg related
                if "subprocess" in error_str or "CreateProcess" in str(e):
                    last_error = e
                continue
        
        # If all attempts failed
        print(f"  ✗ All transcription attempts failed")
        if last_error:
            # Don't re-raise, just return None with error message
            error_msg = str(last_error)
            if "CreateProcess" in error_msg or "subprocess" in error_msg.lower():
                print(f"  ⚠ This might be an ffmpeg issue. Whisper requires ffmpeg to be installed.")
                print(f"     Make sure ffmpeg is installed and in your PATH.")
            print(f"  Error details: {error_msg}")
        return None
            
    except FileNotFoundError as e:
        print(f"  ✗ FileNotFoundError during transcription: {e}")
        print(f"    Path attempted: {abs_path if 'abs_path' in locals() else audio_path}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"  ✗ Error transcribing audio: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_radio_messages(radio_data: List[Dict], model) -> List[Dict]:
    print("In process_radio_messages")
    print("radio_data: ", radio_data)
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


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available (required by Whisper)."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ ffmpeg is available")
            return True
        else:
            print("✗ ffmpeg check failed")
            return False
    except FileNotFoundError:
        print("✗ ffmpeg not found in PATH")
        print("  Whisper requires ffmpeg to process audio files.")
        print("  Please install ffmpeg: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"⚠ Error checking ffmpeg: {e}")
        return False


def process_session_driver(session_key: int, driver_number: int, model, 
                           window_seconds: int) -> List[Dict]:
    """Process a single session-driver combination and return training pairs."""
    print(f"\n{'='*60}")
    print(f"Processing Session {session_key}, Driver {driver_number}")
    print(f"{'='*60}")
    
    # Fetch telemetry data
    print("  Fetching telemetry data...")
    telemetry_df = get_telemetry_data(session_key, driver_number)
    if telemetry_df is None:
        print("  ✗ No telemetry data available")
        return []
    print(f"  ✓ Loaded {len(telemetry_df)} telemetry records")
    
    # Fetch radio data
    print("  Fetching radio data...")
    radio_data = get_radio_data(session_key, driver_number)
    if not radio_data:
        print("  ✗ No radio data available")
        return []
    print(f"  ✓ Found {len(radio_data)} radio messages")
    
    # Process radio messages (download and transcribe)
    print("  Processing radio messages...")
    radio_list = process_radio_messages(radio_data, model)
    
    if not radio_list:
        print("  ✗ No radio transcripts available")
        return []
    print(f"  ✓ Transcribed {len(radio_list)} radio messages")
    
    # Create training pairs
    print("  Creating training pairs...")
    dataset = create_training_pairs(radio_list, telemetry_df, window_seconds)
    
    if dataset:
        print(f"  ✓ Created {len(dataset)} training pairs")
    else:
        print("  ✗ No training pairs created")
    
    return dataset


def save_dataset(dataset: List[Dict], filename: str, append: bool = False):
    """Save dataset as JSONL file."""
    if not dataset:
        print("Warning: No data to save.")
        return
    
    try:
        mode = 'a' if append else 'w'
        with open(filename, mode, encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        action = "Appended" if append else "Saved"
        print(f"{action} {len(dataset)} training examples to {filename}")
    except Exception as e:
        print(f"Error saving dataset: {e}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("F1 Race Engineer Dataset Builder")
    print("=" * 60)
    print(f"Years: {YEARS}")
    if SESSION_TYPE:
        print(f"Session Type: {SESSION_TYPE}")
    print()
    
    # Check for ffmpeg (required by Whisper)
    print("Checking for ffmpeg...")
    if not check_ffmpeg():
        print("\n⚠ Warning: ffmpeg not found. Transcription may fail.")
        print("  Continuing anyway, but errors may occur...\n")
    
    # Load Whisper model
    print("Loading Whisper model...")
    try:
        model = whisper.load_model(WHISPER_MODEL)
        print(f"✓ Loaded Whisper model: {WHISPER_MODEL}")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return
    
    print()
    
    # Get all sessions from all years
    print("Fetching sessions...")
    sessions = []
    for year in YEARS:
        print(f"  Fetching sessions for {year}...")
        year_sessions = get_sessions(year, SESSION_TYPE)
        if year_sessions:
            sessions.extend(year_sessions)
            print(f"    ✓ Found {len(year_sessions)} sessions")
        else:
            print(f"    ✗ No sessions found for {year}")
    
    if not sessions:
        print("✗ No sessions found for any year")
        return
    
    # Filter sessions if specified
    if SPECIFIC_SESSIONS:
        sessions = [s for s in sessions if s.get('session_key') in SPECIFIC_SESSIONS]
    
    # Limit sessions if specified
    if MAX_SESSIONS:
        sessions = sessions[:MAX_SESSIONS]
    
    print(f"✓ Found {len(sessions)} sessions to process")
    
    # Display sessions
    sessions_df = pd.DataFrame(sessions)
    if not sessions_df.empty and 'session_name' in sessions_df.columns:
        print("\nSessions to process:")
        for idx, session in enumerate(sessions, 1):
            session_key = session.get('session_key', 'N/A')
            session_name = session.get('session_name', 'N/A')
            print(f"  {idx}. {session_name} (Session {session_key})")
    
    print()
    
    # Process all sessions and drivers
    all_dataset = []
    total_processed = 0
    total_pairs = 0
    first_write = True  # Track if this is the first write to file
    
    for session_idx, session in enumerate(sessions, 1):
        session_key = session.get('session_key')
        session_name = session.get('session_name', f'Session {session_key}')
        
        if not session_key:
            print(f"\n⚠ Skipping session {session_idx}: No session_key")
            continue
        
        print(f"\n{'#'*60}")
        print(f"Session {session_idx}/{len(sessions)}: {session_name}")
        print(f"{'#'*60}")
        
        # Get drivers with radio data for this session
        print(f"Finding drivers with radio data...")
        drivers = get_drivers_with_radio(session_key)
        
        if not drivers:
            print(f"  ✗ No drivers with radio data found for session {session_key}")
            continue
        
        # Filter drivers if specified
        if SPECIFIC_DRIVERS:
            drivers = [d for d in drivers if d in SPECIFIC_DRIVERS]
        
        # Limit drivers if specified
        if MAX_DRIVERS_PER_SESSION:
            drivers = drivers[:MAX_DRIVERS_PER_SESSION]
        
        print(f"  ✓ Found {len(drivers)} drivers: {drivers}")
        
        # Process each driver
        for driver_idx, driver_number in enumerate(drivers, 1):
            print(f"\n  Driver {driver_idx}/{len(drivers)}: Driver {driver_number}")
            
            try:
                pairs = process_session_driver(
                    session_key, 
                    driver_number, 
                    model, 
                    TELEMETRY_WINDOW_SECONDS
                )
                
                if pairs:
                    all_dataset.extend(pairs)
                    total_pairs += len(pairs)
                    total_processed += 1
                    
                    # Save incrementally (append after first write)
                    save_dataset(pairs, OUTPUT_FILE, append=not first_write)
                    first_write = False
            except Exception as e:
                print(f"  ✗ Error processing driver {driver_number}: {type(e).__name__}: {e}")
                continue
    
    # Final summary
    print()
    print("=" * 60)
    print("Dataset Creation Summary")
    print("=" * 60)
    print(f"Total sessions processed: {len(sessions)}")
    print(f"Total session-driver combinations processed: {total_processed}")
    print(f"Total training pairs created: {total_pairs}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 60)
    
    if total_pairs == 0:
        print("\n⚠ Warning: No training pairs were created.")
        print("  Check that sessions have both telemetry and radio data.")
    else:
        print(f"\n✓ Successfully created dataset with {total_pairs} training examples!")


if __name__ == "__main__":
    main()
