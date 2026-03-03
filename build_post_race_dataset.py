"""
F1 Post-Race Analysis Dataset Builder

Generates fine-tuning data using Gemini as a "teacher model":
- Gemini sees: Telemetry + Race Events (for rich analysis generation)
- Training input: Telemetry ONLY (what the fine-tuned model will receive)
- Training output: Gemini's professional engineering debrief

This creates a "distilled" model that learns to generate F1 analysis
from minimal telemetry data, trained on Gemini's richer outputs.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

import requests
import json
import time
from pathlib import Path
from datetime import datetime
import fastf1
from google import genai
from google.genai import types

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3-flash-preview"
OUTPUT_FILE = "temp.jsonl"

# Rate limiting settings
REQUEST_DELAY = 0.5  # seconds between API requests

# All 2023 F1 races
RACES_2023 = [
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Australian Grand Prix",
    "Azerbaijan Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Spanish Grand Prix",
    "Canadian Grand Prix",
    "Austrian Grand Prix",
    "British Grand Prix",
    "Hungarian Grand Prix",
    "Belgian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Singapore Grand Prix",
    "Japanese Grand Prix",
    "Qatar Grand Prix",
    "United States Grand Prix",
    "Mexico City Grand Prix",
    "São Paulo Grand Prix",
    "Las Vegas Grand Prix",
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


def get_driver_telemetry(session_key, driver_number):
    """Get telemetry data for a specific driver from OpenF1"""
    print(f"     📡 Fetching OpenF1 telemetry for driver {driver_number}...")
    
    try:
        # Get lap times
        laps_url = f"https://api.openf1.org/v1/laps?session_key={session_key}&driver_number={driver_number}"
        time.sleep(REQUEST_DELAY)
        laps_response = requests.get(laps_url)
        
        if laps_response.status_code != 200:
            return {}
        
        laps = laps_response.json()
        
        # Get position data
        position_url = f"https://api.openf1.org/v1/position?session_key={session_key}&driver_number={driver_number}"
        time.sleep(REQUEST_DELAY)
        position_response = requests.get(position_url)
        
        positions = []
        if position_response.status_code == 200:
            positions = position_response.json()
        
        # Get pit stops
        pit_url = f"https://api.openf1.org/v1/pit?session_key={session_key}&driver_number={driver_number}"
        time.sleep(REQUEST_DELAY)
        pit_response = requests.get(pit_url)
        
        pits = []
        if pit_response.status_code == 200:
            pits = pit_response.json()
        
        # Get stints (tire info)
        stint_url = f"https://api.openf1.org/v1/stints?session_key={session_key}&driver_number={driver_number}"
        time.sleep(REQUEST_DELAY)
        stint_response = requests.get(stint_url)
        
        stints = []
        if stint_response.status_code == 200:
            stints = stint_response.json()
        
        # Get car data (throttle, brake, RPM, speed, gear, DRS)
        car_data_url = f"https://api.openf1.org/v1/car_data?session_key={session_key}&driver_number={driver_number}"
        time.sleep(REQUEST_DELAY)
        car_data_response = requests.get(car_data_url)
        
        car_data = []
        if car_data_response.status_code == 200:
            car_data = car_data_response.json()
        
        telemetry = {
            "laps": laps,  # All laps for training input
            "positions": positions,  # All positions for training input
            "pit_stops": pits,
            "stints": stints,
            "car_data": car_data  # Raw telemetry: throttle, brake, RPM, speed, gear, DRS
        }
        
        print(f"     ✓ Got {len(laps)} laps, {len(pits)} pit stops, {len(stints)} stints, {len(car_data)} car data points")
        return telemetry
    
    except Exception as e:
        print(f"     ⚠️  Telemetry error: {e}")
        return {}


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


def generate_with_gemini(events, telemetry, gp_name, year, driver_num, driver_name):
    """Generate post-race analysis with Gemini using OpenF1 telemetry + FastF1 race events"""
    print(f"  🤖 Generating Gemini analysis (full telemetry + race events)...")
    
    # Format ALL lap times for Gemini
    lap_summary = ""
    if telemetry.get("laps"):
        fastest_lap = min(telemetry["laps"], key=lambda x: x.get("lap_duration", float('inf')) if x.get("lap_duration") else float('inf'))
        slowest_lap = max(telemetry["laps"], key=lambda x: x.get("lap_duration", 0) if x.get("lap_duration") else 0)
        avg_lap = sum(lap.get('lap_duration', 0) for lap in telemetry["laps"] if lap.get('lap_duration')) / max(len([l for l in telemetry["laps"] if l.get('lap_duration')]), 1)
        
        # All lap times
        lap_times = [f"Lap {lap.get('lap_number')}: {lap.get('lap_duration'):.3f}s" 
                    for lap in telemetry["laps"] if lap.get('lap_duration')]
        
        lap_summary = f"""
Lap Statistics:
- Total Laps: {len([l for l in telemetry["laps"] if l.get('lap_duration')])}
- Fastest Lap: Lap {fastest_lap.get('lap_number')} - {fastest_lap.get('lap_duration'):.3f}s
- Slowest Lap: Lap {slowest_lap.get('lap_number')} - {slowest_lap.get('lap_duration'):.3f}s
- Average Lap: {avg_lap:.3f}s

All Lap Times:
""" + "\n".join(lap_times)
    
    # Format ALL pit stops
    pit_summary = ""
    if telemetry.get("pit_stops"):
        pit_summary = "\n\nPit Stops:\n" + "\n".join([f"Lap {pit.get('lap_number')}: {pit.get('pit_duration'):.2f}s" 
                                                       for pit in telemetry["pit_stops"] if pit.get('pit_duration')])
    
    # Format ALL tire stints
    stint_summary = ""
    if telemetry.get("stints"):
        stint_summary = "\n\nTire Stints:\n" + "\n".join([f"Stint {stint.get('stint_number')}: {stint.get('compound')} compound (Laps {stint.get('lap_start')}-{stint.get('lap_end')}, {stint.get('lap_end') - stint.get('lap_start') + 1} laps)" 
                                                          for stint in telemetry["stints"] if stint.get('compound')])
    
    # Format position changes with more detail
    position_summary = ""
    if telemetry.get("positions") and len(telemetry["positions"]) > 0:
        positions = telemetry["positions"]
        start_pos = positions[0].get("position", "?")
        end_pos = positions[-1].get("position", "?")
        
        # Track position changes
        position_changes = []
        prev_pos = start_pos
        for pos_data in positions[::5]:  # Sample every 5th position to avoid overwhelming
            curr_pos = pos_data.get("position")
            if curr_pos != prev_pos:
                position_changes.append(f"Changed to P{curr_pos}")
                prev_pos = curr_pos
        
        position_summary = f"\n\nPosition Data:\n- Start Position: P{start_pos}\n- Finish Position: P{end_pos}"
        if position_changes:
            position_summary += f"\n- Key Position Changes: {', '.join(position_changes[:10])}"
    
    # Format car data summary (sampled telemetry points)
    car_data_summary = ""
    if telemetry.get("car_data"):
        car_data = telemetry["car_data"]
        # Sample key moments (start, middle, end, plus some race points)
        sample_indices = [0, len(car_data)//4, len(car_data)//2, 3*len(car_data)//4, -1]
        samples = []
        for idx in sample_indices:
            if 0 <= idx < len(car_data):
                cd = car_data[idx]
                samples.append(f"  {cd.get('date', 'N/A')}: Speed={cd.get('speed')}km/h, RPM={cd.get('rpm')}, Gear={cd.get('n_gear')}, Throttle={cd.get('throttle')}%, Brake={cd.get('brake')}%, DRS={cd.get('drs')}")
        
        if samples:
            car_data_summary = f"\n\nCar Data Samples ({len(car_data)} total points):\n" + "\n".join(samples)
    
    # Format race events
    events_text = "\n".join([f"Lap {e['lap']}: {e['message']}" for e in events[:30]])  # More events
    
    # Gemini prompt with FULL telemetry data
    prompt = f"""You are an F1 race engineer analyzing {driver_name}'s performance at the {year} {gp_name}.

TELEMETRY DATA:{lap_summary}{pit_summary}{stint_summary}{position_summary}{car_data_summary}

RACE EVENTS (Safety Cars, Flags, Incidents):
{events_text}

Write a professional post-race engineering debrief in a direct technical report format. Start with a subject line identifying the driver and race (e.g., "{driver_name} - {year} {gp_name}"), then proceed directly to the analysis sections:

1. **Overall Performance and Result**: Summarize the race outcome, finishing position, and key achievements
2. **Pace Consistency and Key Lap Times**: Analyze lap-to-lap consistency across the entire race, identify fastest/slowest laps, examine pace windows within each stint
3. **Tire Management and Degradation**: Evaluate tire wear patterns by analyzing lap time degradation curves within each stint, assess compound performance and stint lengths
4. **Strategy Effectiveness**: Analyze pit stop timing and duration, tire compound choices, strategy execution relative to race events
5. **Strengths**: Highlight areas where the driver excelled (pace, consistency, tire management, overtaking)
6. **Weaknesses**: Identify areas for improvement based on lap time variations, slow laps, or inconsistent sectors
7. **Comparison to Teammate/Field**: Compare performance relative to others if position changes or lap time patterns suggest insights
8. **Analysis of Car Data**: Analyze telemetry patterns from car_data including:
   - Throttle application patterns and lift points
   - Braking intensity and points (brake %)
   - RPM utilization and shift points
   - DRS usage effectiveness
   - Speed traces through different race phases
   - Gear selection patterns
9. **Recommendations**: Provide actionable technical insights for future races based on the data

Format: Subject line, then section headings with markdown (###). Write in a concise, technical engineering style.

CRITICAL ANALYSIS GUIDELINES:
- Use ALL the lap time data to identify patterns, degradation curves, and performance windows
- Reference specific lap numbers and times extensively (e.g., "Lap 23: 1:32.456 shows a 0.8s drop from the previous lap")
- Calculate and comment on lap time deltas between consecutive laps to identify tire deg, traffic, or issues
- When you see lap time spikes (>5s slower than average), consider pit stops, safety cars, or incidents from race events
- For tire degradation: Compare first 3 laps of a stint vs last 3 laps - quantify the delta
- When referencing race events: Use inferential language like "the lap time data suggests", "likely corresponding to", "possibly indicating", rather than stating events as absolute facts
- Language: Use driver names or "the driver". Avoid gendered pronouns (he/she/his/her) and plural pronouns (they/them). Rephrase with "the" (e.g., "the driver's pace", "the fastest lap")
- Be quantitative: Always cite specific lap numbers, times, and deltas
- Technical depth: Treat this as an internal engineering debrief, not a public commentary"""
    
    if not GEMINI_API_KEY:
        print(f"  ⚠️  Gemini API key not set, using mock data")
        return "MOCK - Set GEMINI_API_KEY environment variable"
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        completion = response.text.strip()
        print(f"  ✓ Analysis generated ({len(completion)} chars)")
        return completion
    
    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ Gemini error: {error_msg}")
        # Don't fall back to anything - raise the error
        raise Exception(f"Gemini API failed: {error_msg}")


def process_race(year, gp_name):
    """Process one race - generates per-driver training examples"""
    print(f"\n{'='*60}")
    print(f"🏁 {year} {gp_name}")
    print(f"{'='*60}")
    
    # Get session key
    session_key = get_session_key(year, gp_name)
    if not session_key:
        print("⚠️  Skipping - no session key found\n")
        return []
    
    # Get FastF1 events
    events = get_fastf1_events(year, gp_name)
    
    # Get list of drivers from OpenF1
    url = f"https://api.openf1.org/v1/drivers?session_key={session_key}"
    print(f"  🔍 Fetching driver list from OpenF1...")
    time.sleep(REQUEST_DELAY)
    response = requests.get(url)
    
    if response.status_code != 200:
        print("  ⚠️  Skipping - couldn't get driver list\n")
        return []
    
    drivers = response.json()
    # Create mapping of driver_number -> full_name
    driver_map = {d.get('driver_number'): d.get('full_name', f'Driver {d.get("driver_number")}') 
                  for d in drivers if d.get('driver_number')}
    driver_numbers = list(driver_map.keys())
    print(f"  📊 Found {len(driver_numbers)} drivers")
    
    # Generate one training example per driver
    examples = []
    for driver_num in driver_numbers:
        driver_name = driver_map.get(driver_num, f'Driver {driver_num}')
        print(f"\n  ── Driver {driver_num} ({driver_name}) ──")
        
        # Get telemetry for this driver
        telemetry = get_driver_telemetry(session_key, driver_num)
        
        if not telemetry.get("laps"):
            print(f"     ⚠️  No lap data, skipping")
            continue
        
        # Calculate fastest lap
        fastest_lap = min(telemetry["laps"], key=lambda x: x.get("lap_duration", float('inf')) if x.get("lap_duration") else float('inf'))
        
        # Skip if fastest lap has no duration (invalid data)
        if not fastest_lap.get('lap_duration'):
            print(f"     ⚠️  No valid lap times, skipping")
            continue
        
        positions = telemetry.get("positions", [])
        
        # Format telemetry for Gemini prompt (used for generation, not training input)
        lap_times = [f"Lap {lap.get('lap_number')}: {lap.get('lap_duration'):.3f}s" for lap in telemetry["laps"][:10] if lap.get('lap_duration')]
        lap_summary = f"""\nFastest Lap: Lap {fastest_lap.get('lap_number')} - {fastest_lap.get('lap_duration'):.3f}s
First 10 lap times:
""" + "\n".join(lap_times)
        
        pit_summary = ""
        if telemetry.get("pit_stops"):
            pit_summary = "\n\nPit Stops:\n" + "\n".join([f"Lap {pit.get('lap_number')}: {pit.get('pit_duration'):.2f}s" for pit in telemetry["pit_stops"][:5] if pit.get('pit_duration')])
        
        stint_summary = ""
        if telemetry.get("stints"):
            stint_summary = "\n\nTire Stints:\n" + "\n".join([f"Stint {stint.get('stint_number')}: {stint.get('compound')} ({stint.get('lap_start')}-{stint.get('lap_end')})" for stint in telemetry["stints"] if stint.get('compound')])
        
        position_summary = ""
        if positions and len(positions) > 0:
            start_pos = positions[0].get("position", "?")
            end_pos = positions[-1].get("position", "?")
            position_summary = f"\n\nPosition: Started P{start_pos}, Finished P{end_pos}"
        
        # Training input: FULL telemetry data (what fine-tuned model will receive at inference)
        # Downsample car_data to keep size reasonable (~100 points)
        car_data_raw = telemetry.get("car_data", [])
        car_data_sampled = []
        if car_data_raw:
            step = max(1, len(car_data_raw) // 100)
            car_data_sampled = car_data_raw[::step][:100]
        
        training_input_data = {
            "laps": [{"lap": lap.get('lap_number'), "time": lap.get('lap_duration')} 
                     for lap in telemetry["laps"] if lap.get('lap_duration')],
            "positions": [{"position": pos.get('position'), "date": pos.get('date')} 
                         for pos in positions[:100]] if positions else [],  # Sample positions
            "pit_stops": [{"lap": pit.get('lap_number'), "duration": pit.get('pit_duration')} 
                         for pit in telemetry.get("pit_stops", []) if pit.get('pit_duration')],
            "stints": [{"stint": stint.get('stint_number'), "compound": stint.get('compound'), 
                       "lap_start": stint.get('lap_start'), "lap_end": stint.get('lap_end')} 
                      for stint in telemetry.get("stints", []) if stint.get('compound')],
            "car_data": [{"date": cd.get('date'), "rpm": cd.get('rpm'), "speed": cd.get('speed'),
                         "gear": cd.get('n_gear'), "throttle": cd.get('throttle'), 
                         "brake": cd.get('brake'), "drs": cd.get('drs')}
                        for cd in car_data_sampled]
        }
        
        training_input = json.dumps(training_input_data)
        
        # Gemini generation: Uses SUMMARY + race events (cheaper API calls)
        completion = generate_with_gemini(events, telemetry, gp_name, year, driver_num, driver_name)
        
        example = {
            "input": training_input,
            "output": completion,
            "metadata": {
                "year": year,
                "gp": gp_name,
                "driver_number": driver_num,
                "driver_name": driver_name,
                "session_key": session_key,
                "num_events": len(events),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        examples.append(example)
        
        # Write immediately after each driver
        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(example) + '\n')
        
        print(f"     ✅ Saved to {OUTPUT_FILE}")
    
    return examples


def main():
    print("="*60)
    print("F1 Post-Race Dataset Builder")
    print("  Year: 2023")
    print("  Data: OpenF1 telemetry + FastF1 race events")
    print("  Output: Gemini-generated engineering analysis")
    print("="*60)
    
    # Process races
    training_examples = []
    for gp in RACES_2023:
        examples = process_race(2023, gp)
        
        if examples:
            training_examples.extend(examples)
            print(f"  ✅ Race complete: {len(examples)} driver examples\n")
    
    print(f"\n{'='*60}")
    print(f"✅ Complete! Generated {len(training_examples)} examples")
    print(f"   (~{len(training_examples) / len(RACES_2023):.1f} drivers per race average)")
    print(f"   Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
