"""
F1 Post-Race Analysis Dataset Builder

Generates fine-tuning data using Gemini as a "teacher model":
- Gemini sees: Telemetry + Race Events (for rich analysis generation)
- Training input: Telemetry ONLY (what the fine-tuned model will receive)
- Training output: Gemini's professional engineering debrief

This creates a "distilled" model that learns to generate F1 analysis
from minimal telemetry data, trained on Gemini's richer outputs.

Data Source: FastF1 (free, open-source F1 telemetry library)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

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
# OUTPUT
OUTPUT_FILE = "post_race_training_data_2025.jsonl"

# All 2023 F1 races
RACES_2023 = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Azerbaijan",
    "Miami",
    "Monaco",
    "Spain",
    "Canada",
    "Austria",
    "Great Britain",
    "Hungary",
    "Belgium",
    "Netherlands",
    "Italy",
    "Singapore",
    "Japan",
    "Qatar",
    "United States",
    "Mexico",
    "São Paulo",
    "Las Vegas",
    "Abu Dhabi",
]

# All 2024 F1 races
RACES_2024 = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Japan",
    "China",
    "Miami",
    "Emilia Romagna",
    "Monaco",
    "Canada",
    "Spain",
    "Austria",
    "Great Britain",
    "Hungary",
    "Belgium",
    "Netherlands",
    "Italy",
    "Azerbaijan",
    "Singapore",
    "United States",
    "Mexico",
    "São Paulo",
    "Las Vegas",
    "Qatar",
    "Abu Dhabi",
]

# All 2025 F1 races
RACES_2025 = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Japan",
    "China",
    "Miami",
    "Emilia Romagna",
    "Monaco",
    "Canada",
    "Spain",
    "Austria",
    "Great Britain",
    "Hungary",
    "Belgium",
    "Netherlands",
    "Italy",
    "Azerbaijan",
    "Singapore",
    "United States",
    "Mexico",
    "São Paulo",
    "Las Vegas",
    "Qatar",
    "Abu Dhabi",
]

# Enable FastF1 cache
cache_dir = Path('cache')
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache('cache')


def get_fastf1_session(year, gp_name):
    """Load FastF1 session for a race"""
    print(f"  📊 Loading FastF1 session for {gp_name}...")
    
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load()
        
        print(f"  ✓ Session loaded successfully")
        return session
    
    except Exception as e:
        print(f"  ❌ FastF1 error: {e}")
        return None


def get_driver_telemetry_fastf1(session, driver_abbr):
    """Get telemetry data for a specific driver from FastF1"""
    print(f"     📡 Fetching FastF1 telemetry for driver {driver_abbr}...")
    
    try:
        # Get all laps for this driver
        driver_laps = session.laps.pick_driver(driver_abbr)
        
        if len(driver_laps) == 0:
            return {}
        
        # Extract lap times
        laps = []
        for idx, lap in driver_laps.iterrows():
            if lap['LapTime'] is not None and not lap['LapTime'] != lap['LapTime']:  # Check not NaN
                laps.append({
                    "lap_number": int(lap['LapNumber']),
                    "lap_duration": lap['LapTime'].total_seconds()
                })
        
        # Get position data (from results)
        positions = []
        if hasattr(session, 'results') and session.results is not None:
            driver_result = session.results[session.results['Abbreviation'] == driver_abbr]
            if len(driver_result) > 0:
                final_pos = driver_result.iloc[0]['Position']
                if final_pos == final_pos:  # Check not NaN
                    positions.append({
                        "position": int(final_pos),
                        "date": str(session.date)
                    })
        
        # Get pit stop data
        pits = []
        for idx, lap in driver_laps.iterrows():
            if lap.get('PitOutTime') is not None and lap.get('PitOutTime') == lap.get('PitOutTime'):
                if lap.get('PitInTime') is not None and lap.get('PitInTime') == lap.get('PitInTime'):
                    pit_duration = (lap['PitOutTime'] - lap['PitInTime']).total_seconds()
                    pits.append({
                        "lap_number": int(lap['LapNumber']),
                        "pit_duration": pit_duration
                    })
        
        # Get tire stints
        stints = []
        if 'Compound' in driver_laps.columns:
            current_compound = None
            stint_start = None
            stint_number = 0
            
            for idx, lap in driver_laps.iterrows():
                lap_compound = lap.get('Compound')
                if lap_compound != current_compound and lap_compound is not None and lap_compound == lap_compound:
                    if current_compound is not None:
                        stints.append({
                            "stint_number": stint_number,
                            "compound": current_compound,
                            "lap_start": stint_start,
                            "lap_end": int(lap['LapNumber']) - 1
                        })
                    current_compound = lap_compound
                    stint_start = int(lap['LapNumber'])
                    stint_number += 1
            
            # Add final stint
            if current_compound is not None:
                stints.append({
                    "stint_number": stint_number,
                    "compound": current_compound,
                    "lap_start": stint_start,
                    "lap_end": int(driver_laps.iloc[-1]['LapNumber'])
                })
        
        # Get car telemetry data (throttle, brake, RPM, speed, gear, DRS)
        car_data = []
        # Sample from multiple laps to get representative data
        sample_lap_indices = [0, len(driver_laps)//4, len(driver_laps)//2, 3*len(driver_laps)//4, max(0, len(driver_laps)-1)]
        
        for lap_idx in sample_lap_indices:
            if 0 <= lap_idx < len(driver_laps):
                try:
                    lap = driver_laps.iloc[lap_idx]
                    telemetry = lap.get_telemetry()
                    
                    if telemetry is not None and len(telemetry) > 0:
                        # Sample every Nth point to avoid too much data
                        sample_step = max(1, len(telemetry) // 20)
                        for tel_idx in range(0, len(telemetry), sample_step):
                            tel = telemetry.iloc[tel_idx]
                            car_data.append({
                                "date": str(tel.get('Time', '')),
                                "rpm": int(tel['RPM']) if 'RPM' in tel.index and tel['RPM'] == tel['RPM'] else 0,
                                "speed": int(tel['Speed']) if 'Speed' in tel.index and tel['Speed'] == tel['Speed'] else 0,
                                "n_gear": int(tel['nGear']) if 'nGear' in tel.index and tel['nGear'] == tel['nGear'] else 0,
                                "throttle": int(tel['Throttle']) if 'Throttle' in tel.index and tel['Throttle'] == tel['Throttle'] else 0,
                                "brake": int(tel['Brake']) if 'Brake' in tel.index else 0,
                                "drs": int(tel['DRS']) if 'DRS' in tel.index and tel['DRS'] == tel['DRS'] else 0
                            })
                except:
                    continue
        
        telemetry_data = {
            "laps": laps,
            "positions": positions,
            "pit_stops": pits,
            "stints": stints,
            "car_data": car_data
        }
        
        print(f"     ✓ Got {len(laps)} laps, {len(pits)} pit stops, {len(stints)} stints, {len(car_data)} car data points")
        return telemetry_data
    
    except Exception as e:
        print(f"     ⚠️  Telemetry error: {e}")
        return {}


def get_fastf1_events(session):
    """Get race events from FastF1 session"""
    print(f"  🏁 Extracting race events...")
    
    try:
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
        print(f"  ⚠️  Error extracting events: {e}")
        return []


def generate_with_gemini(events, telemetry, gp_name, year, driver_abbr, driver_name):
    """Generate post-race analysis with Gemini using FastF1 telemetry + race events"""
    print(f"  🤖 Generating Gemini analysis (full telemetry + race events)...")
    
    # Format ALL lap times for Gemini
    lap_summary = ""
    if telemetry.get("laps"):
        fastest_lap = min(telemetry["laps"], key=lambda x: x.get("lap_duration", float('inf')))
        slowest_lap = max(telemetry["laps"], key=lambda x: x.get("lap_duration", 0))
        avg_lap = sum(lap.get('lap_duration', 0) for lap in telemetry["laps"]) / max(len(telemetry["laps"]), 1)
        
        # All lap times
        lap_times = [f"Lap {lap.get('lap_number')}: {lap.get('lap_duration'):.3f}s" 
                    for lap in telemetry["laps"]]
        
        lap_summary = f"""
Lap Statistics:
- Total Laps: {len(telemetry["laps"])}
- Fastest Lap: Lap {fastest_lap.get('lap_number')} - {fastest_lap.get('lap_duration'):.3f}s
- Slowest Lap: Lap {slowest_lap.get('lap_number')} - {slowest_lap.get('lap_duration'):.3f}s
- Average Lap: {avg_lap:.3f}s

All Lap Times:
""" + "\n".join(lap_times)
    
    # Format ALL pit stops
    pit_summary = ""
    if telemetry.get("pit_stops"):
        pit_summary = "\n\nPit Stops:\n" + "\n".join([f"Lap {pit.get('lap_number')}: {pit.get('pit_duration'):.2f}s" 
                                                       for pit in telemetry["pit_stops"]])
    
    # Format ALL tire stints
    stint_summary = ""
    if telemetry.get("stints"):
        stint_summary = "\n\nTire Stints:\n" + "\n".join([f"Stint {stint.get('stint_number')}: {stint.get('compound')} compound (Laps {stint.get('lap_start')}-{stint.get('lap_end')}, {stint.get('lap_end') - stint.get('lap_start') + 1} laps)" 
                                                          for stint in telemetry["stints"]])
    
    # Format position data
    position_summary = ""
    if telemetry.get("positions") and len(telemetry["positions"]) > 0:
        final_pos = telemetry["positions"][0].get("position", "?")
        position_summary = f"\n\nFinish Position: P{final_pos}"
    
    # Format car data summary (sampled telemetry points)
    car_data_summary = ""
    if telemetry.get("car_data"):
        car_data = telemetry["car_data"]
        # Sample key moments
        sample_indices = [0, len(car_data)//4, len(car_data)//2, 3*len(car_data)//4, max(0, len(car_data)-1)]
        samples = []
        for idx in sample_indices:
            if 0 <= idx < len(car_data):
                cd = car_data[idx]
                samples.append(f"  Speed={cd.get('speed')}km/h, RPM={cd.get('rpm')}, Gear={cd.get('n_gear')}, Throttle={cd.get('throttle')}%, Brake={cd.get('brake')}%, DRS={cd.get('drs')}")
        
        if samples:
            car_data_summary = f"\n\nCar Data Samples ({len(car_data)} total points):\n" + "\n".join(samples)
    
    # Format race events
    events_text = "\n".join([f"Lap {e['lap']}: {e['message']}" for e in events[:30]])
    
    # Gemini prompt with FULL telemetry data
    prompt = f"""You are an F1 race engineer analyzing {driver_name}'s performance at the {year} {gp_name} Grand Prix.

TELEMETRY DATA:{lap_summary}{pit_summary}{stint_summary}{position_summary}{car_data_summary}

RACE EVENTS (Safety Cars, Flags, Incidents) - FOR CONTEXT ONLY:
{events_text}

**CRITICAL: RACE EVENTS USAGE RULES**
The race events above are PROVIDED FOR YOUR CONTEXT ONLY to help you understand what happened during the race.
HOWEVER, the fine-tuned model will NOT have access to these race events - it will only see telemetry data.

Therefore, you MUST follow these rules:
1. DO NOT reference specific race event details by name (e.g., "starting procedure infringement on Lap 6", "Hamilton track limits incident")
2. DO NOT quote race control messages directly
3. DO NOT mention specific penalties, flags, or incidents unless they can be CLEARLY INFERRED from telemetry alone
4. ONLY use race events to understand the context, then write analysis based on what telemetry reveals

ALLOWED: "Lap 28 (114s) and Lap 42 (139s) suggest Safety Car periods"
FORBIDDEN: "The starting procedure infringement noted on Lap 6..."

**DATA QUALITY AWARENESS**
Be aware that telemetry data may have limitations:

- Lap sequences may have gaps due to data collection issues
- Some races may show fewer laps than expected if data logging stopped early
- Telemetry values occasionally contain anomalies from sensor errors
- Stint tracking may be incomplete in some cases

**When you encounter data limitations:**
- Mention them naturally within the relevant analysis section (e.g., "Note: lap data available for 57 of 58 laps" or "Stint data shows 2 stints; additional stops may have occurred")
- Work with the available data and state what conclusions can/cannot be drawn
- In the Analysis of Car Data section, if you notice unusual telemetry values (e.g., gear readings outside 0-8 range), briefly note "some telemetry readings appear anomalous"
- Keep the tone professional and matter-of-fact, not alarming

Write a professional post-race engineering debrief in a direct technical report format. Start with a subject line identifying the driver and race (e.g., "{driver_name} - {year} {gp_name} Grand Prix"), then proceed directly to the analysis sections:

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
        raise Exception(f"Gemini API failed: {error_msg}")


def get_processed_entries():
    """Get set of already processed (gp, driver_abbr) combinations"""
    processed = set()
    try:
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    gp = data['metadata']['gp']
                    driver = data['metadata']['driver_abbr']
                    processed.add((gp, driver))
    except FileNotFoundError:
        pass
    return processed


def process_race(year, gp_name, skip_until_gp=None, skip_until_driver=None):
    """Process one race - generates per-driver training examples"""
    print(f"\n{'='*60}")
    print(f"🏁 {year} {gp_name} Grand Prix")
    print(f"{'='*60}")
    
    # Load FastF1 session
    session = get_fastf1_session(year, gp_name)
    if not session:
        print("⚠️  Skipping - no session found\n")
        return []
    
    # Get race events
    events = get_fastf1_events(session)
    
    # Get list of drivers
    drivers = session.results['Abbreviation'].tolist()
    driver_names = dict(zip(session.results['Abbreviation'], session.results['FullName']))
    
    print(f"  📊 Found {len(drivers)} drivers")
    
    # Get already processed entries
    processed = get_processed_entries()
    
    # Generate one training example per driver
    examples = []
    skip_mode = skip_until_gp is not None and gp_name != skip_until_gp
    
    for driver_abbr in drivers:
        driver_name = driver_names.get(driver_abbr, f'Driver {driver_abbr}')
        
        # Skip logic for resuming
        if skip_mode:
            print(f"\n  ── {driver_abbr} ({driver_name}) [SKIPPED - before {skip_until_gp}] ──")
            continue
        
        if skip_until_driver and driver_abbr != skip_until_driver:
            print(f"\n  ── {driver_abbr} ({driver_name}) [SKIPPED - before {skip_until_driver}] ──")
            continue
        
        # After finding the resume driver, stop skipping
        if skip_until_driver and driver_abbr == skip_until_driver:
            skip_until_driver = None
        
        # Skip already processed
        if (gp_name, driver_abbr) in processed:
            print(f"\n  ── {driver_abbr} ({driver_name}) [ALREADY PROCESSED] ──")
            continue
        
        print(f"\n  ── {driver_abbr} ({driver_name}) ──")
        
        # Get telemetry for this driver
        telemetry = get_driver_telemetry_fastf1(session, driver_abbr)
        
        if not telemetry.get("laps"):
            print(f"     ⚠️  No lap data, skipping")
            continue
        
        # Training input: FULL telemetry data (what fine-tuned model will receive at inference)
        # Downsample car_data to keep size reasonable (~100 points)
        car_data_raw = telemetry.get("car_data", [])
        car_data_sampled = []
        if car_data_raw:
            step = max(1, len(car_data_raw) // 100)
            car_data_sampled = car_data_raw[::step][:100]
        
        training_input_data = {
            "laps": [{"lap": lap.get('lap_number'), "time": lap.get('lap_duration')} 
                     for lap in telemetry["laps"]],
            "positions": [{"position": pos.get('position'), "date": pos.get('date')} 
                         for pos in telemetry.get("positions", [])],
            "pit_stops": [{"lap": pit.get('lap_number'), "duration": pit.get('pit_duration')} 
                         for pit in telemetry.get("pit_stops", [])],
            "stints": [{"stint": stint.get('stint_number'), "compound": stint.get('compound'), 
                       "lap_start": stint.get('lap_start'), "lap_end": stint.get('lap_end')} 
                      for stint in telemetry.get("stints", [])],
            "car_data": car_data_sampled
        }
        
        training_input = json.dumps(training_input_data)
        
        # Gemini generation: Uses FULL telemetry + race events
        completion = generate_with_gemini(events, telemetry, gp_name, year, driver_abbr, driver_name)
        
        example = {
            "input": training_input,
            "output": completion,
            "metadata": {
                "year": year,
                "gp": gp_name,
                "driver_abbr": driver_abbr,
                "driver_name": driver_name,
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
    print("  Year: 2025")
    print("  Data: FastF1 telemetry + race events")
    print("  Output: Gemini-generated engineering analysis")
    print("="*60)
    
    # Resume configuration
    RESUME_FROM_GP = None  # Start from beginning
    RESUME_FROM_DRIVER = None  # Use duplicate detection to skip completed drivers
    
    # Process races
    training_examples = []
    skip_until_gp = RESUME_FROM_GP
    skip_until_driver = RESUME_FROM_DRIVER
    
    for gp in RACES_2025:
        # Pass skip parameters
        examples = process_race(2025, gp, skip_until_gp=skip_until_gp, skip_until_driver=skip_until_driver)
        
        # Once we reach the resume GP, clear the skip flags
        if gp == RESUME_FROM_GP:
            skip_until_gp = None
            skip_until_driver = None  # Clear driver skip after resume GP
        
        if examples:
            training_examples.extend(examples)
            print(f"  ✅ Race complete: {len(examples)} driver examples\n")
    
    print(f"\n{'='*60}")
    print(f"✅ Complete! Generated {len(training_examples)} examples")
    print(f"   (~{len(training_examples) / len(RACES_2025):.1f} drivers per race average)")
    print(f"   Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
