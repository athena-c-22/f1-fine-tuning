#!/usr/bin/env python3
"""
Script to build F1 telemetry → analysis training dataset using OpenF1 API + LLM
"""

import json
import os
import re
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Or set GEMINI_API_KEY as an environment variable manually.")

# ================= CONFIGURATION =================
CONFIG = {
    # API settings
    "openf1_base_url": "https://api.openf1.org/v1",
    "output_dir": Path("f1_training_dataset"),
    "jsonl_output": Path("f1_telemetry_analysis.jsonl"),
    
    # LLM settings (using Gemini Pro)
    "llm": {
        "provider": "gemini",           # Google Gemini
        "api_key": os.getenv("GEMINI_API_KEY", ""),  # Load from .env file
        "model": "gemini-3-flash-preview",          # or "gemini-1.5-pro" for latest
        "max_output_tokens": 1800,
        "temperature": 0.7,
        "request_delay": 6,             # Seconds to wait between requests (6s = 10 RPM for flash models free tier)
        "max_retries": 5,               # Maximum retry attempts for rate limits
        "retry_base_delay": 2,          # Base delay for exponential backoff (seconds)
    },
    
    # Data selection
    "years": [2023, 2024, 2025],
    "session_types": ["Race"],      # can add "Qualifying", "Sprint"
    "max_races_per_year": 10,       # limit for initial dataset
    "drivers": None,                # None = all drivers, or list e.g. ["1", "16", "44", "63"]
    
    # Processing
    "downsample_strategy": "per_lap_average",  # or "every_n_points" (not implemented yet)
    "max_car_data_points": 100,     # Reduced: only essential car data points
    "max_laps_to_include": 5,       # Only include fastest/slowest/key laps
    "summarize_position": True,      # Summarize position data instead of full history
    "max_pit_stops": 10,            # Limit pit stop records
}

# ================= PROMPT TEMPLATE =================
ANALYSIS_PROMPT = """You are an expert Formula 1 race engineer and analyst.

Given the following telemetry and session data for {driver_name} ({driver_number}) 
in the {year} {grand_prix} {session_name}, write a detailed, professional race debrief.

IMPORTANT FORMATTING INSTRUCTIONS:
- Start with "Subject: [brief descriptive subject line]" on the first line
- Do NOT include FROM:, TO:, Date:, or any other email headers
- Write the analysis directly after the Subject line
- Use a professional report format, not an email format

Focus on:
• Overall performance & result
• Pace consistency & key lap times
• Tire management & degradation
• Strategy effectiveness (pit stops, timing)
• Strengths (sectors, overtaking, DRS usage)
• Weaknesses (braking, traction, errors)
• Comparison to teammate/field (when data available)

Use clear structure with paragraphs and bullet points for metrics.
Base your analysis strictly on the provided data only.
Do NOT add external knowledge or fabricate information.

Data:
{data_json}

Write 800-1500 words in a professional, analytical tone.
"""


def get_sessions(year: int) -> List[Dict]:
    """Get all race sessions for a given year"""
    url = f"{CONFIG['openf1_base_url']}/sessions?year={year}&session_name=Race"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching sessions for {year}: {e}")
        return []


def get_driver_name(session_key: int, driver_number: str) -> Optional[str]:
    """Try to get driver full name (fallback to number)"""
    try:
        url = f"{CONFIG['openf1_base_url']}/drivers?session_key={session_key}&driver_number={driver_number}"
        data = requests.get(url, timeout=10).json()
        if data and len(data) > 0:
            return data[0].get("full_name", f"Driver {driver_number}")
    except:
        pass
    return f"Driver {driver_number}"


def fetch_driver_telemetry(session_key: int, driver_number: str) -> Dict:
    """Fetch main telemetry data for one driver in one session"""
    data = {
        "session_key": session_key,
        "driver_number": driver_number,
        "laps": [],
        "car_data": [],
        "pit": [],
        "position": [],
        "stints": [],
        "summary": {}
    }

    endpoints = [
        ("laps",       f"laps?session_key={session_key}&driver_number={driver_number}"),
        ("car_data",   f"car_data?session_key={session_key}&driver_number={driver_number}"),
        ("pit",        f"pit?session_key={session_key}&driver_number={driver_number}"),
        ("position",   f"position?session_key={session_key}&driver_number={driver_number}"),
        ("stints",     f"stints?session_key={session_key}&driver_number={driver_number}"),
    ]

    for name, endpoint in endpoints:
        try:
            url = f"{CONFIG['openf1_base_url']}/{endpoint}"
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data[name] = resp.json()
            print(f"  ✓ Fetched {name}: {len(data[name])} records")
            time.sleep(0.6)  # polite rate limiting
        except Exception as e:
            print(f"  ✗ Failed to fetch {name}: {e}")

    # Basic summary stats
    if data["laps"]:
        lap_durations = [lap.get("lap_duration", 0) for lap in data["laps"] if lap.get("lap_duration")]
        if lap_durations:
            data["summary"]["avg_lap"] = sum(lap_durations) / len(lap_durations)
            data["summary"]["fastest_lap"] = min(lap_durations)
            data["summary"]["slowest_lap"] = max(lap_durations)

    return data


def downsample_car_data(car_data: List[Dict]) -> List[Dict]:
    """Aggressive downsampling - keep only key points (start, end, every 20 seconds)"""
    if not car_data:
        return []

    if len(car_data) <= CONFIG["max_car_data_points"]:
        return car_data

    result = []
    # Always include first point
    result.append(car_data[0])
    
    # Sample every N points to get roughly max_car_data_points total
    step = max(1, len(car_data) // CONFIG["max_car_data_points"])
    
    for i in range(step, len(car_data) - 1, step):
        if len(result) >= CONFIG["max_car_data_points"] - 1:
            break
        result.append(car_data[i])
    
    # Always include last point
    if len(car_data) > 1 and len(result) < CONFIG["max_car_data_points"]:
        result.append(car_data[-1])
    
    return result


def summarize_laps(laps: List[Dict]) -> Dict:
    """Summarize lap data - keep only key metrics and fastest/slowest laps"""
    if not laps:
        return {}
    
    valid_laps = [lap for lap in laps if lap.get("lap_duration")]
    if not valid_laps:
        return {"total_laps": len(laps)}
    
    lap_durations = [lap.get("lap_duration") for lap in valid_laps]
    sorted_laps = sorted(valid_laps, key=lambda x: x.get("lap_duration", float('inf')))
    
    summary = {
        "total_laps": len(laps),
        "avg_lap_time": sum(lap_durations) / len(lap_durations),
        "fastest_lap": {
            "lap_number": sorted_laps[0].get("lap_number"),
            "duration": sorted_laps[0].get("lap_duration"),
            "sector1": sorted_laps[0].get("sector_1_session_time"),
            "sector2": sorted_laps[0].get("sector_2_session_time"),
            "sector3": sorted_laps[0].get("sector_3_session_time"),
        },
        "slowest_lap": {
            "lap_number": sorted_laps[-1].get("lap_number"),
            "duration": sorted_laps[-1].get("lap_duration"),
        },
    }
    
    # Include a few key laps (if available)
    key_laps = []
    max_key_laps = CONFIG.get("max_laps_to_include", 5)
    for lap in sorted_laps[:max_key_laps]:
        key_laps.append({
            "lap_number": lap.get("lap_number"),
            "duration": lap.get("lap_duration"),
            "sector1": lap.get("sector_1_session_time"),
            "sector2": lap.get("sector_2_session_time"),
            "sector3": lap.get("sector_3_session_time"),
        })
    
    if key_laps:
        summary["key_laps"] = key_laps
    
    return summary


def summarize_position(position_data: List[Dict]) -> Dict:
    """Summarize position data - keep start, end, best, worst, and key changes"""
    if not position_data:
        return {}
    
    if len(position_data) <= 10:
        return {"positions": position_data}  # Small enough, keep all
    
    positions = [p.get("position", 0) for p in position_data if p.get("position")]
    if not positions:
        return {}
    
    summary = {
        "start_position": position_data[0].get("position"),
        "end_position": position_data[-1].get("position"),
        "best_position": min(positions),
        "worst_position": max(positions),
        "total_position_changes": len([i for i in range(1, len(positions)) if positions[i] != positions[i-1]]),
    }
    
    # Include key position changes (every 5th record)
    key_positions = []
    step = max(1, len(position_data) // 10)
    for i in range(0, len(position_data), step):
        key_positions.append({
            "date": position_data[i].get("date"),
            "position": position_data[i].get("position"),
        })
    summary["key_positions"] = key_positions[:10]  # Max 10 key positions
    
    return summary


def filter_telemetry_data(telemetry: Dict) -> Dict:
    """Filter and summarize telemetry to reduce token usage"""
    filtered = {
        "session_key": telemetry.get("session_key"),
        "driver_number": telemetry.get("driver_number"),
        "year": telemetry.get("year"),
        "grand_prix": telemetry.get("grand_prix"),
        "session_name": telemetry.get("session_name"),
    }
    
    # Summarize laps instead of including all
    if "laps" in telemetry:
        filtered["laps_summary"] = summarize_laps(telemetry["laps"])
    
    # Aggressively downsample car_data
    if "car_data" in telemetry:
        filtered["car_data"] = downsample_car_data(telemetry["car_data"])
    
    # Keep pit stops but limit
    if "pit" in telemetry:
        filtered["pit"] = telemetry["pit"][:CONFIG.get("max_pit_stops", 10)]
    
    # Summarize position data
    if "position" in telemetry:
        if CONFIG.get("summarize_position", True):
            filtered["position_summary"] = summarize_position(telemetry["position"])
        else:
            filtered["position"] = telemetry["position"][:50]  # Limit to 50 points
    
    # Keep stints (important for strategy, usually small)
    if "stints" in telemetry:
        filtered["stints"] = telemetry["stints"]
    
    # Keep summary if exists
    if "summary" in telemetry:
        filtered["summary"] = telemetry["summary"]
    
    return filtered


def generate_analysis(telemetry_data: Dict, driver_name: str) -> str:
    """Call LLM to generate race analysis"""
    if CONFIG["llm"]["provider"] == "gemini":
        from google import genai
        from google.genai import types
        
        # Configure Gemini API
        api_key = CONFIG["llm"]["api_key"]
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
        
        client = genai.Client(api_key=api_key)

        data_str = json.dumps(telemetry_data, indent=2)
        prompt = ANALYSIS_PROMPT.format(
            driver_name=driver_name,
            driver_number=telemetry_data["driver_number"],
            year=telemetry_data.get("year", "unknown"),
            grand_prix=telemetry_data.get("grand_prix", "unknown"),
            session_name=telemetry_data.get("session_name", "Race"),
            data_json=data_str
        )

        # Retry logic with exponential backoff for rate limits
        max_retries = CONFIG["llm"].get("max_retries", 5)
        retry_base_delay = CONFIG["llm"].get("retry_base_delay", 2)
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=CONFIG["llm"]["model"],
                    contents=types.Part.from_text(text=prompt),
                    config=types.GenerateContentConfig(
                        temperature=CONFIG["llm"]["temperature"],
                        max_output_tokens=CONFIG["llm"]["max_output_tokens"],
                    )
                )
                return response.text.strip()
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    # Try to extract retry delay from error message
                    retry_delay = None
                    if "retry in" in error_str.lower():
                        delay_match = re.search(r'retry in ([\d.]+)s', error_str, re.IGNORECASE)
                        if delay_match:
                            retry_delay = float(delay_match.group(1))
                    
                    # Use extracted delay or exponential backoff
                    if retry_delay:
                        wait_time = int(retry_delay) + 5  # Add 5s buffer
                    else:
                        wait_time = retry_base_delay * (2 ** attempt)  # Exponential backoff
                    
                    if attempt < max_retries - 1:
                        print(f"  ⚠ Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  ✗ Rate limit error after {max_retries} attempts: {error_str[:200]}")
                        return f"[ERROR] Rate limit exceeded after {max_retries} retries. Please wait and try again later."
                else:
                    # Non-rate-limit error
                    print(f"  ✗ LLM error: {error_str[:200]}")
                    return f"[ERROR] Could not generate analysis: {error_str[:200]}"
        
        return "[ERROR] Failed after all retry attempts"

    elif CONFIG["llm"]["provider"] == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=CONFIG["llm"]["api_key"])

        data_str = json.dumps(telemetry_data, indent=2)
        prompt = ANALYSIS_PROMPT.format(
            driver_name=driver_name,
            driver_number=telemetry_data["driver_number"],
            year=telemetry_data.get("year", "unknown"),
            grand_prix=telemetry_data.get("grand_prix", "unknown"),
            session_name=telemetry_data.get("session_name", "Race"),
            data_json=data_str
        )

        try:
            response = client.chat.completions.create(
                model=CONFIG["llm"]["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=CONFIG["llm"].get("max_tokens", 1800),
                temperature=CONFIG["llm"]["temperature"],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return f"[ERROR] Could not generate analysis: {str(e)}"

    else:
        raise NotImplementedError(f"LLM provider '{CONFIG['llm']['provider']}' not implemented yet")


def main():
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    with open(CONFIG["jsonl_output"], "w", encoding="utf-8") as jsonl_file:
        total = 0
        
        for year in CONFIG["years"]:
            print(f"\n=== Processing year {year} ===")
            sessions = get_sessions(year)[:CONFIG["max_races_per_year"]]
            
            for session in sessions:
                session_key = session["session_key"]
                country = session.get("country_name", "Unknown")
                print(f"\n→ {year} {country} Grand Prix (key: {session_key})")
                
                # Get all drivers in this session
                try:
                    drivers_url = f"{CONFIG['openf1_base_url']}/drivers?session_key={session_key}"
                    drivers = requests.get(drivers_url).json()
                    driver_numbers = [d["driver_number"] for d in drivers]
                except:
                    print("  Could not fetch drivers list")
                    continue

                if CONFIG["drivers"]:
                    driver_numbers = [n for n in driver_numbers if n in CONFIG["drivers"]]

                for driver_no in driver_numbers:
                    print(f"  Processing driver {driver_no}...")
                    
                    telemetry = fetch_driver_telemetry(session_key, driver_no)
                    
                    # Add context fields
                    telemetry["year"] = year
                    telemetry["grand_prix"] = country
                    telemetry["session_name"] = session.get("session_name", "Race")
                    
                    # Filter and summarize telemetry to reduce token usage
                    filtered_telemetry = filter_telemetry_data(telemetry)
                    
                    driver_name = get_driver_name(session_key, driver_no)
                    
                    print(f"  Generating analysis for {driver_name}...")
                    print(f"  Data size: {len(json.dumps(telemetry))} → {len(json.dumps(filtered_telemetry))} chars (reduced by {100 * (1 - len(json.dumps(filtered_telemetry)) / len(json.dumps(telemetry))):.1f}%)")
                    analysis = generate_analysis(filtered_telemetry, driver_name)
                    
                    # Save example (save filtered version to reduce file size)
                    example = {
                        "input": json.dumps(filtered_telemetry, ensure_ascii=False),
                        "output": analysis,
                        "metadata": {
                            "year": year,
                            "grand_prix": country,
                            "driver_number": driver_no,
                            "driver_name": driver_name,
                            "session_key": session_key,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                    
                    jsonl_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                    jsonl_file.flush()
                    
                    total += 1
                    print(f"  ✓ Saved example #{total}")
                    
                    # Wait between requests to avoid rate limits
                    request_delay = CONFIG["llm"].get("request_delay", 6)
                    if request_delay > 0:
                        print(f"  ⏳ Waiting {request_delay}s before next request...")
                        time.sleep(request_delay)

    print(f"\nDataset building complete!")
    print(f"Total examples created: {total}")
    print(f"Saved to: {CONFIG['jsonl_output']}")


if __name__ == "__main__":
    main()