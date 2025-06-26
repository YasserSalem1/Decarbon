import requests
from datetime import datetime, timedelta
import pandas as pd
import time

def get_energy_data(latitude, longitude, days=3):
    """
    Try multiple APIs to get solar/wind data with fallback logic
    Returns DataFrame with columns: timestamp, solar, wind_speed
    """
   
    
    # Fallback to Open-Meteo if NASA fails
    print("NASA failed, trying Open-Meteo...")
    openmeteo_data = try_openmeteo_api(latitude, longitude, days)
    if openmeteo_data is not None and not openmeteo_data.empty:
        return openmeteo_data
    
    # Final fallback to NREL (solar only)
    print("Open-Meteo failed, trying NREL NSRDB (solar only)...")
    nrel_data = try_nrel_api(latitude, longitude, days)
    return nrel_data

def try_openmeteo_api(lat, lon, days):
    """Fallback to Open-Meteo API"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["shortwave_radiation", "windspeed_10m"],
            "forecast_days": days,
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        timestamps = [datetime.strptime(t, "%Y-%m-%dT%H:%M") 
                      for t in data["hourly"]["time"]]
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "solar": data["hourly"]["shortwave_radiation"],
            "wind_speed": data["hourly"]["windspeed_10m"]
        })
        
        return df.set_index("timestamp")
    
    except Exception as e:
        print(f"Open-Meteo Error: {str(e)}")
    return None

def try_nrel_api(lat, lon, days):
    """Final fallback to NREL NSRDB (solar only)"""
    try:
        # Note: Requires API key (free registration)
        api_key = "DEMO_KEY"  # Replace with your key
        url = f"https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"
        params = {
            "api_key": api_key,
            "lat": lat,
            "lon": lon,
            "email": "your@email.com",
            "names": datetime.now().strftime("%Y"),
            "interval": "60",
            "utc": "false"
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        # Process CSV response
        df = pd.read_csv(response.url)
        df = df.iloc[:24*days]  # Get requested days
        df["timestamp"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour"]]
                                         .rename(columns={"Hour":"hour"}))
        df = df[["timestamp", "GHI"]].rename(columns={"GHI":"solar"})
        df["wind_speed"] = None  # NREL doesn't provide wind
        
        return df.set_index("timestamp")
    
    except Exception as e:
        print(f"NREL Error: {str(e)}")
    return None

# Example Usage
if __name__ == "__main__":
    # Test coordinates (New York)
    energy_data = get_energy_data(latitude=40.71, longitude=-74.01, days=3)
    
    if energy_data is not None and not energy_data.empty:
        print("\nSuccessfully fetched energy data:")
        print(energy_data.head(24))  # Show first day
        energy_data.to_csv("energy_forecast.csv")
        print("\nSaved to 'energy_forecast.csv'")
    else:
        print("\nFailed to get data from all APIs. Possible solutions:")
        print("1. Try different coordinates")
        print("2. Check internet connection")
        print("3. Register for NREL API key (free) at https://developer.nrel.gov/signup/")