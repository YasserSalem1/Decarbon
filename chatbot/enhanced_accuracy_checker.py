import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

def calculate_comprehensive_metrics():
    """
    Calculate comprehensive metrics that provide all the information the LLM needs
    including the missing peak hour data
    """
    csv_file = "current_data.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return None
    
    try:
        # Read the data
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        
        # Process timestamp if exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['date'] = df['timestamp'].dt.date
        elif df.index.name == 'timestamp' or 'Unnamed: 0' in df.columns:
            # Handle case where timestamp is in index or first column
            if 'Unnamed: 0' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Unnamed: 0'])
                df = df.drop('Unnamed: 0', axis=1)
            else:
                df['timestamp'] = pd.to_datetime(df.index)
            df['hour'] = df['timestamp'].dt.hour
            df['date'] = df['timestamp'].dt.date
        
        print(f"Columns available: {list(df.columns)}")
        
        # Initialize comprehensive metrics
        metrics = {
            'data_summary': {
                'total_records': len(df),
                'columns': list(df.columns),
                'time_span_hours': len(df) if 'timestamp' in df.columns else 'unknown'
            },
            'solar_analysis': {},
            'wind_analysis': {},
            'peak_hour_analysis': {},
            'hourly_patterns': {},
            'daily_breakdown': [],
            'energy_planning_insights': {}
        }
        
        # Time range analysis
        if 'timestamp' in df.columns:
            metrics['data_summary']['start_time'] = str(df['timestamp'].min())
            metrics['data_summary']['end_time'] = str(df['timestamp'].max())
            metrics['data_summary']['duration_days'] = (df['timestamp'].max() - df['timestamp'].min()).days + 1
        
        # SOLAR ANALYSIS
        if 'solar' in df.columns:
            solar_data = df['solar']
            
            metrics['solar_analysis'] = {
                'statistics': {
                    'average': round(float(solar_data.mean()), 3),
                    'maximum': round(float(solar_data.max()), 1),
                    'minimum': round(float(solar_data.min()), 1),
                    'standard_deviation': round(float(solar_data.std()), 2),
                    'median': round(float(solar_data.median()), 1)
                },
                'distribution': {
                    'zero_radiation_hours': int((solar_data == 0).sum()),
                    'low_radiation_hours': int((solar_data < 100).sum()),
                    'moderate_radiation_hours': int(((solar_data >= 100) & (solar_data < 400)).sum()),
                    'high_radiation_hours': int((solar_data >= 400).sum()),
                    'peak_radiation_hours': int((solar_data > solar_data.quantile(0.9)).sum())
                }
            }
            
            # Peak hour analysis for solar
            if 'hour' in df.columns:
                hourly_solar_avg = df.groupby('hour')['solar'].agg(['mean', 'max', 'count']).round(1)
                peak_hour = int(hourly_solar_avg['mean'].idxmax())
                
                metrics['peak_hour_analysis']['solar'] = {
                    'peak_hour_24h': peak_hour,
                    'peak_hour_formatted': f"{peak_hour:02d}:00",
                    'peak_hour_12h': f"{peak_hour if peak_hour <= 12 else peak_hour-12}:00 {'AM' if peak_hour < 12 else 'PM'}",
                    'peak_hour_radiation': round(float(hourly_solar_avg.loc[peak_hour, 'mean']), 1),
                    'peak_hour_max_radiation': round(float(hourly_solar_avg.loc[peak_hour, 'max']), 1)
                }
                
                # Store full hourly pattern
                metrics['hourly_patterns']['solar'] = {}
                for hour in range(24):
                    if hour in hourly_solar_avg.index:
                        metrics['hourly_patterns']['solar'][f"{hour:02d}:00"] = {
                            'average': round(float(hourly_solar_avg.loc[hour, 'mean']), 1),
                            'maximum': round(float(hourly_solar_avg.loc[hour, 'max']), 1),
                            'hours_of_data': int(hourly_solar_avg.loc[hour, 'count'])
                        }
        
        # WIND ANALYSIS
        if 'wind_speed' in df.columns:
            wind_data = df['wind_speed']
            
            metrics['wind_analysis'] = {
                'statistics': {
                    'average': round(float(wind_data.mean()), 3),
                    'maximum': round(float(wind_data.max()), 1),
                    'minimum': round(float(wind_data.min()), 1),
                    'standard_deviation': round(float(wind_data.std()), 2),
                    'median': round(float(wind_data.median()), 1)
                },
                'wind_categories': {
                    'calm_hours': int((wind_data < 1).sum()),
                    'light_wind_hours': int(((wind_data >= 1) & (wind_data < 5)).sum()),
                    'moderate_wind_hours': int(((wind_data >= 5) & (wind_data < 10)).sum()),
                    'strong_wind_hours': int(((wind_data >= 10) & (wind_data < 15)).sum()),
                    'very_strong_wind_hours': int((wind_data >= 15).sum())
                }
            }
            
            # Peak hour analysis for wind
            if 'hour' in df.columns:
                hourly_wind_avg = df.groupby('hour')['wind_speed'].agg(['mean', 'max', 'count']).round(1)
                peak_hour = int(hourly_wind_avg['mean'].idxmax())
                
                metrics['peak_hour_analysis']['wind'] = {
                    'peak_hour_24h': peak_hour,
                    'peak_hour_formatted': f"{peak_hour:02d}:00",
                    'peak_hour_12h': f"{peak_hour if peak_hour <= 12 else peak_hour-12}:00 {'AM' if peak_hour < 12 else 'PM'}",
                    'peak_hour_wind_speed': round(float(hourly_wind_avg.loc[peak_hour, 'mean']), 1),
                    'peak_hour_max_wind_speed': round(float(hourly_wind_avg.loc[peak_hour, 'max']), 1)
                }
                
                # Store full hourly pattern
                metrics['hourly_patterns']['wind'] = {}
                for hour in range(24):
                    if hour in hourly_wind_avg.index:
                        metrics['hourly_patterns']['wind'][f"{hour:02d}:00"] = {
                            'average': round(float(hourly_wind_avg.loc[hour, 'mean']), 1),
                            'maximum': round(float(hourly_wind_avg.loc[hour, 'max']), 1),
                            'hours_of_data': int(hourly_wind_avg.loc[hour, 'count'])
                        }
        
        # DAILY BREAKDOWN
        if 'date' in df.columns and len(df['date'].unique()) > 1:
            for date, group in df.groupby('date'):
                day_metrics = {
                    'date': str(date),
                    'weekday': group['timestamp'].iloc[0].strftime('%A') if 'timestamp' in group.columns else 'unknown',
                    'total_hours': len(group)
                }
                
                if 'solar' in group.columns:
                    solar_peak_idx = group['solar'].idxmax()
                    day_metrics['solar'] = {
                        'average': round(float(group['solar'].mean()), 1),
                        'maximum': round(float(group['solar'].max()), 1),
                        'minimum': round(float(group['solar'].min()), 1),
                        'peak_hour': int(group.loc[solar_peak_idx, 'hour']) if 'hour' in group.columns else None,
                        'zero_hours': int((group['solar'] == 0).sum()),
                        'productive_hours': int((group['solar'] > 50).sum())
                    }
                
                if 'wind_speed' in group.columns:
                    wind_peak_idx = group['wind_speed'].idxmax()
                    day_metrics['wind'] = {
                        'average': round(float(group['wind_speed'].mean()), 1),
                        'maximum': round(float(group['wind_speed'].max()), 1),
                        'minimum': round(float(group['wind_speed'].min()), 1),
                        'peak_hour': int(group.loc[wind_peak_idx, 'hour']) if 'hour' in group.columns else None,
                        'calm_hours': int((group['wind_speed'] < 1).sum()),
                        'productive_hours': int((group['wind_speed'] > 3).sum())
                    }
                
                metrics['daily_breakdown'].append(day_metrics)
        
        # ENERGY PLANNING INSIGHTS
        metrics['energy_planning_insights'] = {
            'solar_reliability': {
                'productive_percentage': round((metrics['solar_analysis']['distribution']['moderate_radiation_hours'] + 
                                              metrics['solar_analysis']['distribution']['high_radiation_hours']) / len(df) * 100, 1) if 'solar' in df.columns else 0,
                'zero_production_percentage': round(metrics['solar_analysis']['distribution']['zero_radiation_hours'] / len(df) * 100, 1) if 'solar' in df.columns else 0
            },
            'wind_reliability': {
                'productive_percentage': round((metrics['wind_analysis']['wind_categories']['moderate_wind_hours'] + 
                                              metrics['wind_analysis']['wind_categories']['strong_wind_hours'] + 
                                              metrics['wind_analysis']['wind_categories']['very_strong_wind_hours']) / len(df) * 100, 1) if 'wind_speed' in df.columns else 0,
                'calm_percentage': round(metrics['wind_analysis']['wind_categories']['calm_hours'] / len(df) * 100, 1) if 'wind_speed' in df.columns else 0
            }
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating comprehensive metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_metrics(metrics):
    """Display the comprehensive metrics in a readable format"""
    if not metrics:
        print("No metrics to display")
        return
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ENERGY DATA ANALYSIS")
    print("="*60)
    
    # Data Summary
    print(f"\n📊 DATA SUMMARY:")
    print(f"   Total Records: {metrics['data_summary']['total_records']}")
    print(f"   Time Span: {metrics['data_summary'].get('duration_days', 'unknown')} days")
    if 'start_time' in metrics['data_summary']:
        print(f"   Period: {metrics['data_summary']['start_time']} to {metrics['data_summary']['end_time']}")
    
    # Solar Analysis
    if metrics['solar_analysis']:
        print(f"\n☀️ SOLAR RADIATION ANALYSIS:")
        solar = metrics['solar_analysis']['statistics']
        print(f"   Average: {solar['average']} W/m²")
        print(f"   Maximum: {solar['maximum']} W/m²")
        print(f"   Minimum: {solar['minimum']} W/m²")
        print(f"   Standard Deviation: {solar['standard_deviation']} W/m²")
        
        if 'solar' in metrics['peak_hour_analysis']:
            peak = metrics['peak_hour_analysis']['solar']
            print(f"   Peak Hour: {peak['peak_hour_12h']} ({peak['peak_hour_formatted']})")
            print(f"   Peak Hour Average: {peak['peak_hour_radiation']} W/m²")
        
        dist = metrics['solar_analysis']['distribution']
        print(f"   Zero Production Hours: {dist['zero_radiation_hours']}")
        print(f"   High Production Hours: {dist['high_radiation_hours']}")
    
    # Wind Analysis
    if metrics['wind_analysis']:
        print(f"\n💨 WIND SPEED ANALYSIS:")
        wind = metrics['wind_analysis']['statistics']
        print(f"   Average: {wind['average']} m/s")
        print(f"   Maximum: {wind['maximum']} m/s")
        print(f"   Minimum: {wind['minimum']} m/s")
        print(f"   Standard Deviation: {wind['standard_deviation']} m/s")
        
        if 'wind' in metrics['peak_hour_analysis']:
            peak = metrics['peak_hour_analysis']['wind']
            print(f"   Peak Hour: {peak['peak_hour_12h']} ({peak['peak_hour_formatted']})")
            print(f"   Peak Hour Average: {peak['peak_hour_wind_speed']} m/s")
        
        cats = metrics['wind_analysis']['wind_categories']
        print(f"   Calm Hours (<1 m/s): {cats['calm_hours']}")
        print(f"   Strong Wind Hours (>10 m/s): {cats['strong_wind_hours']}")
    
    # Daily Breakdown
    if metrics['daily_breakdown']:
        print(f"\n📅 DAILY BREAKDOWN:")
        for day in metrics['daily_breakdown']:
            print(f"   {day['date']} ({day['weekday']}):")
            if 'solar' in day:
                print(f"      Solar: Avg {day['solar']['average']} W/m², Peak at {day['solar']['peak_hour']}:00")
            if 'wind' in day:
                print(f"      Wind: Avg {day['wind']['average']} m/s, Peak at {day['wind']['peak_hour']}:00")
    
    # Energy Planning Insights
    insights = metrics['energy_planning_insights']
    print(f"\n⚡ ENERGY PLANNING INSIGHTS:")
    print(f"   Solar Productive Hours: {insights['solar_reliability']['productive_percentage']}%")
    print(f"   Solar Zero Production: {insights['solar_reliability']['zero_production_percentage']}%")
    print(f"   Wind Productive Hours: {insights['wind_reliability']['productive_percentage']}%")
    print(f"   Wind Calm Periods: {insights['wind_reliability']['calm_percentage']}%")

def save_metrics_to_file(metrics, filename="comprehensive_metrics.json"):
    """Save metrics to JSON file for use by LLM"""
    try:
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Comprehensive metrics saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving metrics: {e}")
        return False

if __name__ == "__main__":
    print("Enhanced Energy Data Accuracy Checker")
    print("=" * 50)
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics()
    
    if metrics:
        # Display the metrics
        display_metrics(metrics)
        
        # Save to file for LLM use
        save_metrics_to_file(metrics)
        
        print(f"\n✅ Analysis complete!")
        print(f"The LLM now has access to:")
        print(f"   ✓ Precise peak hours for solar and wind")
        print(f"   ✓ Hourly patterns throughout the day")
        print(f"   ✓ Daily breakdown for energy planning")
        print(f"   ✓ Energy reliability statistics")
        print(f"   ✓ Comprehensive distribution analysis")
        
    else:
        print(f"\n❌ Analysis failed. Please check the data file.")