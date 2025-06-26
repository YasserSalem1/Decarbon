import pandas as pd
import numpy as np
import os
from datetime import datetime

def check_analysis_accuracy():
    """
    Mathematically verify the accuracy of the LLM's analysis claims
    against the actual data in current_data.csv
    """
    csv_file = "current_data.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    try:
        # Read the data
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        print(f"Columns: {list(df.columns)}\n")
        
        # Convert timestamp if exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
        
        # LLM Claims to verify
        llm_claims = {
            "avg_solar": 306,  # W/m²
            "max_solar": 835,  # W/m²
            "min_solar": 0.0,  # W/m²
            "peak_solar_time": "15:00 (3 PM)",
            "avg_wind": 9.4,   # m/s
            "min_wind": 4.3,   # m/s
            "max_wind": 17.3,  # m/s
            "peak_wind_time": "17:00 (5 PM)"
        }
        
        print("=== ACCURACY VERIFICATION ===")
        print("LLM Claims vs Actual Data")
        print("=" * 50)
        
        accuracy_results = {}
        
        # Check Solar Statistics
        if 'solar' in df.columns:
            actual_avg_solar = df['solar'].mean()
            actual_max_solar = df['solar'].max()
            actual_min_solar = df['solar'].min()
            
            print(f"📊 SOLAR RADIATION ANALYSIS:")
            print(f"   Average Solar:")
            print(f"     LLM Claim: {llm_claims['avg_solar']} W/m²")
            print(f"     Actual:    {actual_avg_solar:.1f} W/m²")
            solar_avg_error = abs(llm_claims['avg_solar'] - actual_avg_solar)
            solar_avg_accuracy = max(0, 100 - (solar_avg_error / actual_avg_solar * 100))
            print(f"     Error:     {solar_avg_error:.1f} W/m² ({100-solar_avg_accuracy:.1f}% error)")
            print(f"     Accuracy:  {solar_avg_accuracy:.1f}%")
            accuracy_results['solar_avg'] = solar_avg_accuracy
            
            print(f"\n   Maximum Solar:")
            print(f"     LLM Claim: {llm_claims['max_solar']} W/m²")
            print(f"     Actual:    {actual_max_solar:.1f} W/m²")
            solar_max_error = abs(llm_claims['max_solar'] - actual_max_solar)
            solar_max_accuracy = max(0, 100 - (solar_max_error / actual_max_solar * 100))
            print(f"     Error:     {solar_max_error:.1f} W/m²")
            print(f"     Accuracy:  {solar_max_accuracy:.1f}%")
            accuracy_results['solar_max'] = solar_max_accuracy
            
            print(f"\n   Minimum Solar:")
            print(f"     LLM Claim: {llm_claims['min_solar']} W/m²")
            print(f"     Actual:    {actual_min_solar:.1f} W/m²")
            if actual_min_solar == 0:
                solar_min_accuracy = 100 if llm_claims['min_solar'] == 0 else 0
            else:
                solar_min_error = abs(llm_claims['min_solar'] - actual_min_solar)
                solar_min_accuracy = max(0, 100 - (solar_min_error / actual_min_solar * 100))
            print(f"     Accuracy:  {solar_min_accuracy:.1f}%")
            accuracy_results['solar_min'] = solar_min_accuracy
            
            # Check peak solar time
            if 'hour' in df.columns:
                hourly_avg_solar = df.groupby('hour')['solar'].mean()
                actual_peak_solar_hour = hourly_avg_solar.idxmax()
                print(f"\n   Peak Solar Time:")
                print(f"     LLM Claim: 15:00 (3 PM)")
                print(f"     Actual:    {actual_peak_solar_hour:02d}:00")
                peak_solar_accuracy = 100 if actual_peak_solar_hour == 15 else max(0, 100 - abs(actual_peak_solar_hour - 15) * 10)
                print(f"     Accuracy:  {peak_solar_accuracy:.1f}%")
                accuracy_results['peak_solar_time'] = peak_solar_accuracy
        
        print(f"\n" + "=" * 50)
        
        # Check Wind Statistics
        if 'wind_speed' in df.columns:
            actual_avg_wind = df['wind_speed'].mean()
            actual_max_wind = df['wind_speed'].max()
            actual_min_wind = df['wind_speed'].min()
            
            print(f"💨 WIND SPEED ANALYSIS:")
            print(f"   Average Wind:")
            print(f"     LLM Claim: {llm_claims['avg_wind']} m/s")
            print(f"     Actual:    {actual_avg_wind:.1f} m/s")
            wind_avg_error = abs(llm_claims['avg_wind'] - actual_avg_wind)
            wind_avg_accuracy = max(0, 100 - (wind_avg_error / actual_avg_wind * 100))
            print(f"     Error:     {wind_avg_error:.1f} m/s ({100-wind_avg_accuracy:.1f}% error)")
            print(f"     Accuracy:  {wind_avg_accuracy:.1f}%")
            accuracy_results['wind_avg'] = wind_avg_accuracy
            
            print(f"\n   Maximum Wind:")
            print(f"     LLM Claim: {llm_claims['max_wind']} m/s")
            print(f"     Actual:    {actual_max_wind:.1f} m/s")
            wind_max_error = abs(llm_claims['max_wind'] - actual_max_wind)
            wind_max_accuracy = max(0, 100 - (wind_max_error / actual_max_wind * 100))
            print(f"     Error:     {wind_max_error:.1f} m/s")
            print(f"     Accuracy:  {wind_max_accuracy:.1f}%")
            accuracy_results['wind_max'] = wind_max_accuracy
            
            print(f"\n   Minimum Wind:")
            print(f"     LLM Claim: {llm_claims['min_wind']} m/s")
            print(f"     Actual:    {actual_min_wind:.1f} m/s")
            wind_min_error = abs(llm_claims['min_wind'] - actual_min_wind)
            wind_min_accuracy = max(0, 100 - (wind_min_error / actual_min_wind * 100))
            print(f"     Error:     {wind_min_error:.1f} m/s")
            print(f"     Accuracy:  {wind_min_accuracy:.1f}%")
            accuracy_results['wind_min'] = wind_min_accuracy
            
            # Check peak wind time
            if 'hour' in df.columns:
                hourly_avg_wind = df.groupby('hour')['wind_speed'].mean()
                actual_peak_wind_hour = hourly_avg_wind.idxmax()
                print(f"\n   Peak Wind Time:")
                print(f"     LLM Claim: 17:00 (5 PM)")
                print(f"     Actual:    {actual_peak_wind_hour:02d}:00")
                peak_wind_accuracy = 100 if actual_peak_wind_hour == 17 else max(0, 100 - abs(actual_peak_wind_hour - 17) * 10)
                print(f"     Accuracy:  {peak_wind_accuracy:.1f}%")
                accuracy_results['peak_wind_time'] = peak_wind_accuracy
        
        # Overall Accuracy Assessment
        print(f"\n" + "=" * 50)
        print(f"📈 OVERALL ACCURACY ASSESSMENT:")
        print(f"=" * 50)
        
        if accuracy_results:
            overall_accuracy = np.mean(list(accuracy_results.values()))
            print(f"Overall LLM Accuracy: {overall_accuracy:.1f}%")
            
            print(f"\nDetailed Breakdown:")
            for metric, accuracy in accuracy_results.items():
                status = "✅" if accuracy >= 90 else "⚠️" if accuracy >= 70 else "❌"
                print(f"  {status} {metric.replace('_', ' ').title()}: {accuracy:.1f}%")
            
            # Accuracy Grade
            if overall_accuracy >= 95:
                grade = "A+ (Excellent)"
            elif overall_accuracy >= 90:
                grade = "A (Very Good)"
            elif overall_accuracy >= 80:
                grade = "B (Good)"
            elif overall_accuracy >= 70:
                grade = "C (Fair)"
            else:
                grade = "D (Poor)"
            
            print(f"\n🎯 LLM Analysis Grade: {grade}")
            
            # Show actual data summary for comparison
            print(f"\n" + "=" * 50)
            print(f"📋 ACTUAL DATA SUMMARY:")
            print(f"=" * 50)
            print(df.describe())
            
        else:
            print("❌ Could not verify accuracy - missing required columns in data")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("LLM Analysis Accuracy Checker")
    print("=" * 40)
    check_analysis_accuracy()