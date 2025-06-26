# import random
# import requests
# import pandas as pd

# def get_grid_load(factory_count=10):
#     url = "https://api.openei.org/utility_rates"
#     params = {
#         "version": 3,
#         "format": "json",
#         "api_key": "TZfYbQfvkLkgfc0hMb95ltWzIcsPOZBmt295Psbc",  # Free tier
#         "limit": factory_count
#     }
#     response = requests.get(url, params=params)
#     data = response.json()
    
#     # Simulate factory loads (1-5 MW each)
#     factories = []
#     for i, utility in enumerate(data["items"][:factory_count]):
#         factories.append({
#             "factory_id": f"machine_{i}",
#             "grid_load_mw": round(random.uniform(1, 5), 2),  # Random load
#             "utility": utility["utility"],
#             "timestamp": pd.Timestamp.now()
#         })
    
#     return pd.DataFrame(factories)

# df = get_grid_load(10)
# df.to_csv("factory_grid_loads.csv", index=False)
import requests
import pandas as pd

def get_eia_industrial_data(api_key):
    """Fetch U.S. industrial electricity data from EIA API"""
    url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": "4XVxrVQ7H3ddVOvZKhYuggmOf64DCT9JvuKlZ0Sl",
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": ["PJM", "MISO"],  # Grid operators
        "facets[type][]": ["D"],  # Demand data
        "start": "2024-01",
        "end": "2024-06",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()['response']['data']
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error: {str(e)}")
        return pd.DataFrame()

# Usage (get free API key at https://www.eia.gov/opendata/)
df = get_eia_industrial_data("YOUR_API_KEY")
if not df.empty:
    df.to_csv("industrial_electricity_demand.csv", index=False)