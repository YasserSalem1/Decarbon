import requests

# Your API key
API_KEY = "fe38edd8b8cd882230bcd457bd5758a3"

# City to test
city = "Berlin"

# API endpoint
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"

# Send request
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    clouds = data['clouds']['all']  # Cloudiness %
    wind_speed = data['wind']['speed']  # Wind speed in m/s

    # Convert to 0–100 index
    sun_index = max(0, 100 - clouds)
    wind_index = min(100, int(wind_speed * 10))  # 10 m/s ≈ 100%

    print(f"☀️  Sun index: {sun_index}/100")
    print(f"🌬️  Wind index: {wind_index}/100")
else:
    print("❌ Failed to fetch weather data")
    print("Status Code:", response.status_code)
    print("Message:", response.text)
