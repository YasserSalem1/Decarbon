from fastapi import FastAPI
from schemas import MachineState, EnergyResponse, MachineOutput
import requests
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MACHINE_DEMAND = {
    "Machine 1": 30,
    "Machine 2": 40,
    "Machine 3": 50,
}


def get_weather(city: str):
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        response = requests.get(url).json()
        clouds = response["clouds"]["all"]
        wind_speed = response["wind"]["speed"]

        solar = max(0, 100 - clouds)
        wind = min(100, wind_speed * 10)
        return int(solar), int(wind)
    except:
        return 70, 60  # fallback


@app.post("/compute", response_model=EnergyResponse)
def compute_energy(input_data: MachineState):
    solar_percent, _ = get_weather(input_data.city)
    solar_kw = int(100 * (solar_percent / 100))
    fossil_kw = input_data.fossil_kw
    total_supply = solar_kw + fossil_kw
    used_energy = 0

    machines = {}

    for machine, on in input_data.machine_states.items():
        demand = MACHINE_DEMAND[machine] if on else 0
        supplied = min(demand, max(0, total_supply - used_energy))
        used_energy += supplied
        machines[machine] = MachineOutput(demand=demand, supplied=supplied, status=on)

    return EnergyResponse(
        solar_kw=solar_kw,
        fossil_kw=fossil_kw,
        total_supply=total_supply,
        used_energy=used_energy,
        machines=machines
    )
