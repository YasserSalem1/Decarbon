from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import requests

OPENWEATHER_API_KEY = "fe38edd8b8cd882230bcd457bd5758a3"
DEFAULT_CITY = "Berlin"

@dataclass
class Machine:
    name: str
    demand_kw: int
    is_on: bool = True
    image_path: str = ""

@dataclass
class EnergyInput:
    sun_percent: int = 100    # % of 100 kW max
    wind_percent: int = 50   # % of 100 kW max
    fossil_kw: int = 100     # actual kW out of 200 max

    def update_from_weather(self, city: str = DEFAULT_CITY):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                clouds = data['clouds']['all']
                wind_speed = data['wind']['speed']

                self.sun_percent = max(0, 100 - clouds)  # More clouds → less sun
                self.wind_percent = min(100, int(wind_speed * 10))  # wind in m/s scaled
            else:
                print("❌ Failed to fetch weather data")
        except Exception as e:
            print("❌ Exception while fetching weather data:", e)

    @property
    def sun_kw(self) -> int:
        return int(self.sun_percent * 1.0)  # Max 100 kW

    @property
    def wind_kw(self) -> int:
        return int(self.wind_percent * 1.0)  # Max 100 kW

    @property
    def renewable_kw(self) -> int:
        return self.sun_kw + self.wind_kw  # Max 200 kW

    @property
    def total_supply(self) -> int:
        return self.renewable_kw + self.fossil_kw  # Max 400 kW

@dataclass
class SimulationData:
    energy: EnergyInput = field(default_factory=EnergyInput)
    machines: Dict[str, Machine] = field(default_factory=lambda: {
        "Compressor A": Machine("Compressor A", 70, True, "images/machine1.png"),
        "Furnace B": Machine("Furnace B", 110, True, "images/machine2.png"),
        "Pump C": Machine("Pump C", 60, True, "images/machine3.png"),
        "Cutter D": Machine("Cutter D", 50, True, "images/machine1.png"),
        "Lathe E": Machine("Lathe E", 40, True, "images/machine2.png"),
        "Welder F": Machine("Welder F", 45, True, "images/machine3.png"),
    })

    def compute_energy_allocation(self) -> Tuple[Dict[str, Dict], int]:
        allocation = {}
        used = 0
        for machine in self.machines.values():
            demand = machine.demand_kw if machine.is_on else 0
            supply = min(demand, max(0, self.energy.total_supply - used))
            used += supply
            allocation[machine.name] = {
                "demand": demand,
                "supplied": supply,
                "status": "🟢 On" if machine.is_on else "🔴 Off"
            }
        return allocation, used

    def optimize_machine_schedule(
        self, time_slots: int = 8, min_runtime: int = 2
    ) -> Tuple[List[Set[str]], List[Dict[str, int]]]:
        schedule = [set() for _ in range(time_slots)]
        usage_per_slot = []

        available_machines = [m for m in self.machines.values() if m.is_on]
        run_counts = {m.name: 0 for m in available_machines}

        base_renewable = self.energy.renewable_kw
        base_fossil = self.energy.fossil_kw
        base_cycle_capacity = base_renewable + base_fossil

        total_energy_available = base_cycle_capacity  # Start with one full cycle

        # Step 1: Assign minimum runtime for each machine, spread across slots
        step = max(1, time_slots // max(1, min_runtime))
        for idx, m in enumerate(available_machines):
            assigned = 0
            for slot in range(idx, time_slots, step):
                if assigned >= min_runtime:
                    break
                # Only assign if machine fits in slot
                slot_demand = sum(self.machines[name].demand_kw for name in schedule[slot])
                if slot_demand + m.demand_kw <= base_cycle_capacity:
                    schedule[slot].add(m.name)
                    run_counts[m.name] += 1
                    assigned += 1

        # Step 2: Fill remaining capacity in each slot
        for hour in range(time_slots):
            factor = min(1.0, 0.6 + 0.05 * (hour + 1))  # 60% → 100% scaling
            renewable_cycle = int(base_renewable * factor)
            fossil_cycle = int(base_fossil * factor)
            cycle_capacity = renewable_cycle + fossil_cycle

            current_set = schedule[hour]
            current_demand = sum(self.machines[name].demand_kw for name in current_set)
            remaining_capacity = cycle_capacity - current_demand

            # Try to schedule more machines without exceeding current cycle capacity
            candidates = [m for m in available_machines if m.name not in current_set]
            candidates.sort(key=lambda m: run_counts[m.name])  # prioritize machines with fewer runs

            for m in candidates:
                if m.demand_kw <= remaining_capacity:
                    schedule[hour].add(m.name)
                    run_counts[m.name] += 1
                    remaining_capacity -= m.demand_kw
                    current_demand += m.demand_kw

            # ⚡ ENERGY ALLOCATION LOGIC
            remaining_demand = current_demand

            renewable_used = min(renewable_cycle, remaining_demand)
            remaining_demand -= renewable_used

            fossil_used = min(fossil_cycle, remaining_demand)
            remaining_demand -= fossil_used

            total_used_this_hour = renewable_used + fossil_used

            # 💡 Check if we exceed the global energy pool
            if total_used_this_hour > total_energy_available:
                print(f"⚠️ Hour {hour + 1}: Demand exceeds available energy "
                      f"({total_used_this_hour} kW > {total_energy_available} kW)")

            total_energy_available -= total_used_this_hour
            if total_energy_available < 0:
                total_energy_available = 0  # avoid negative energy

            # 📝 Store usage report BEFORE adding new cycle
            usage_per_slot.append({
                "hour": hour + 1,
                "total_demand": current_demand,
                "renewable_used": renewable_used,
                "fossil_used": fossil_used,
                "capacity": cycle_capacity,
                "total_energy_remaining": total_energy_available  # <-- This is now correct!
            })

            # 🔄 Add one new cycle for the next hour
            total_energy_available += cycle_capacity

        # Step 3: Final check for minimum runtime guarantee
        for m in available_machines:
            if run_counts[m.name] < min_runtime:
                print(f"⚠️ {m.name} scheduled only {run_counts[m.name]} times (< {min_runtime})")

        return schedule, usage_per_slot
