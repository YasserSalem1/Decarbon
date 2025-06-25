import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from backend import SimulationData

# === Page Config ===
st.set_page_config(layout="wide", page_title="NetZero Optimizer")

# === Load Images for header ===
solar_img = Image.open("images/ren.png")
fossil_img = Image.open("images/fossil.png")

# === Initialize simulation ===
sim = SimulationData()

# === Sidebar Controls ===
st.sidebar.title("⚙️ Simulation Controls")

city = st.sidebar.text_input("City for Weather", "Berlin")
if st.sidebar.button("🌦 Fetch Real Weather"):
    sim.energy.update_from_weather(city)

sim.energy.sun_percent = st.sidebar.slider("☀️ Sun Availability (%)", 0, 100, sim.energy.sun_percent)
sim.energy.wind_percent = st.sidebar.slider("🌬 Wind Availability (%)", 0, 100, sim.energy.wind_percent)
sim.energy.fossil_kw = st.sidebar.slider("⛽ Fossil Power (kW)", 0, 200, sim.energy.fossil_kw)  # reduce fossil for challenge

for name, machine in sim.machines.items():
    machine.is_on = st.sidebar.toggle(f"⚙️ {name}", machine.is_on)

# === Page Title ===
st.title("🔋 NetZero Optimizer – Real-Time Load Simulation")
st.markdown("---")

# === Energy Overview ===
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image(solar_img, width=100)
    st.metric("Renewables", f"{sim.energy.renewable_kw} kW")
    st.markdown(f"☀️ Sun: {sim.energy.sun_percent}%  \n🌬 Wind: {sim.energy.wind_percent}%")

with col3:
    st.image(fossil_img, width=100)
    st.metric("Fossil", f"{sim.energy.fossil_kw} kW")

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sim.energy.total_supply,
        title={'text': "Total Available Energy"},
        gauge={
            'axis': {'range': [0, 400]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, sim.energy.renewable_kw], 'color': "lightgreen"},
                {'range': [sim.energy.renewable_kw, sim.energy.total_supply], 'color': "lightblue"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# === Energy Allocation Display with 6 machines in 2 rows of 3 columns ===
st.markdown("### ⚡ Energy Flow")

allocations, used_energy = sim.compute_energy_allocation()

machine_names = list(allocations.keys())
batch_size = 3
for i in range(0, len(machine_names), batch_size):
    cols = st.columns(batch_size)
    batch = machine_names[i:i + batch_size]
    for j, name in enumerate(batch):
        data = allocations[name]
        with cols[j]:
            st.image(sim.machines[name].image_path, width=100)
            st.markdown(f"### {name}")
            progress_val = int((data['supplied'] / (data['demand'] or 1)) * 100)
            st.progress(progress_val)
            st.markdown(f"Demand: `{data['demand']} kW`")
            st.markdown(f"Supplied: `{data['supplied']} kW`")
            st.markdown(f"Status: {data['status']}")

st.markdown("---")
st.markdown(f"🔄 **Total Used Energy**: `{used_energy} kW`  |  ⚖️ **Remaining Energy**: `{sim.energy.total_supply - used_energy} kW`")

# === Optimization Section ===
st.markdown("### 📅 Optimize Machine Usage Over the Day")
if st.button("🔄 Optimize Machine Usage"):
    schedule, usage = sim.optimize_machine_schedule(time_slots=8, min_runtime=2)

    st.write("#### Machine Timetable (✅ = ON, ❌ = OFF)")
    timetable = {name: [] for name in sim.machines}
    for slot in schedule:
        for name in sim.machines:
            timetable[name].append("✅" if name in slot else "❌")
    st.dataframe(timetable)

    st.markdown("---")
    st.write("#### Hourly Energy Usage Breakdown")
    for i, usage_slot in enumerate(usage, 1):
        st.write(
            f"""
            🔢 **Hour {usage_slot['hour']}**
            - 🔋 **Total Demand**: `{usage_slot['total_demand']} kW`
            - ☀️ **Renewable Used**: `{usage_slot['renewable_used']} kW`
            - ⛽ **Fossil Used**: `{usage_slot['fossil_used']} kW`
            - ⚡ **Total Available Energy**: `{usage_slot['capacity']} kW`
            """
        )


    total_energy = sum(slot['total_demand'] for slot in usage)
    renewable_used = sum(slot['renewable_used'] for slot in usage)
    fossil_used = sum(slot['fossil_used'] for slot in usage)

    st.success(f"🔋 Total Energy Used Over Day: `{total_energy} kW` → ☀️ Renewable: `{renewable_used} kW`, ⛽ Fossil: `{fossil_used} kW`")
