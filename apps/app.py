import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# === Page Config ===
st.set_page_config(layout="wide", page_title="NetZero Optimizer")

# === Load Images ===
solar_img = Image.open("images/ren.png")
fossil_img = Image.open("images/fossil.png")
machine_imgs = {
    "Machine 1": Image.open("images/machine1.png"),
    "Machine 2": Image.open("images/machine2.png"),
    "Machine 3": Image.open("images/machine3.png"),
}

# === Sidebar Controls ===
st.sidebar.title("⚙️ Simulation Controls")

# New: Separate sliders
sun_percent = st.sidebar.slider("☀️ Sun Availability (%)", 0, 100, 50)
wind_percent = st.sidebar.slider("🌬 Wind Availability (%)", 0, 100, 50)
fossil_kw = st.sidebar.slider("⛽ Fossil Power (kW)", 0, 200, 150)

machine_toggle = {
    "Machine 1": st.sidebar.toggle("⚙️ Machine 1", True),
    "Machine 2": st.sidebar.toggle("⚙️ Machine 2", True),
    "Machine 3": st.sidebar.toggle("⚙️ Machine 3", True),
}

# === Energy Source Calculations ===
# Max renewable is capped at 100 kW
renewable_kw = sun_percent + wind_percent

total_supply = renewable_kw + fossil_kw

# === Machine Power Demand ===
machine_demand = {
    "Machine 1": 30,
    "Machine 2": 40,
    "Machine 3": 50,
}

# === Page Title ===
st.title("🔋 NetZero Optimizer – Real-Time Load Simulation")
st.markdown("---")

# === Top: Energy Source Display ===
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image(solar_img, width=100)
    st.metric(label="Renewables (Wind + Sun)", value=f"{renewable_kw} kW")
    st.markdown(f"☀️ Sun: {sun_percent}%  \n🌬 Wind: {wind_percent}%")

with col3:
    st.image(fossil_img, width=100)
    st.metric(label="Fossil (kW)", value=f"{fossil_kw} kW")

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_supply,
        title={'text': "Total Available Energy"},
        gauge={
            'axis': {'range': [0, 300]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, renewable_kw], 'color': "lightgreen"},
                {'range': [renewable_kw, total_supply], 'color': "lightblue"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# === Energy Flow Arrow ===
st.markdown("### ⚡ Energy Flow")

# === Machine Display ===
machine_cols = st.columns(3)
used_energy = 0

for i, (machine, is_on) in enumerate(machine_toggle.items()):
    demand = machine_demand[machine] if is_on else 0
    supplied = min(demand, max(0, total_supply - used_energy))
    used_energy += supplied

    with machine_cols[i]:
        st.image(machine_imgs[machine], width=100)
        st.markdown(f"### {machine}")
        st.progress(int((supplied / (machine_demand[machine] or 1)) * 100))
        st.markdown(f"Demand: `{machine_demand[machine]} kW`")
        st.markdown(f"Supplied: `{supplied} kW`")
        st.markdown(f"Status: {'🟢 On' if is_on else '🔴 Off'}")

# === Summary Footer ===
st.markdown("---")
st.markdown(f"🔄 **Total Energy Used**: `{used_energy} kW`  |  ⚖️ **Remaining**: `{total_supply - used_energy} kW`")
st.markdown("### 🌍 Striving for NetZero Together!")