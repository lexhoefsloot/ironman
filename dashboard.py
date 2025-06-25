import streamlit as st
import gpxpy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# --- Physical Constants ---
AIR_DENSITY = 1.225  # kg/m^3
GRAVITY = 9.81  # m/s^2

@st.cache_data
def load_gpx_data(gpx_file):
    """Load and parse the GPX file."""
    gpx = gpxpy.parse(gpx_file)
    return gpx.tracks[0].segments[0].points

def solve_for_velocity(power_out, gradient_rad, cda, mass, powertrain_loss, rolling_resistance_power):
    """Calculates velocity by solving the power equation."""
    effective_power = power_out * (1 - powertrain_loss)
    a = 0.5 * cda * AIR_DENSITY
    b = 0
    c = mass * GRAVITY * np.sin(gradient_rad)
    d = rolling_resistance_power - effective_power

    coeffs = [a, b, c, d]
    roots = np.roots(coeffs)
    
    real_positive_roots = [r.real for r in roots if r.imag == 0 and r.real > 0]
    
    velocity = 20.0  # Default to 72km/h if no solution
    if real_positive_roots:
        velocity = min(real_positive_roots)
        
    return velocity

def run_simulation(points, power_uphill, power_downhill, power_flat, uphill_gradient, downhill_gradient, cda, mass, powertrain_loss, rolling_resistance_power, max_speed_kmh):
    """Runs the full bike leg simulation."""
    distances, elevations, velocities, times, powers = [], [], [], [], []
    total_distance = 0
    total_elevation_gain = 0
    
    for i in range(1, len(points)):
        p1, p2 = points[i-1], points[i]
        segment_distance = p1.distance_3d(p2)
        if segment_distance is None or segment_distance == 0:
            continue

        total_distance += segment_distance
        elevation_change = p2.elevation - p1.elevation
        if elevation_change > 0:
            total_elevation_gain += elevation_change

        gradient_percent = (elevation_change / segment_distance) * 100
        
        if gradient_percent > uphill_gradient:
            power = power_uphill
        elif gradient_percent < downhill_gradient:
            power = power_downhill
        else:
            power = power_flat

        gradient_rad = np.arctan(gradient_percent / 100)
        velocity = solve_for_velocity(power, gradient_rad, cda, mass, powertrain_loss, rolling_resistance_power)
        
        max_speed_ms = max_speed_kmh / 3.6
        velocity = min(velocity, max_speed_ms)

        segment_time = segment_distance / velocity if velocity > 0 else 0

        distances.append(total_distance / 1000)
        elevations.append(p2.elevation)
        velocities.append(velocity * 3.6)
        times.append(segment_time)
        powers.append(power)
        
    return distances, elevations, velocities, times, powers, total_distance, total_elevation_gain

def main():
    st.set_page_config(layout="wide")
    st.title("🚴‍♂️ Interactive Ironman Bike Leg Simulator")

    st.sidebar.header("Simulation Parameters")
    uploaded_file = st.sidebar.file_uploader("Upload your GPX file", type="gpx")

    if uploaded_file is not None:
        points = load_gpx_data(uploaded_file)
        
        st.sidebar.subheader("Power Strategy (Watts)")
        power_uphill = st.sidebar.slider("Uphill Power", 150, 400, 230)
        power_flat = st.sidebar.slider("Flat Power", 150, 400, 200)
        power_downhill = st.sidebar.slider("Downhill Power", 100, 300, 190)

        st.sidebar.subheader("Gradient Thresholds (%)")
        uphill_gradient = st.sidebar.slider("Uphill Starts At >", 0.0, 5.0, 2.0, 0.5)
        downhill_gradient = st.sidebar.slider("Downhill Starts At <", -5.0, 0.0, -2.0, 0.5)

        st.sidebar.subheader("Rider & Bike Specs")
        mass = st.sidebar.slider("Combined Mass (kg)", 60, 100, 76)
        cda = st.sidebar.slider("CdA (m^2)", 0.15, 0.40, 0.25, 0.01)
        rolling_resistance_power = st.sidebar.slider("Rolling Resistance (Watts)", 5, 30, 15)
        powertrain_loss = st.sidebar.slider("Powertrain Loss (%)", 1, 10, 5) / 100.0
        
        st.sidebar.subheader("Constraints")
        max_speed_kmh = st.sidebar.slider("Max Speed (km/h)", 40, 80, 55)

        # --- Ironman Parameters ---
        st.sidebar.header("Ironman Swim, Run & Transitions")
        swim_length_km = st.sidebar.number_input("Swim Length (km)", min_value=1.0, max_value=5.0, value=3.8, step=0.1)
        # Swim pace as min:sec per 100m
        st.sidebar.markdown("**Swim Pace (per 100m)**")
        swim_pace_min = st.sidebar.number_input("Minutes", min_value=0, max_value=20, value=1)
        swim_pace_sec = st.sidebar.number_input("Seconds", min_value=0, max_value=59, value=40)
        swim_pace_total_sec = swim_pace_min * 60 + swim_pace_sec
        # Convert pace (sec/100m) to speed (km/h): speed_kmh = 0.1 km / (pace_sec/3600)
        swim_speed_kmh = 0.1 / (swim_pace_total_sec / 3600) if swim_pace_total_sec > 0 else 1.0
        t1_time_min = st.sidebar.number_input("T1 (Swim-Bike) Transition (min)", min_value=1, max_value=30, value=8)
        run_length_km = st.sidebar.number_input("Run Length (km)", min_value=10.0, max_value=60.0, value=42.2, step=0.1)
        run_speed_kmh = st.sidebar.number_input("Run Speed (km/h)", min_value=5.0, max_value=20.0, value=10.0, step=0.1)
        t2_time_min = st.sidebar.number_input("T2 (Bike-Run) Transition (min)", min_value=1, max_value=30, value=5)

        # Run Simulation
        distances, elevations, velocities, times, powers, total_distance, total_elevation_gain = run_simulation(
            points, power_uphill, power_downhill, power_flat, uphill_gradient, downhill_gradient,
            cda, mass, powertrain_loss, rolling_resistance_power, max_speed_kmh
        )

        total_time_seconds = sum(times)
        total_time_hours = total_time_seconds / 3600
        hours = int(total_time_hours)
        minutes = int((total_time_hours * 60) % 60)
        
        # --- Display Results ---
        st.header("Simulation Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Finish Time", f"{hours}h {minutes}m")
        col2.metric("Avg Speed", f"{(total_distance / 1000) / total_time_hours:.2f} km/h")
        col3.metric("Avg Power", f"{np.average(powers, weights=times):.0f} W")
        col4.metric("Elevation Gain", f"{total_elevation_gain:.0f} m")

        # --- Ironman Total Time Estimation ---
        swim_time_hr = swim_length_km / swim_speed_kmh
        run_time_hr = run_length_km / run_speed_kmh
        t1_hr = t1_time_min / 60
        t2_hr = t2_time_min / 60
        total_ironman_time_hr = swim_time_hr + total_time_hours + run_time_hr + t1_hr + t2_hr
        total_ironman_hours = int(total_ironman_time_hr)
        total_ironman_minutes = int((total_ironman_time_hr * 60) % 60)
        total_ironman_seconds = int((total_ironman_time_hr * 3600) % 60)

        st.subheader("Ironman Total Time Estimation")
        st.markdown(f"""
        | Segment   | Distance | Pace/Speed      | Time         |
        |-----------|----------|----------------|--------------|
        | Swim      | {swim_length_km:.1f} km | {swim_pace_min}:{swim_pace_sec:02d} /100m | {int(swim_time_hr)}h {int((swim_time_hr*60)%60)}m |
        | T1        | -        | -              | {t1_time_min} min         |
        | Bike      | {total_distance/1000:.1f} km | {(total_distance/1000)/total_time_hours:.2f} km/h | {hours}h {minutes}m |
        | T2        | -        | -              | {t2_time_min} min         |
        | Run       | {run_length_km:.1f} km | {run_speed_kmh:.1f} km/h | {int(run_time_hr)}h {int((run_time_hr*60)%60)}m |
        """)
        st.success(f"Estimated Total Ironman Time: {total_ironman_hours}h {total_ironman_minutes}m {total_ironman_seconds}s")

        # --- Visualizations ---
        st.header("Performance Analysis")
        
        # Elevation and Speed Profile
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Elevation Profile', 'Simulated Speed Profile'))
        fig.add_trace(go.Scatter(x=distances, y=elevations, mode='lines', name='Elevation', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=distances, y=velocities, mode='lines', name='Speed', line=dict(color='green')), row=2, col=1)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Time Distribution
        time_uphill = sum(t for t, p in zip(times, powers) if p == power_uphill)
        time_downhill = sum(t for t, p in zip(times, powers) if p == power_downhill)
        time_flat = sum(t for t, p in zip(times, powers) if p == power_flat)
        pie_labels = ['Uphill', 'Downhill', 'Flat']
        pie_values = [time_uphill, time_downhill, time_flat]
        pie_fig = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, hole=.3, title='Time in Each Zone')])
        st.plotly_chart(pie_fig, use_container_width=True)

    else:
        st.info("Please upload a GPX file to begin the simulation.")
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/e2e/scripts/test_data/bike.jpg")

if __name__ == "__main__":
    main() 