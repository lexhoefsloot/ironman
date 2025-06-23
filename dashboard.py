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