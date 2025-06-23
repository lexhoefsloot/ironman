import gpxpy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation Parameters ---
CDA = 0.25  # Aerodynamic drag coefficient (m^2)
MASS = 76  # Combined mass of rider and bike (kg)
ROLLING_RESISTANCE_POWER = 15  # Watts
POWERTRAIN_LOSS = 0.05  # 5%

POWER_UPHILL = 230  # Watts
POWER_DOWNHILL = 190  # Watts
POWER_FLAT = 200  # Watts

# Gradient thresholds
UPHILL_GRADIENT = 2.0  # %
DOWNHILL_GRADIENT = -2.0  # %

MAX_SPEED_KMH = 55.0 # km/h

# --- Physical Constants ---
AIR_DENSITY = 1.225  # kg/m^3
GRAVITY = 9.81  # m/s^2

def get_power(gradient):
    """Determines power output based on gradient."""
    if gradient > UPHILL_GRADIENT:
        return POWER_UPHILL
    elif gradient < DOWNHILL_GRADIENT:
        return POWER_DOWNHILL
    else:
        return POWER_FLAT

def solve_for_velocity(power_out, gradient_rad):
    """
    Calculates velocity by solving the power equation.
    P_effective = (F_aero + F_gravity) * v + P_rolling
    P_effective - P_rolling = (0.5 * CdA * rho * v^2 + m * g * sin(theta)) * v
    This forms a cubic equation for v: Ax^3 + Bx^2 + Cx + D = 0
    """
    effective_power = power_out * (1 - POWERTRAIN_LOSS)
    
    # Coefficients for the cubic equation: a*v^3 + b*v^2 + c*v + d = 0
    a = 0.5 * CDA * AIR_DENSITY
    b = 0
    c = MASS * GRAVITY * np.sin(gradient_rad)
    d = ROLLING_RESISTANCE_POWER - effective_power

    coeffs = [a, b, c, d]
    roots = np.roots(coeffs)
    
    # Find the real, positive root for velocity
    real_positive_roots = [r.real for r in roots if r.imag == 0 and r.real > 0]
    
    velocity = 20.0  # Default to 72km/h if no solution, will be capped
    if real_positive_roots:
        velocity = min(real_positive_roots)
        
    # Cap the speed
    max_speed_ms = MAX_SPEED_KMH / 3.6
    return min(velocity, max_speed_ms)


def main():
    """Main simulation function."""
    gpx_file_path = 'IRONMAN Switzerland Thun - Bike Course 2.gpx'
    
    try:
        with open(gpx_file_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
    except FileNotFoundError:
        print(f"Error: The file '{gpx_file_path}' was not found.")
        print("Please make sure the GPX file is in the same directory as the script.")
        return

    points = gpx.tracks[0].segments[0].points

    distances = []
    elevations = []
    gradients = []
    velocities = []
    times = []
    powers = []
    
    total_distance = 0
    total_elevation_gain = 0

    for i in range(1, len(points)):
        p1 = points[i-1]
        p2 = points[i]

        segment_distance = p1.distance_3d(p2)
        if segment_distance is None or segment_distance == 0:
            continue

        total_distance += segment_distance
        
        elevation_change = p2.elevation - p1.elevation
        if elevation_change > 0:
            total_elevation_gain += elevation_change

        gradient_percent = (elevation_change / segment_distance) * 100 if segment_distance > 0 else 0
        gradient_rad = np.arctan(gradient_percent / 100)

        power = get_power(gradient_percent)
        velocity = solve_for_velocity(power, gradient_rad)

        segment_time = segment_distance / velocity if velocity > 0 else 0

        distances.append(total_distance / 1000) # km
        elevations.append(p2.elevation)
        gradients.append(gradient_percent)
        velocities.append(velocity * 3.6) # km/h
        times.append(segment_time)
        powers.append(power)

    total_time_seconds = sum(times)
    total_time_hours = total_time_seconds / 3600
    
    hours = int(total_time_hours)
    minutes = int((total_time_hours * 60) % 60)
    seconds = int((total_time_hours * 3600) % 60)

    # More metrics
    max_speed_kmh = max(velocities)
    weighted_avg_power = np.average(powers, weights=times)

    print(f"--- Simulation Results ---")
    print(f"Total distance: {total_distance / 1000:.2f} km")
    print(f"Total elevation gain: {total_elevation_gain:.2f} m")
    print(f"Estimated bike time: {hours} hours, {minutes} minutes, {seconds} seconds")
    print(f"Average speed: {(total_distance / 1000) / total_time_hours:.2f} km/h")
    print(f"Maximum speed: {max_speed_kmh:.2f} km/h")
    print(f"Average power: {weighted_avg_power:.2f} W")

    # --- Interactive Visualization ---
    
    # Elevation and Speed Profile
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Elevation Profile', 'Simulated Speed Profile'))

    fig.add_trace(go.Scatter(x=distances, y=elevations, mode='lines', name='Elevation', line=dict(color='blue')), row=1, col=1)
    fig.update_yaxes(title_text="Elevation (m)", row=1, col=1)

    fig.add_trace(go.Scatter(x=distances, y=velocities, mode='lines', name='Speed', line=dict(color='green')), row=2, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=2, col=1)
    
    fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
    fig.update_layout(title_text='Ironman Bike Leg Simulation', height=800)
    fig.write_html('interactive_simulation_results.html')


    # Time spent in each zone
    time_uphill = sum(t for t, p in zip(times, powers) if p == POWER_UPHILL)
    time_downhill = sum(t for t, p in zip(times, powers) if p == POWER_DOWNHILL)
    time_flat = sum(t for t, p in zip(times, powers) if p == POWER_FLAT)

    labels = ['Uphill', 'Downhill', 'Flat']
    values = [time_uphill, time_downhill, time_flat]

    pie_fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    pie_fig.update_layout(title_text='Proportion of Time Spent in Each Zone')
    pie_fig.write_html('interactive_time_distribution.html')

    print("\nInteractive visualizations have been saved to:")
    print("- interactive_simulation_results.html")
    print("- interactive_time_distribution.html")


if __name__ == "__main__":
    main() 