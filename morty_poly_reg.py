# morty_final_corrected_with_tuning.py
import os
import sys
import time
from api_client import SphinxAPIClient
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from data_collector import DataCollector
from visualizations import (
    plot_survival_rates,
    plot_survival_by_planet,
    plot_moving_average,
    plot_risk_evolution,
    plot_episode_summary,
)

# --- Configuration ---
INITIAL_EXPLORATION_TRIPS = 5  # Initial trips per planet for data collection
UCB_C = 2.0                    # UCB exploration constant
POLY_DEGREE = 3                # Degree of polynomial for prediction
BUFFER_SIZE = 100              # Size of the rolling buffer for regression
MIN_DATA_FOR_POLY = 5          # Minimum data points needed to fit polynomial
MAX_MORTIES_PER_TRIP = 3       # Maximum Morties to send in a single trip (API limit)

# --- Agent State ---
# Rolling buffer: list of (global_trip_number, outcome) for each planet
planet_buffer = {
    0: [],  # "On a Cob" Planet
    1: [],  # Cronenberg World
    2: []   # The Purge Planet
}

# Track the number of times each planet has been tried (for UCB)
planet_tries = {0: 0, 1: 0, 2: 0}
# Track the total number of trips taken (for UCB)
total_trips_taken = 0
# Store the last predicted survival rates from polynomial regression for each planet (for UCB calculation)
last_poly_predictions = {0: 0.5, 1: 0.5, 2: 0.5}


def update_buffer_and_predict(planet_idx, global_trip_num, outcome):
    """Add new point to rolling buffer and return updated prediction."""
    global planet_buffer

    # Add new data point (trip number, outcome) to the planet's buffer
    planet_buffer[planet_idx].append((global_trip_num, outcome))
    
    # Enforce the rolling window size by removing the oldest point if necessary
    if len(planet_buffer[planet_idx]) > BUFFER_SIZE:
        planet_buffer[planet_idx].pop(0)

    # Predict for the *next* trip using the updated buffer
    return predict_survival_rate_poly(planet_buffer[planet_idx], global_trip_num + 1)


def predict_survival_rate_poly(buffer, next_trip_num):
    """
    Fit a polynomial to the buffer data and predict the survival rate for the next trip.
    """
    if len(buffer) < MIN_DATA_FOR_POLY:
        # If not enough data, return a default prediction
        return 0.5

    # Extract trip numbers and outcomes from the buffer
    trip_nums = np.array([h[0] for h in buffer])
    outcomes = np.array([h[1] for h in buffer])

    try:
        # Fit the polynomial
        coeffs = np.polyfit(trip_nums, outcomes, POLY_DEGREE)
        poly = np.poly1d(coeffs)
        # Predict for the next trip number
        pred = poly(next_trip_num)
        # Clamp the prediction between 0 and 1
        return float(np.clip(pred, 0.0, 1.0))
    except Exception:
        # If fitting fails (e.g., due to numerical issues), fall back to the average of the buffer
        return np.mean(outcomes)


def choose_planet_with_ucb():
    """
    Choose a planet using Upper Confidence Bound (UCB) policy.
    Uses polynomial regression predictions as the average rate estimate.
    """
    global total_trips_taken, last_poly_predictions

    if total_trips_taken == 0:
        # If no trips taken yet, return a random planet to start exploration
        chosen_planet = random.choice([0, 1, 2])
        print(f"Main Agent: No data yet, choosing planet {chosen_planet} randomly.")
        # Initialize predictions with default
        for p in [0, 1, 2]:
            last_poly_predictions[p] = 0.5
        return chosen_planet

    ucb_values = {}
    for planet_idx in [0, 1, 2]:
        n_i = planet_tries[planet_idx]

        # Update the polynomial prediction for this planet based on its current rolling buffer
        if len(planet_buffer[planet_idx]) > 0:
            last_poly_predictions[planet_idx] = predict_survival_rate_poly(
                planet_buffer[planet_idx], total_trips_taken + 1  # Predict for the *next* trip
            )

        if n_i == 0:
            # If a planet hasn't been tried yet, give it the highest possible UCB to ensure it's tried
            ucb_values[planet_idx] = float('inf')
            print(f"  Planet {planet_idx}: Not tried yet, UCB = inf")
        else:
            # Use the polynomial prediction as the 'average' rate for UCB
            poly_rate = last_poly_predictions[planet_idx]
            # Calculate the UCB value
            ucb_value = poly_rate + UCB_C * np.sqrt(np.log(total_trips_taken) / n_i)
            ucb_values[planet_idx] = ucb_value
            print(f"  Planet {planet_idx}: Poly Pred = {poly_rate:.4f}, Tries = {n_i}, UCB = {ucb_value:.4f}")

    # Choose the planet with the highest UCB value
    chosen_planet = max(ucb_values, key=ucb_values.get)
    print(f"Main Agent: Chose planet {chosen_planet} (UCB: {ucb_values[chosen_planet]:.4f})")
    return chosen_planet


def decide_trip_size(predicted_rate, remaining_in_citadel, planet_idx):
    """
    Decide trip size based on predicted rate and confidence from the polynomial fit (R-squared).
    """
    max_possible_size = min(MAX_MORTIES_PER_TRIP, remaining_in_citadel)
    return max_possible_size
    """
    if len(planet_buffer[planet_idx]) < MIN_DATA_FOR_POLY:
        # If not enough data for a good fit, send a small probe
        return min(1, max_possible_size)

    # Get the buffer data for the chosen planet
    buffer_data = planet_buffer[planet_idx]
    trip_nums = np.array([h[0] for h in buffer_data])
    outcomes = np.array([h[1] for h in buffer_data])

    try:
        # Fit the polynomial
        coeffs = np.polyfit(trip_nums, outcomes, POLY_DEGREE)
        poly = np.poly1d(coeffs)

        # Calculate R-squared as a measure of fit confidence
        predicted_outcomes = poly(trip_nums)
        ss_res = np.sum((outcomes - predicted_outcomes) ** 2)
        ss_tot = np.sum((outcomes - np.mean(outcomes)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Define thresholds based on rate and confidence (R-squared)
        # High confidence AND high rate -> send more
        if predicted_rate >= 0.85 and r_squared > 0.7:
            return min(3, max_possible_size)
        elif predicted_rate >= 0.75 and r_squared > 0.5:
            return min(2, max_possible_size)
        # If confidence is low, be more conservative even if rate is high
        elif predicted_rate >= 0.85 and r_squared <= 0.7:
            return min(2, max_possible_size)
        # Otherwise, default logic based on rate
        elif predicted_rate >= 0.70:
            return min(2, max_possible_size)
        else:
            return min(1, max_possible_size)

    except Exception:
        # If fitting fails, fall back to a conservative approach
        if predicted_rate >= 0.70:
            return min(2, max_possible_size)
        else:
            return min(1, max_possible_size)
    """

def print_fixed_progress_bar(jessica, citadel, lost, total=1000):
    """
    Simulate a fixed bar at the bottom by clearing the previous line and reprinting.
    """
    safe_pct = jessica / total
    pending_pct = citadel / total
    lost_pct = lost / total

    bar_length = 50
    safe_len = int(bar_length * safe_pct)
    pending_len = int(bar_length * pending_pct)
    lost_len = bar_length - safe_len - pending_len

    bar = "\033[32m" + "█" * safe_len + \
          "\033[33m" + "█" * pending_len + \
          "\033[31m" + "█" * lost_len + \
          "\033[0m"

    # Move cursor up one line and clear it, then print the new bar
    print(f"\033[F\033[K[Progress] {bar} | Safe: {jessica} | Citadel: {citadel} | Lost: {lost} ", flush=True)


def collect_initial_data(collector):
    """
    Send a fixed number of trips to each planet to collect initial data.
    Uses trip_size of 1 for initial exploration.
    """
    global total_trips_taken # Need to update the global trip counter

    print("Collecting initial data from all planets (trip_size=1)...")
    for planet_idx in [0, 1, 2]:
        planet_name = collector.client.get_planet_name(planet_idx)
        print(f"\nExploring {planet_name} ({planet_idx})...")

        for _ in range(INITIAL_EXPLORATION_TRIPS):
            result = collector.client.send_morties(planet=planet_idx, morty_count=1)
            outcome = 1 if result['survived'] else 0
            global_trip = total_trips_taken + 1

            # Update the rolling buffer for this planet and get the *next* prediction
            update_buffer_and_predict(planet_idx, global_trip, outcome)

            # Update tries counter
            planet_tries[planet_idx] += 1
            total_trips_taken += 1

            # Add the trip data to the collector's internal list
            trip_data = {
                'trip_number': total_trips_taken, # Use the cumulative trip number
                'planet': planet_idx,
                'planet_name': planet_name,
                'morties_sent': result['morties_sent'],
                'survived': result['survived'],
                'steps_taken': result['steps_taken'],
                'morties_in_citadel': result['morties_in_citadel'],
                'morties_on_planet_jessica': result['morties_on_planet_jessica'],
                'morties_lost': result['morties_lost']
            }
            collector.trips_data.append(trip_data)

            # Print progress
            if (total_trips_taken) % 5 == 0: # Less frequent print for initial data
                print(f"  Initial data trip {total_trips_taken}/{INITIAL_EXPLORATION_TRIPS*3} completed.")


def run_single_episode(client, collector):
    """
    Run a single episode of the integrated agent.
    """
    global total_trips_taken, planet_buffer, planet_tries, last_poly_predictions # Access global state variables

    # Reset state for this episode
    planet_buffer = {0: [], 1: [], 2: []}
    planet_tries = {0: 0, 1: 0, 2: 0}
    total_trips_taken = 0
    last_poly_predictions = {0: 0.5, 1: 0.5, 2: 0.5}

    # Start a new episode
    start_info = client.start_episode()
    print(f"✅ Episode started. Initial Morties in Citadel: {start_info['morties_in_citadel']}")
    total_morties = 1000

    # Step 1: Collect initial data from all planets (optional, UCB handles initial exploration well)
    collect_initial_data(collector)

    # After initial data, print the FIRST bar (no previous line to clear)
    print("[Progress] " + " " * 80)  # Placeholder line for the bar
    print_fixed_progress_bar(0, total_morties, 0)

    # Step 2: Main decision-making loop
    while True:
        status = client.get_status()
        morties_left = status['morties_in_citadel']

        if morties_left <= 0:
            print("\nNo more morties left in the citadel. Mission complete!")
            break

        # --- Main Agent Decision (using UCB with polynomial estimate) ---
        # Choose the planet using UCB (based on polynomial regression predictions)
        chosen_planet = choose_planet_with_ucb()

        # Get the current prediction for the chosen planet
        pred_rate = last_poly_predictions[chosen_planet]

        # --- Decide Trip Size based on Prediction and Confidence ---
        # Pass planet_idx here to use planet-specific buffer for R-squared calculation
        trip_size = decide_trip_size(pred_rate, morties_left, chosen_planet)

        # Send the morties using the client
        print(f"\nPreparing to send {trip_size} morties to planet {chosen_planet} (index {chosen_planet})...", end='')
        result = client.send_morties(planet=chosen_planet, morty_count=trip_size)

        # Update the *rolling* history for the chosen planet (for polynomial regression estimation)
        outcome = 1 if result['survived'] else 0
        global_trip_num = total_trips_taken + 1
        # Use the function to update buffer and get next prediction
        update_buffer_and_predict(chosen_planet, global_trip_num, outcome)

        # Update the tries counter for UCB
        planet_tries[chosen_planet] += 1
        # Update the total trip counter for UCB
        total_trips_taken += 1

        # Add the trip data to the collector's internal list
        trip_data = {
            'trip_number': total_trips_taken, # Use the cumulative trip number
            'planet': chosen_planet,
            'planet_name': client.get_planet_name(chosen_planet),
            'morties_sent': result['morties_sent'],
            'survived': result['survived'],
            'steps_taken': result['steps_taken'],
            'morties_in_citadel': result['morties_in_citadel'],
            'morties_on_planet_jessica': result['morties_on_planet_jessica'],
            'morties_lost': result['morties_lost']
        }
        collector.trips_data.append(trip_data)

        # Print trip log normally (this will push the bar down)
        print(f"📤 Sent {trip_size} to P{chosen_planet} (pred={pred_rate:.2f}) → "
              f"Survived: {result['survived']}, "
              f"Jessica: {result['morties_on_planet_jessica']}, "
              f"Citadel: {result['morties_in_citadel']}")

        # Now update the bar: this will clear the log line and replace it with the bar
        print_fixed_progress_bar(
            result['morties_on_planet_jessica'],
            result['morties_in_citadel'],
            result['morties_lost']
        )

        time.sleep(0.05)

    # --- After the loop ends, get final status ---
    final_status = client.get_status()
    print("\n") # Add a newline after the final progress bar
    print("\nFinal Status:")
    print(f"Morties on Planet Jessica: {final_status['morties_on_planet_jessica']}")
    print(f"Morties Lost: {final_status['morties_lost']}")
    print(f"Steps Taken: {final_status['steps_taken']}")
    success_rate = (final_status['morties_on_planet_jessica'] / 1000) * 100
    print(f"Success Rate: {success_rate:.2f}%")

    return success_rate, pd.DataFrame(collector.trips_data).copy() # Return success rate and a copy of the data for this episode


def plot_poly_predictions_per_planet(df, ma_window=10):
    """
    Plot for each planet:
      - Actual outcomes (scatter)
      - Actual moving average (solid line)
      - Polynomial prediction (dashed line)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle('Survival Rate: Actual, Moving Avg & Polynomial Prediction (Rolling Buffer)', fontsize=16)

    colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    planet_names = {
        0: '"On a Cob" Planet',
        1: 'Cronenberg World',
        2: 'The Purge Planet'
    }

    for i, planet_idx in enumerate([0, 1, 2]):
        ax = axes[i]
        planet_df = df[df['planet'] == planet_idx].copy().reset_index(drop=True)
        planet_name = planet_names[planet_idx]

        if planet_df.empty:
            ax.set_title(f"{planet_name} (No trips)")
            ax.set_ylim(-5, 105)
            continue

        trip_nums = planet_df['trip_number'].values
        outcomes = planet_df['survived'].astype(int).values

        # 1. Actual outcomes
        ax.scatter(trip_nums, outcomes * 100, color=colors[planet_idx], alpha=0.5, s=15, label='Actual Outcome')

        # 2. Actual Moving Average (CRITICAL!)
        planet_df['ma'] = planet_df['survived'].rolling(window=ma_window, min_periods=1).mean()
        ax.plot(trip_nums, planet_df['ma'] * 100, color=colors[planet_idx], linewidth=2.2,
                linestyle='-', label=f'Actual Moving Avg ({ma_window}-trip)', alpha=0.9)

        # 3. Polynomial Prediction (using rolling buffer logic)
        pred_vals = []
        for idx in range(len(planet_df)):
            # Simulate rolling buffer up to this point
            hist = list(zip(planet_df['trip_number'][:idx+1], planet_df['survived'][:idx+1].astype(int)))
            if len(hist) > BUFFER_SIZE:
                hist = hist[-BUFFER_SIZE:]
            if len(hist) >= MIN_DATA_FOR_POLY:
                try:
                    t_nums = [h[0] for h in hist]
                    outs = [h[1] for h in hist]
                    coeffs = np.polyfit(t_nums, outs, POLY_DEGREE)
                    poly = np.poly1d(coeffs)
                    pred = np.clip(poly(trip_nums[idx]), 0, 1)
                    pred_vals.append(pred * 100)
                except:
                    pred_vals.append(np.mean(outs) * 100)
            else:
                pred_vals.append(50.0)
        
        ax.plot(trip_nums, pred_vals, color='purple', linestyle='--', linewidth=2,
                label=f'Polynomial Prediction (deg={POLY_DEGREE}, buffer={BUFFER_SIZE})')

        ax.set_title(planet_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Trip Number', fontsize=12)
        ax.set_ylabel('Survival Rate (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.show()


def run_integrated_agent(num_episodes=1):
    """
    Main function to run the integrated agent for a specified number of episodes.
    """
    print(f"🚀 Starting Morty Express Challenge with Polynomial Regression + UCB Agent (w/ Dynamic Trip Sizing & Progress Bar)...")
    print(f"Running {num_episodes} episode(s).")
    print(f"Buffer Size: {BUFFER_SIZE}, Poly Degree: {POLY_DEGREE}, Max Trip Size: {MAX_MORTIES_PER_TRIP}")

    # Initialize the API client
    try:
        client = SphinxAPIClient()
        print("✓ API client initialized successfully!")
    except ValueError as e:
        print(f"✗ Error initializing client: {e}")
        print("Make sure your .env file contains SPHINX_API_TOKEN=your_token_here")
        return

    success_rates = []
    all_episode_data = []

    for episode_num in range(num_episodes):
        print(f"\n--- Running Episode {episode_num + 1} ---")

        # Initialize the DataCollector with the client for this episode
        collector = DataCollector(client)

        # Run the single episode logic
        success_rate, episode_df = run_single_episode(client, collector)
        success_rates.append(success_rate)
        all_episode_data.append(episode_df)

    # --- After all episodes, print summary ---
    if num_episodes > 1:
        print("\n--- Results Summary ---")
        for i, rate in enumerate(success_rates):
            print(f"Episode {i+1}: {rate:.2f}%")
        avg_rate = np.mean(success_rates)
        print(f"Average Success Rate: {avg_rate:.2f}%")

    # --- Plot data from the *last* episode ---
    if not all_episode_data[-1].empty:
        print("\nGenerating plots using data from the LAST episode...")
        df_last = all_episode_data[-1]
        try:
            plot_survival_rates(df_last)
            plot_survival_by_planet(df_last)
            plot_moving_average(df_last, window=10)
            plot_risk_evolution(df_last)
            plot_episode_summary(df_last)
            
            # NEW: Plot Polynomial predictions vs actual
            # FIXED: Use the correct parameter name 'ma_window'
            print("\n- Polynomial Prediction vs Actual Survival...")
            plot_poly_predictions_per_planet(df_last, ma_window=10)

            print("\nAll plots generated successfully!")
        except Exception as e:
            print(f"Error generating plots: {e}")
    else:
        print("No data collected in the last episode to plot.")


def run_tuning(tuning_config, num_episodes_per_config=3):
    """
    Run hyperparameter tuning by testing different configurations.
    """
    print(f"🚀 Starting Hyperparameter Tuning...")
    print(f"Testing {len(tuning_config)} configurations, {num_episodes_per_config} episodes each.")

    results = []
    for i, config in enumerate(tuning_config):
        print(f"\n--- Testing Configuration {i+1}/{len(tuning_config)}: {config} ---")
        rates = []
        for j in range(num_episodes_per_config):
            # Temporarily set global variables based on config
            globals()['INITIAL_EXPLORATION_TRIPS'] = config['INITIAL_EXPLORATION_TRIPS']
            globals()['UCB_C'] = config['UCB_C']
            globals()['POLY_DEGREE'] = config['POLY_DEGREE']
            globals()['BUFFER_SIZE'] = config['BUFFER_SIZE']

            # Re-initialize agent state to reset any stateful variables based on new config
            global planet_buffer, planet_tries, total_trips_taken, last_poly_predictions
            planet_buffer = {0: [], 1: [], 2: []}
            planet_tries = {0: 0, 1: 0, 2: 0}
            total_trips_taken = 0
            last_poly_predictions = {0: 0.5, 1: 0.5, 2: 0.5}

            # Run a single episode (without plots or progress bar for tuning)
            try:
                client = SphinxAPIClient()
                collector = DataCollector(client)
                rate, _ = run_single_episode(client, collector)
                rates.append(rate)
                print(f"  Episode {j+1} Success Rate: {rate:.2f}%")
            except Exception as e:
                print(f"  Error running episode {j+1}: {e}")
                rates.append(0) # Default score if episode fails

        avg_rate = np.mean(rates)
        std_rate = np.std(rates)
        results.append({
            'config': config,
            'avg_rate': avg_rate,
            'std_rate': std_rate,
            'rates': rates
        })
        print(f"  -> Config {config} Average Rate: {avg_rate:.2f}% (std: {std_rate:.2f}%)")

    # Find the best configuration
    best_result = max(results, key=lambda x: x['avg_rate'])
    print(f"\n🏆 Best Configuration Found:")
    print(f"  Config: {best_result['config']}")
    print(f"  Average Success Rate: {best_result['avg_rate']:.2f}%")
    print(f"  Standard Deviation: {best_result['std_rate']:.2f}%")
    print(f"  Individual Rates: {[f'{r:.2f}%' for r in best_result['rates']]}")

    return results, best_result


if __name__ == "__main__":
    # Define the hyperparameter tuning configuration
    # This is a simple grid search example. Adjust the values as needed.
    tuning_config = [
        {'INITIAL_EXPLORATION_TRIPS': 5, 'UCB_C': 2.0, 'POLY_DEGREE': 2, 'BUFFER_SIZE': 50},
        {'INITIAL_EXPLORATION_TRIPS': 5, 'UCB_C': 2.0, 'POLY_DEGREE': 2, 'BUFFER_SIZE': 100},
        {'INITIAL_EXPLORATION_TRIPS': 5, 'UCB_C': 2.0, 'POLY_DEGREE': 3, 'BUFFER_SIZE': 50},
        {'INITIAL_EXPLORATION_TRIPS': 5, 'UCB_C': 2.0, 'POLY_DEGREE': 3, 'BUFFER_SIZE': 100},
        {'INITIAL_EXPLORATION_TRIPS': 10, 'UCB_C': 2.0, 'POLY_DEGREE': 2, 'BUFFER_SIZE': 100},
        {'INITIAL_EXPLORATION_TRIPS': 5, 'UCB_C': 1.5, 'POLY_DEGREE': 2, 'BUFFER_SIZE': 100},
        {'INITIAL_EXPLORATION_TRIPS': 5, 'UCB_C': 2.5, 'POLY_DEGREE': 2, 'BUFFER_SIZE': 100},
    ]

    # Check for command-line arguments
    num_episodes_to_run = 1 # Default to 1 episode
    if len(sys.argv) >= 2:
        try:
            arg = sys.argv[1]
            if arg == 'tune':
                # Run tuning instead of normal execution
                run_tuning(tuning_config, num_episodes_per_config=3) # Adjust num_episodes_per_config as needed
                sys.exit(0) # Exit after tuning
            else:
                num_episodes_to_run = int(arg)
                print(f"Command-line argument detected: Running {num_episodes_to_run} episodes.")
        except ValueError:
            print(f"Invalid argument '{sys.argv[1]}', running 1 episode or 'tune'.")
            num_episodes_to_run = 1

    run_integrated_agent(num_episodes_to_run)