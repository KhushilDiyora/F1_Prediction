import pandas as pd
import numpy as np
import time

# Track-specific modifiers for multiple GPs
track_factors = {
    'bahrain': {
        'base_time': 89.5,
        'track_type': 'desert',
        'team_modifiers': {
            'Red Bull Racing': 0.996,
            'Ferrari': 0.997,
            'McLaren': 0.998,
            'Mercedes': 0.999,
            'Aston Martin': 1.000,
            'AlphaTauri': 1.001,
            'Williams': 1.002,
            'Haas F1 Team': 1.003,
            'Alfa Romeo': 1.004,
            'Alpine': 1.005,
        },
        'driver_modifiers': {
            'Max Verstappen': 0.996,
            'Charles Leclerc': 0.997,
            'Lando Norris': 0.998,
            'Lewis Hamilton': 0.999,
            'Carlos Sainz': 0.999,
            'George Russell': 1.000,
            'Fernando Alonso': 1.000,
            'Oscar Piastri': 1.001,
            'Sergio Perez': 1.001,
            'Yuki Tsunoda': 1.002,
        }
    },
    'jeddah': {
        'base_time': 87.8,
        'track_type': 'street',
        'team_modifiers': {
            'Red Bull Racing': 0.997,
            'Ferrari': 0.998,
            'McLaren': 0.999,
            'Mercedes': 1.000,
            'Aston Martin': 1.001,
            'AlphaTauri': 1.002,
            'Williams': 1.003,
            'Haas F1 Team': 1.004,
            'Alfa Romeo': 1.005,
            'Alpine': 1.006,
        },
        'driver_modifiers': {
            'Max Verstappen': 0.997,
            'Charles Leclerc': 0.998,
            'Lando Norris': 0.999,
            'Lewis Hamilton': 0.998,
            'Carlos Sainz': 0.999,
            'George Russell': 1.000,
            'Fernando Alonso': 1.000,
            'Oscar Piastri': 1.000,
            'Sergio Perez': 1.001,
            'Yuki Tsunoda': 1.002,
        }
    },
    'australia': {
        'base_time': 81.5,
        'track_type': 'street',
        'team_modifiers': {
            'Red Bull Racing': 0.996,
            'Ferrari': 0.997,
            'McLaren': 0.998,
            'Mercedes': 0.998,
            'Aston Martin': 0.999,
            'AlphaTauri': 1.000,
            'Williams': 1.001,
            'Haas F1 Team': 1.002,
            'Alfa Romeo': 1.003,
            'Alpine': 1.004,
        },
        'driver_modifiers': {
            'Max Verstappen': 0.996,
            'Lewis Hamilton': 0.997,
            'Charles Leclerc': 0.998,
            'Lando Norris': 0.998,
            'George Russell': 0.998,
            'Oscar Piastri': 0.999,
            'Fernando Alonso': 1.000,
            'Sergio Perez': 1.000,
            'Carlos Sainz': 1.000,
            'Valtteri Bottas': 1.001,
        }
    },
    'monaco': {
        'base_time': 71.0,
        'track_type': 'street',
        'team_modifiers': {
            'Red Bull Racing': 0.997,
            'Ferrari': 0.998,
            'McLaren': 0.999,
            'Mercedes': 1.000,
            'Aston Martin': 1.001,
            'AlphaTauri': 1.002,
            'Williams': 1.003,
            'Haas F1 Team': 1.004,
            'Alfa Romeo': 1.005,
            'Alpine': 1.006,
        },
        'driver_modifiers': {
            'Max Verstappen': 0.997,
            'Charles Leclerc': 0.998,
            'Lando Norris': 0.999,
            'Lewis Hamilton': 0.998,
            'Carlos Sainz': 0.999,
            'George Russell': 1.000,
            'Fernando Alonso': 1.000,
            'Oscar Piastri': 1.000,
            'Sergio Perez': 1.001,
            'Yuki Tsunoda': 1.002,
        }
    },
    'spain': {
        'base_time': 78.5,
        'track_type': 'permanent',
        'team_modifiers': {
            'Red Bull Racing': 0.996,
            'Ferrari': 0.997,
            'McLaren': 0.998,
            'Mercedes': 0.998,
            'Aston Martin': 1.000,
            'AlphaTauri': 1.001,
            'Williams': 1.002,
            'Haas F1 Team': 1.003,
            'Alfa Romeo': 1.004,
            'Alpine': 1.005,
        },
        'driver_modifiers': {
            'Max Verstappen': 0.996,
            'Lewis Hamilton': 0.997,
            'Charles Leclerc': 0.998,
            'Lando Norris': 0.998,
            'George Russell': 0.998,
            'Oscar Piastri': 0.999,
            'Fernando Alonso': 1.000,
            'Sergio Perez': 1.000,
            'Carlos Sainz': 1.000,
            'Valtteri Bottas': 1.001,
        }
    }
}

historical_pole_times = {
    'bahrain': 88.265,
    'jeddah': 87.241,
    'australia': 80.703,
    'monaco': 70.270,
    'spain': 77.978,
}

driver_form = {
    'Max Verstappen': 0.998,
    'Charles Leclerc': 1.000,
    'Lando Norris': 0.999,
    'Lewis Hamilton': 1.000,
    'Carlos Sainz': 1.000,
    'George Russell': 1.000,
    'Fernando Alonso': 1.000,
    'Oscar Piastri': 1.001,
    'Sergio Perez': 1.000,
    'Yuki Tsunoda': 1.002,
    'Daniel Ricciardo': 1.000,
    'Lance Stroll': 1.001,
    'Alex Albon': 1.001,
    'Logan Sargeant': 1.001,
    'Kevin Magnussen': 1.001,
    'Nico Hulkenberg': 1.001,
    'Valtteri Bottas': 1.002,
    'Zhou Guanyu': 1.002,
    'Esteban Ocon': 1.001,
    'Pierre Gasly': 1.001,
}

def apply_performance_factors(predictions_df, gp_name='monaco'):
    gp_key = gp_name.lower()
    factors = track_factors.get(gp_key, track_factors['monaco'])

    base_time = factors['base_time']
    team_factors = factors['team_modifiers']
    driver_factors = factors['driver_modifiers']

    np.random.seed(int(time.time()) % 100000)
    for idx, row in predictions_df.iterrows():
        team_factor = team_factors.get(row['Team'], 1.005)
        driver_factor = driver_factors.get(row['Driver'], 1.002)
        form_factor = driver_form.get(row['Driver'], 1.000)
        base_prediction = base_time * team_factor * driver_factor * form_factor
        random_variation = np.random.uniform(-0.1, 0.1)
        predictions_df.loc[idx, 'Predicted_Q3'] = base_prediction + random_variation

    return predictions_df

def predict_gp(gp_name='monaco'):
    # 2025 driver lineups with real teams
    driver_teams = {
        'Max Verstappen': 'Red Bull Racing',
        'Sergio Perez': 'Red Bull Racing',
        'Charles Leclerc': 'Ferrari',
        'Lewis Hamilton': 'Ferrari',
        'Carlos Sainz': 'Ferrari',
        'Lando Norris': 'McLaren',
        'Oscar Piastri': 'McLaren',
        'George Russell': 'Mercedes',
        'Valtteri Bottas': 'Mercedes',
        'Fernando Alonso': 'Aston Martin',
        'Lance Stroll': 'Aston Martin',
        'Pierre Gasly': 'Alpine',
        'Esteban Ocon': 'Alpine',
        'Yuki Tsunoda': 'AlphaTauri',
        'Daniel Ricciardo': 'AlphaTauri',
        'Kevin Magnussen': 'Haas F1 Team',
        'Nico Hulkenberg': 'Haas F1 Team',
        'Alex Albon': 'Williams',
        'Logan Sargeant': 'Williams',
        'Zhou Guanyu': 'Alfa Romeo',
        'Guanyu Zhou': 'Alfa Romeo',
    }

    results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])
    results_df = apply_performance_factors(results_df, gp_name=gp_name)
    results_df = results_df.sort_values('Predicted_Q3')
    return results_df

def display_results(results_df, gp_name):
    pole_time = historical_pole_times.get(gp_name.lower())

    print(f"\n{gp_name.title()} GP 2025 Qualifying Predictions:")
    print("=" * 80)
    print(f"{'Pos':<5}{'Driver':<20}{'Team':<25}{'Pred Time':<15}", end="")
    if pole_time:
        print(f"{'Delta to 2023 Pole':<15}")
    else:
        print()
    print("-" * 80)

    for idx, row in results_df.iterrows():
        position = results_df.index.get_loc(idx) + 1
        pred_time = row['Predicted_Q3']
        print(f"{position:<5}{row['Driver']:<20}{row['Team']:<25}{pred_time:.3f}s", end="")
        if pole_time:
            print(f"{pred_time - pole_time:+.3f}s")
        else:
            print()

def save_predictions(results_df, gp_name):
    filename = f"{gp_name.lower()}_gp_2025_qualifying_predictions.csv"
    results_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def calculate_accuracy(results_df, gp_name):
    pole_time = historical_pole_times.get(gp_name.lower())
    if not pole_time:
        return None

    # Predicted pole = fastest predicted time
    predicted_pole = results_df['Predicted_Q3'].min()

    # Accuracy: 1 - (abs(predicted - actual)/actual)
    accuracy = 1 - abs(predicted_pole - pole_time) / pole_time
    return accuracy

def main():
    available_gps = list(track_factors.keys())
    print("Available GPs:", ", ".join(available_gps))
    
    gp_input = input("Enter the GP name to predict qualifying (e.g., spain): ").strip().lower()
    
    if gp_input not in available_gps:
        print(f"Sorry, predictions for '{gp_input}' are not available.")
        return
    
    predictions = predict_gp(gp_input)
    display_results(predictions, gp_input)
    
    accuracy = calculate_accuracy(predictions, gp_input)
    if accuracy is not None:
        print(f"\nModel Accuracy compared to 2023 pole time: {accuracy*100:.2f}%")
    else:
        print("\nNo historical pole time data to calculate accuracy.")
    
    save_predictions(predictions, gp_input)

if __name__ == "__main__":
    main()
