from headless_simulation_core import run_headless_simulation
import time

experiments = [
   
    {
        "name": "Baseline_With_Assistance",
        "config": {
            # Using all defaults
        },
        "results_file": "results/baseline.csv"
    },
    {
        "name": "More_Neurons",
        "config": {
            "hidden_nodes": 20, # Changed parameter
        },
        "results_file": "results/more_neurons.csv"
    },
    {
        "name": "No_Auto_Flee",
        "config": {
            "use_auto_flee": False # Changed parameter
        },
        "results_file": "results/no_auto_flee.csv"
    },
    {
        "name": "No_Assistance_At_All",
        "config": {
            "use_auto_gather": False, # Changed parameter
            "use_auto_flee": False    # Changed parameter
        },
        "results_file": "results/no_assistance.csv"
    },
    {
        "name": "Higher_Mutation",
        "config": {
            "mutation_rate": 0.2,
            "mutation_strength": 0.8
        },
        "results_file": "results/higher_mutation.csv"
    }
    ,

    {
        "name": "Higher_Mutation_with_no_help",
        "config": {
            #"mutation_rate": 0.2,
            #"mutation_strength": 0.8,
            "hidden_nodes": 20

        },
        "results_file": "results/higher_mutation_no_help.csv"
    }
]


# --- Main Runner ---
if __name__ == "__main__":
    TOTAL_GENERATIONS_PER_RUN = 50 # Set how many generations each experiment should last
    start_time = time.time()

    print(f"Starting {len(experiments)} experiments, each running for {TOTAL_GENERATIONS_PER_RUN} generations.")

    for i, exp in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{len(experiments)}: {exp['name']} ---")

        # Add the experiment name to its config for logging purposes
        exp['config']['name'] = exp['name']

        run_headless_simulation(
            config=exp['config'],
            total_generations=TOTAL_GENERATIONS_PER_RUN,
            results_csv_path=exp['results_file']
        )
        print(f"--- Finished Experiment: {exp['name']}. Results saved to {exp['results_file']} ---")

    end_time = time.time()
    print(f"\nAll experiments completed in {end_time - start_time:.2f} seconds.")