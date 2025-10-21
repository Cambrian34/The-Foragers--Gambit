import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
FILE_FORMAT = "png" 


METRICS_TO_PLOT = {
    "avg_fitness": "Average Fitness per Generation",
    "best_fitness": "Best Fitness per Generation",
    "total_apples_deposited": "Total Apples Deposited per Generation",
    "creatures_remaining": "Creatures Remaining at End of Generation"
}

# --- Main Plotting Logic ---
def create_comparison_plots():
    """
    Finds all experiment CSVs in the results directory and generates
    a comparison plot for each specified metric.
    """
    print(f"Looking for result files in '{RESULTS_DIR}/'...")

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created directory: '{PLOTS_DIR}/'")

    try:
        csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
        if not csv_files:
            print(f"Error: No .csv files found in the '{RESULTS_DIR}' directory.")
            print("Please run 'run_experiments.py' first to generate some data.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{RESULTS_DIR}' was not found.")
        print("Please ensure you have run 'run_experiments.py' and it has created the results folder.")
        return

    print(f"Found {len(csv_files)} result files. Generating plots...")

    # Generate one plot for each metric
    for metric, title in METRICS_TO_PLOT.items():
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot data from each CSV file
        for filename in csv_files:
            try:
                # Create a clean label name from the filename
                experiment_name = os.path.splitext(filename)[0].replace('_', ' ').title()
                
                # Read the data using pandas
                file_path = os.path.join(RESULTS_DIR, filename)
                df = pd.read_csv(file_path)

                # Check if the metric column exists in this file
                if metric in df.columns:
                    # Plot the generation vs. the metric
                    ax.plot(df['generation'], df[metric], marker='o', linestyle='-', markersize=4, label=experiment_name)
                else:
                    print(f"Warning: Metric '{metric}' not found in '{filename}'. Skipping.")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

        ax.set_title(title, fontsize=18, pad=20)
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        
        legend = ax.legend(title='Experiments', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout(rect=[0, 0, 0.85, 1]) 

        # Save the plot to a file
        output_filename = os.path.join(PLOTS_DIR, f"{metric}_comparison.{FILE_FORMAT}")
        fig.savefig(output_filename, dpi=150)
        print(f"  - Saved plot: {output_filename}")
        
        plt.close(fig)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    try:
        import pandas
        import matplotlib
    except ImportError:
        print("---")
        print("Error: Required libraries not found.")
        print("Please install pandas and matplotlib by running:")
        print("pip install pandas matplotlib")
        print("---")
    else:
        create_comparison_plots()
