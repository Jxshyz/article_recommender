import psutil
import time
from contextlib import contextmanager
import pandas as pd
from datetime import datetime
import platform
import os


@contextmanager
def resource_tracker(stage_name, sample_interval=0.5, duration=None):
    """
    Context manager to track CPU, RAM, power, CO2 usage, and CPU time over a block of code.

    Args:
        stage_name (str): Name of the stage (e.g., "Data Loading").
        sample_interval (float): Seconds between samples. Defaults to 0.5.
        duration (float, optional): Max duration to track (in seconds); if None, runs until block ends.

    Yields:
        None: The context manager yields control to the wrapped code block.

    Notes:
        - Tracks resource usage in a background thread.
        - Calculates power and CO2 estimates based on CPU usage.
        - Stores results in a global DataFrame and prints summary stats.
    """
    # Initialize lists to store resource usage samples
    cpu_samples = []
    ram_samples = []
    power_samples = []
    co2_samples = []
    memory_samples = []

    # Record start time and get current process
    start_time = time.time()
    process = psutil.Process()

    def sample_resources():
        """Background function to sample system resources at regular intervals."""
        while not stop_sampling:
            cpu = psutil.cpu_percent(interval=None)  # Get instantaneous CPU usage
            ram = psutil.virtual_memory().percent  # Get RAM usage percentage
            power = 10 + (cpu / 100) * 90  # Estimate power (10W base + 90W max CPU)
            co2 = (power / 1000) * 475  # Estimate CO2 in g/h (475g/kWh)
            mem = process.memory_info().rss / (1024**2)  # Process memory in MB
            cpu_samples.append(cpu)
            ram_samples.append(ram)
            power_samples.append(power)
            co2_samples.append(co2)
            memory_samples.append(mem)
            time.sleep(sample_interval)

    import threading

    # Set up and start sampling thread
    stop_sampling = False
    sampler = threading.Thread(target=sample_resources)
    sampler.start()

    # Record initial CPU time
    cpu_start_time = process.cpu_times().user + process.cpu_times().system

    try:
        yield  # Execute the code block within the context manager
    finally:
        # Stop sampling and calculate final metrics
        stop_sampling = True
        sampler.join()
        elapsed_time = time.time() - start_time
        cpu_end_time = process.cpu_times().user + process.cpu_times().system
        cpu_time_used = cpu_end_time - cpu_start_time

        # Calculate average power and total energy consumption
        avg_power = sum(power_samples) / len(power_samples) if power_samples else 0
        energy_used_wh = avg_power * (elapsed_time / 3600)

        # Compile statistics dictionary
        stats = {
            "Stage": stage_name,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Elapsed Time (s)": elapsed_time,
            "CPU Time (s)": cpu_time_used,
            "Avg CPU (%)": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
            "Peak CPU (%)": max(cpu_samples) if cpu_samples else 0,
            "Avg RAM (%)": sum(ram_samples) / len(ram_samples) if ram_samples else 0,
            "Peak RAM (%)": max(ram_samples) if ram_samples else 0,
            "Avg Memory (MB)": sum(memory_samples) / len(memory_samples) if memory_samples else 0,
            "Peak Memory (MB)": max(memory_samples) if memory_samples else 0,
            "Avg Power (W)": avg_power,
            "Peak Power (W)": max(power_samples) if power_samples else 0,
            "Total Energy (Wh)": energy_used_wh,
            "Avg CO2 (g/h)": sum(co2_samples) / len(co2_samples) if co2_samples else 0,
            "Total CO2 (g)": (avg_power / 1000) * 475 * (elapsed_time / 3600) if power_samples else 0,
            "Device Info": platform.platform(),
            "CPU Info": platform.processor(),
        }

        # Print summary of resource usage
        print(
            f"{stage_name} - Elapsed: {elapsed_time:.2f}s, CPU Time: {cpu_time_used:.2f}s, "
            f"Avg CPU: {stats['Avg CPU (%)']:.2f}%, Avg RAM: {stats['Avg RAM (%)']:.2f}%, "
            f"Avg Mem: {stats['Avg Memory (MB)']:.2f}MB"
        )

        # Update global resource DataFrame
        global resource_df
        if "resource_df" not in globals():
            resource_df = pd.DataFrame(columns=stats.keys())
        resource_df = pd.concat([resource_df, pd.DataFrame([stats])], ignore_index=True)


def save_resources(filename="resource_usage.csv"):
    """
    Append the tracked resources to a CSV file without overwriting existing data.

    Args:
        filename (str): Path to the CSV file. Defaults to "resource_usage.csv".

    Notes:
        - Uses global resource_df to store data between calls.
        - Appends new data to existing file if it exists, writes header only on first save.
    """
    global resource_df
    if "resource_df" in globals() and not resource_df.empty:
        file_exists = os.path.exists(filename)
        resource_df.to_csv(filename, mode="a", header=not file_exists, index=False)
        print(f"Appended resource usage to {filename}")
