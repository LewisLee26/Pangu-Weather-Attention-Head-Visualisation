import argparse
import subprocess
from datetime import datetime, timedelta
from tqdm import tqdm

def increment_date(current_date, increment):
    """Increment a date by a given number of days."""
    date_obj = datetime.strptime(current_date, "%Y-%m-%d")
    new_date = date_obj + timedelta(days=increment)
    return new_date.strftime("%Y-%m-%d")

def get_dates_in_range(start_date, end_date):
    """Generate a list of dates between start_date and end_date inclusive."""
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date = increment_date(current_date, 1)
    return dates

def run_command(command):
    """Run a shell command."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")

def download_data(start_date, end_date):
    """Phase 1: Download data."""
    print(f"Phase 1: Downloading data for range {start_date} to {end_date}...")
    command = f"python scripts/download_data.py --start_date {start_date} --end_date {end_date}"
    run_command(command)

def save_activations(dates, model_num, intermediate_layers, num_threads):
    """Phase 2: Save activations."""
    print(f"Phase 2: Saving activations for range {dates[0]} to {dates[-1]}...")
    for date in tqdm(dates, desc="Phase 2 Progress"):
        for time in ["00:00", "12:00"]:
            command = (
                f"python scripts/save_activations.py "
                f"--model_num {model_num} "
                f"--data_date {date} "
                f"--data_time {time} "
                f"--intermediate_layers {intermediate_layers} "
                f"--num_threads {num_threads}"
            )
            run_command(command)

def format_data(dates, intermediate_layers):
    """Phase 3: Format data."""
    print(f"Phase 3: Formatting data for range {dates[0]} to {dates[-1]}...")
    for date in tqdm(dates, desc="Phase 3 Progress"):
        for time in ["00:00", "12:00"]:
            command = (
                f"python scripts/format_data.py "
                f"--data_date {date} "
                f"--data_time {time} "
                f"--intermediate_layers {intermediate_layers}"
            )
            run_command(command)

def main():
    parser = argparse.ArgumentParser(description="Process data with progress bars.")
    parser.add_argument("--start_date", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--model_num", type=int, default=24, help="Model number.")
    parser.add_argument("--intermediate_layers", default="0 1 2 3", help="Intermediate layers.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads.")

    args = parser.parse_args()

    dates = get_dates_in_range(args.start_date, args.end_date)

    download_data(args.start_date, args.end_date)
    save_activations(dates, args.model_num, args.intermediate_layers, args.num_threads)
    format_data(dates, args.intermediate_layers)

    print(f"Data setup completed for range {args.start_date} to {args.end_date}.")

if __name__ == "__main__":
    main()
