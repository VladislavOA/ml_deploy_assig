import argparse
from datetime import datetime
import subprocess
import sys
import time
import requests


def run_pipeline_check(api_url: str) -> bool:
    """
    Executes 'dvc repro' to detect changes in raw data, code, or dependencies.
    If DVC executes stages and produces updated artifacts, notifies FastAPI to reload the model.
    Returns True if stages were re-run, False otherwise.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Running DVC pipeline check ('dvc repro')...")

    try:
        # Run dvc repro using the current python environment or system dvc
        result = subprocess.run(
            ["dvc", "repro"],
            capture_output=True,
            text=True,
            check=False,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        full_output = (stdout + "\n" + stderr).strip()

        if result.returncode != 0:
            print(f"[{timestamp}] [ERROR] 'dvc repro' failed with exit code {result.returncode}:")
            print(full_output)
            return False

        # Determine if any stage was executed
        stages_executed = any(keyword in full_output for keyword in ["Running stage", "checking out outputs"])
        up_to_date = "Data and pipelines are up to date." in full_output

        if up_to_date and not stages_executed:
            print(f"[{timestamp}] [OK] Pipeline is up to date. No data or code changes detected.")
            return False

        print(f"[{timestamp}] [SUCCESS] DVC updated the pipeline stages!")
        for line in full_output.splitlines():
            if line.strip():
                print(f"    {line}")

        # Notify FastAPI to hot-reload the updated model
        print(f"[{timestamp}] Sending reload signal to API at {api_url}...")
        try:
            resp = requests.post(api_url, timeout=10)
            if resp.status_code == 200:
                print(f"[{timestamp}] [RELOADED] Model hot-reloaded successfully in FastAPI service!")
                print(f"    Response: {resp.json()}")
            else:
                print(f"[{timestamp}] [WARNING] API responded with status {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"[{timestamp}] [WARNING] Could not reach API container at {api_url}: {req_err}")
            print("    (Ensure Docker containers are running with 'docker compose up -d')")

        return True

    except FileNotFoundError:
        print(f"[{timestamp}] [ERROR] 'dvc' command not found in system PATH. Ensure DVC is installed and activated.")
        return False
    except Exception as exc:
        print(f"[{timestamp}] [ERROR] Unexpected exception during pipeline check: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="5-minute automated pipeline scheduler using DVC and FastAPI model hot-reloading."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval between checks in seconds (default: 300, i.e., 5 minutes).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/pipeline/reload",
        help="FastAPI hot-reload endpoint URL (default: http://localhost:8000/pipeline/reload).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run pipeline check once and exit immediately (useful for testing and CI).",
    )

    args = parser.parse_args()

    mode_str = "Single Run (--once)" if args.once else "Continuous Loop"
    print("=" * 70)
    print("Customer Churn MLOps Pipeline Scheduler (DVC Orchestrator)")
    print(f"  * Check Interval : {args.interval} seconds (5 minutes)")
    print(f"  * API Endpoint   : {args.api_url}")
    print(f"  * Mode           : {mode_str}")
    print("=" * 70)

    if args.once:
        run_pipeline_check(args.api_url)
        return

    print("Starting scheduler loop. Press Ctrl+C to stop.")
    try:
        while True:
            run_pipeline_check(args.api_url)
            print(f"Waiting {args.interval} seconds until next pipeline check...")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")
        sys.exit(0)


if __name__ == '__main__':
    main()
