#!/usr/bin/env python3
"""
RQ3 Analysis 
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_script(script_path, script_name):
    """
    Run a Python script and handle its execution
    
    Args:
        script_path: Path object to the script
        script_name: Name of the script for logging
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running {script_name}")
    print(f"{'='*80}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,  # Run in the script's directory
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        print(f"✓ {script_name} completed successfully")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    
    except Exception as e:
        print(f"✗ Unexpected error running {script_name}: {str(e)}")
        return False


def main():
    """Main execution function"""
    print("="*80)
    print("RQ3 HYPOTHESIS TESTS")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Define the scripts to run in order
    scripts = [
        "h1.py",
        "h2h6.py",
    ]
    
    # Track results
    results = {}
    base_path = Path(__file__).parent / "RQ3_tests"
    
    # Run each script
    for script_name in scripts:
        script_path = base_path / script_name
        
        # Check if script exists
        if not script_path.exists():
            print(f"\n✗ Warning: {script_path} not found!")
            results[script_name] = False
            continue
        
        # Run the script
        success = run_script(script_path, script_name)
        results[script_name] = success
        
        # Stop if a script fails (optional - remove if you want to continue on errors)
        if not success:
            print(f"\n⚠ {script_name} failed. Stopping execution.")
            break
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    for script_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {script_name}")
    
    # Overall result
    all_success = all(results.values())
    print("\n" + "="*80)
    if all_success:
        print("✓ ALL HYPOTHESIS TESTS COMPLETED SUCCESSFULLY")
    else:
        print("✗ SOME HYPOTHESIS TESTS FAILED")
    print(f"Analysis finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return 0 if all_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)