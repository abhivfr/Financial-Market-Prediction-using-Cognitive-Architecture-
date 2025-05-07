import os
import sys
import importlib.util
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run model analysis")
    parser.add_argument("--cognitive", required=True, help="Path to cognitive model")
    parser.add_argument("--baseline", required=True, help="Path to baseline model")
    parser.add_argument("--test_data", required=True, help="Path to test data")
    parser.add_argument("--output", default="results/comparison", help="Output directory")
    
    args = parser.parse_args()
    
    # Get the absolute path to the scripts/model_compare_fixed.py file
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scripts",
        "model_compare_fixed.py"
    )
    
    # Check if the script exists
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Load the module from the script path
    try:
        spec = importlib.util.spec_from_file_location("model_compare_fixed", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Call the main function with arguments
        sys.argv = [
            script_path,
            "--cognitive", args.cognitive,
            "--baseline", args.baseline,
            "--test_data", args.test_data,
            "--output", args.output
        ]
        module.main()
        
        print("Analysis complete!")
    except Exception as e:
        print(f"Error running analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

print("Analysis complete!")
