import subprocess
import sys
import os

# Get full paths
script_path = os.path.abspath("scripts/enhance_features.py")
output_dir = os.path.abspath("./data")
input_file = os.path.abspath("data/combined_financial.csv")  # Use an existing file

# Make sure the directory exists
os.makedirs(output_dir, exist_ok=True)

# Print debug info
print(f"Current working directory: {os.getcwd()}")
print(f"Script path: {script_path}")
print(f"Output directory: {output_dir}")
print(f"Input file: {input_file}")
print(f"Script exists: {os.path.exists(script_path)}")
print(f"Output dir exists: {os.path.exists(output_dir)}")
print(f"Input file exists: {os.path.exists(input_file)}")

# Test command with input file
cmd = [
    sys.executable,
    script_path,
    "--input_file", input_file,
    "--output_dir", output_dir,
    "--calculate_technicals"
]

print(f"Command: {' '.join(cmd)}")

# Run the command
result = subprocess.run(
    cmd, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE,
    text=True
)

# Print the output
print("\nSTDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")
