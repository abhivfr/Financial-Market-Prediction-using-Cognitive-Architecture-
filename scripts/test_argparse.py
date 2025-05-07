# Save as scripts/test_argparse.py
import argparse
import sys
import os # Import os to check cwd

print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"sys.executable: {sys.executable}")
print(f"sys.argv: {sys.argv}") # This shows what arguments Python receives

parser = argparse.ArgumentParser(description="A simple test script for argparse")

parser.add_argument("--name", type=str, help="Your name")
parser.add_argument("--age", type=int, help="Your age")
parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
parser.add_argument("--input-file", type=str, help="A dummy input file path") # Added a file path argument

print("Attempting to parse arguments...")

try:
    # parse_args() by default parses sys.argv[1:]
    args = parser.parse_args()

    print("\nArguments parsed successfully:")
    print(f"Name: {args.name}")
    print(f"Age: {args.age}")
    print(f"Verbose: {args.verbose}")
    print(f"Input File: {args.input_file}")

except SystemExit as e:
    # argparse.parse_args() calls sys.exit() on failure by default
    print(f"\nArgparse failed with SystemExit. Exit code: {e.code}")
    print("This indicates an issue with argument parsing.")
    # Argparse prints its usage message to stderr before exiting.
    # The error message itself might provide clues.

except Exception as e:
    print(f"\nAn unexpected error occurred during argparse parsing: {e}")
    import traceback
    traceback.print_exc()


print("\nTest finished.")
