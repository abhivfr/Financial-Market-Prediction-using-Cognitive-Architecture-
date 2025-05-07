import os
import pkg_resources
from datetime import datetime

def get_project_structure(root_dir, output_file):
    """Write the project folder structure to the output file"""
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("PROJECT STRUCTURE\n")
    output_file.write("="*80 + "\n\n")
    
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        output_file.write(f"{indent}{os.path.basename(root)}/\n")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            output_file.write(f"{subindent}{f}\n")

def get_file_contents(root_dir, output_file, extensions=None):
    """Write the contents of all files with given extensions to the output file"""
    if extensions is None:
        extensions = ['.py', '.txt', '.md', '.json', '.yaml', '.yml', '.sh', '.js', '.html', '.css']
    
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("FILE CONTENTS\n")
    output_file.write("="*80 + "\n\n")
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    output_file.write(f"\n{'='*40}\n")
                    output_file.write(f"FILE: {file_path}\n")
                    output_file.write(f"{'='*40}\n\n")
                    output_file.write(content + "\n")
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                        output_file.write(f"\n{'='*40}\n")
                        output_file.write(f"FILE: {file_path} (latin-1 encoding)\n")
                        output_file.write(f"{'='*40}\n\n")
                        output_file.write(content + "\n")
                    except Exception as e:
                        output_file.write(f"\nCould not read {file_path}: {str(e)}\n")
                except Exception as e:
                    output_file.write(f"\nCould not read {file_path}: {str(e)}\n")

def get_venv_dependencies(venv_path, output_file):
    """Write the dependencies from the virtual environment to the output file"""
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("VIRTUAL ENVIRONMENT DEPENDENCIES\n")
    output_file.write("="*80 + "\n\n")
    
    # Path to the site-packages directory in the virtual environment
    site_packages = os.path.join(venv_path, 'lib', 'site-packages')
    
    if not os.path.exists(site_packages):
        output_file.write("Could not find site-packages directory in the virtual environment.\n")
        return
    
    try:
        # Get all installed packages
        installed_packages = pkg_resources.find_distributions(site_packages)
        
        output_file.write("Installed packages:\n")
        output_file.write("-"*40 + "\n")
        for package in installed_packages:
            output_file.write(f"{package.key}=={package.version}\n")
        
        # Get requirements.txt if it exists
        req_file = os.path.join(venv_path, 'requirements.txt')
        if os.path.exists(req_file):
            output_file.write("\n" + "="*40 + "\n")
            output_file.write("requirements.txt CONTENTS\n")
            output_file.write("="*40 + "\n\n")
            with open(req_file, 'r') as f:
                output_file.write(f.read())
    except Exception as e:
        output_file.write(f"\nCould not get dependencies: {str(e)}\n")

def main():
    # Configuration
    project_folder = input("Enter the full path to your project folder: ").strip()
    venv_folder = input("Enter the full path to your tf_venv folder: ").strip()
    output_filename = "project_contents_and_dependencies.txt"
    
    # Verify paths
    if not os.path.isdir(project_folder):
        print(f"Error: Project folder '{project_folder}' does not exist.")
        return
    
    if not os.path.isdir(venv_folder):
        print(f"Error: Virtual environment folder '{venv_folder}' does not exist.")
        return
    
    # Create output file
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        # Write header
        output_file.write(f"PROJECT CONTENTS AND DEPENDENCIES REPORT\n")
        output_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write(f"Project folder: {project_folder}\n")
        output_file.write(f"Virtual environment: {venv_folder}\n")
        
        # Get project structure
        get_project_structure(project_folder, output_file)
        
        # Get file contents
        get_file_contents(project_folder, output_file)
        
        # Get venv dependencies
        get_venv_dependencies(venv_folder, output_file)
    
    print(f"\nSuccessfully created report file: {output_filename}")

if __name__ == "__main__":
    main()
