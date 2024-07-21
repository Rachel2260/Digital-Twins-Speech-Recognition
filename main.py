import subprocess
import sys

def run_script(script_name):
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e.stderr}")
        sys.exit(1)

def main():
    scripts = ['whisper_script.py', 'NLP_basic_info.py']
    
    for script in scripts:
        print(f"Running {script}...")
        run_script(script)
        print(f"Finished running {script}.\n")

if __name__ == "__main__":
    main()
