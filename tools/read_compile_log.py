import os

LOG_FILE = "logs/compile_mql5.log"

def read_log():
    if not os.path.exists(LOG_FILE):
        print("Log file not found.")
        return

    try:
        with open(LOG_FILE, 'r', encoding='utf-16') as f:
            lines = f.readlines()
            
        print(f"Total Lines: {len(lines)}")
        for line in lines:
            if "error" in line.lower() or "warning" in line.lower():
                print(line.strip())
                
    except Exception as e:
        print(f"Error reading log: {e}")

if __name__ == "__main__":
    read_log()
