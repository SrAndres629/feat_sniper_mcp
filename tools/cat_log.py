import sys

def cat_log(path):
    try:
        with open(path, 'rb') as f:
            data = f.read()
            # Decode UTF-16 LE
            try:
                text = data.decode('utf-16')
            except:
                text = data.decode('utf-8', errors='ignore')
                
            for line in text.splitlines():
                if "error" in line.lower() or "warning" in line.lower():
                    print(f"LOG: {line.strip()}")
    except Exception as e:
        print(f"Error reading log: {e}")

if __name__ == "__main__":
    cat_log("logs/compile_mql5.log")
