import os
log_path = r"c:\Users\acord\OneDrive\Desktop\Bot\feat_sniper_mcp\logs\compile_mql5.log"
output_path = r"c:\Users\acord\OneDrive\Desktop\Bot\feat_sniper_mcp\logs\debug_output.txt"
if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-16') as f:
        log_content = f.read()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(log_content)
    print(f"Log dumped to {output_path}")
else:
    print("Log not found")
