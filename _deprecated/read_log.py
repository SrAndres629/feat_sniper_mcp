
try:
    with open(r'c:\Users\acord\OneDrive\Desktop\Bot\feat_sniper_mcp\FEAT_Sniper_Master_Core\compile_UnifiedModel_Main.log', 'r', encoding='utf-16') as f:
        for line in f:
            if "error" in line.lower():
                print(line.strip())
except Exception as e:
    print(f"Error reading log: {e}")
