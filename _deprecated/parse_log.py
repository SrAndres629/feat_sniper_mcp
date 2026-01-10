import os

app_data = r"C:\Users\acord\AppData\Roaming\MetaQuotes\Terminal\065434634B76DD288A1DDF20131E8DDB"
log_path = os.path.join(app_data, "MQL5", "Experts", "FEAT_Sniper", "UnifiedModel_Main.mq5.log")
output_path = r"c:\Users\acord\OneDrive\Desktop\Bot\feat_sniper_mcp\error_report.txt"

if not os.path.exists(log_path):
    print(f"Log not found at {log_path}")
    exit(1)

with open(log_path, "rb") as f:
    content = f.read().decode("utf-16le")

lines = content.splitlines()
errors = [line for line in lines if "error" in line.lower() or "warning" in line.lower()]

report = []
report.append(f"Total Errors/Warnings: {len(errors)}")
report.append("-" * 30)

files_with_errors = {}
for err in errors:
    parts = err.split(" : ")
    if len(parts) > 1:
        file_path = parts[0]
        filename = file_path.split("\\")[-1]
        msg = " : ".join(parts[1:])
        if filename not in files_with_errors:
            files_with_errors[filename] = []
        files_with_errors[filename].append(msg)

for filename, msgs in files_with_errors.items():
    report.append(f"\nFILE: {filename} ({len(msgs)} issues)")
    for msg in msgs:
        report.append(f"  - {msg}")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"Report written to {output_path}")
