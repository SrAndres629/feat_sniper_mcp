from datetime import datetime, timezone
import sys
import os
import subprocess
import json

# Ensure parent directory is in path to import nexus modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def run_deep_audit(auto_repair: bool = False):
    """
    SKILL: AUDITOR PROFUNDA
    Ejecuta la auditoría maestra del sistema NEXUS y opcionalmente aplica correcciones.
    Returns: Dict con el estado del sistema y reporte de anomalías.
    """
    try:
        # Run the auditor script directly to get the JSON output
        # We assume nexus_auditor.py is in the project root
        project_root = os.getcwd() # Assumption: Skill runs with CWD at root
        auditor_path = os.path.join(project_root, "nexus_auditor.py")
        
        if not os.path.exists(auditor_path):
             # Fallback if we are in app/skills
             auditor_path = os.path.join(project_root, "../../nexus_auditor.py")

        cmd = ["python", "nexus_auditor.py"]
        
        # Capture output
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=project_root)
        
        # Extract JSON tracer
        import re
        json_match = re.search(r'REPAIR_REQUEST_START\s*(\{.*?\})\s*REPAIR_REQUEST_END', result.stdout, re.DOTALL)
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "UNKNOWN",
            "anomalies": [],
            "critical_fixes_applied": False
        }
        
        if json_match:
            data = json.loads(json_match.group(1))
            report["anomalies"] = data.get("anomalies", [])
            report["critical"] = data.get("critical", [])
            
            if not report["critical"]:
                report["status"] = "OPERATIONAL"
            else:
                report["status"] = "CRITICAL_FAILURE"
                
                if auto_repair:
                    # Trigger the orchestrator's repair logic
                    # For now, we simulate this by returning the intent, 
                    # as nexus_control.py handles the actual process management
                    report["repair_instruction"] = "Run 'python nexus_control.py start' to trigger Deep Healer."
        
        # Check for generated report file
        report_file = os.path.join(project_root, "AUDIT_REPORT.md")
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                report["report_content"] = f.read()

        return report

    except Exception as e:
        return {"error": str(e), "status": "SKILL_FAILURE"}

# Entry point for MCP
tool_name = "skill_auditor_profunda"
tool_description = "Ejecuta auditoría maestra del sistema NEXUS con capacidad de diagnóstico profundo."

def execute(parameters: dict):
    repair = parameters.get("auto_repair", False)
    return run_deep_audit(repair)
