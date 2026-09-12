import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

log_path = r"C:\Users\trang\.gemini\antigravity\brain\bb7e4c9e-5d80-45cb-a92c-c09da587de84\.system_generated\logs\transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        step = obj.get("step_index", 0)
        if 145 <= step < 192:
            print(f"=== Step {step} ({obj.get('source')}/{obj.get('type')}) ===")
            if "content" in obj:
                print(obj["content"][:1000])
            if "tool_calls" in obj:
                print(f"Tool calls: {obj['tool_calls']}")
            print("\n" + "="*50 + "\n")
