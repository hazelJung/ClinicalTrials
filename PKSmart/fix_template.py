
import os

file_path = r"c:/Users/User/Desktop/aivle/project/bigproject/PKSmart/app/templates/project_detail.html"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: The broken variable assignment
broken_str = """var thalfVal = {{ pred.result_json.pk.human_thalf_linear }
                                };"""
fixed_str = """var thalfVal = {{ pred.result_json.pk.human_thalf_linear }};"""

# Normalize line endings for reliable replacement
if broken_str in content:
    print("Found broken string match!")
    content = content.replace(broken_str, fixed_str)
else:
    # Try fuzzy match (just the broken line)
    broken_line = "var thalfVal = {{ pred.result_json.pk.human_thalf_linear }"
    if broken_line in content:
        print("Found broken line, attempting manual fix...")
        # Check if next line is close brace
        idx = content.find(broken_line)
        # Look ahead
        header = content[:idx]
        tail = content[idx:]
        # Simple string replace of the specific sequence that likely exists
        # It seems the file has: var... { \n };
        # We replace that entire sequence
        pass 
        # Actually, let's just use replace on the string literal we expect
        # If strict match failed, maybe indentation is different?
        # Let's try replacing just the broken line with the fixed line + "};" isn't quite right
        
        # Let's simple search and replace 
        content = content.replace("var thalfVal = {{ pred.result_json.pk.human_thalf_linear }\n                                };", 
                                  "var thalfVal = {{ pred.result_json.pk.human_thalf_linear }};")
        # And CRLF version
        content = content.replace("var thalfVal = {{ pred.result_json.pk.human_thalf_linear }\r\n                                };", 
                                  "var thalfVal = {{ pred.result_json.pk.human_thalf_linear }};")

# Fix 2: 'default (0)' to 'default(0)'
content = content.replace("|default (0)", "|default(0)")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Template fixed.")
