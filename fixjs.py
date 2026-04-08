with open("server/app.py", "r", encoding="utf-8") as f:
    text = f.read()

import re
text = re.sub(r"const baseUrl =.*?;", "", text)
text = text.replace("fetch(baseUrl + '/run-interactive-baseline'", "fetch('./run-interactive-baseline'")

with open("server/app.py", "w", encoding="utf-8") as f:
    f.write(text)
