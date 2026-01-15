from ftfy import fix_text

# 1. Read the broken file as raw bytes (DO NOT let Python decode it wrongly)
with open("handwritten_constants.py", "rb") as f:
    raw = f.read()

# 2. Decode using Latin-1 so we preserve the byte values
broken_text = raw.decode("latin-1")

# 3. Fix the mojibake
fixed_text = fix_text(broken_text)

# 4. Write the fixed Sinhala text properly in UTF-8
with open("handwritten_constants.py", "w", encoding="utf-8") as f:
    f.write(fixed_text)

print("✔ Sinhala labels repaired and saved as handwritten_constants.py")
