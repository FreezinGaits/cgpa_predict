import pandas as pd
import numpy as np
import re

print("Starting deep cleaning of Academic_Score_3.xlsx...")

file_path = "CGPA Project/Academic_Score_3.xlsx"
df_raw = pd.read_excel(file_path)

# ═══════════════════════════════════════════════════════════════════
# SMART EXTRACTORS - Keep as much data as possible, only NaN truly unrecoverable values
# ═══════════════════════════════════════════════════════════════════

def extract_gpa(val):
    """Extract GPA (0-10 scale) from messy strings like '7 SGPA', '7.04/10', '8.5sgpa', '7-8', etc."""
    if pd.isna(val): return np.nan
    s = str(val).strip()
    sl = s.lower()
    
    # Truly unrecoverable - no number at all, or explicitly says result pending/not available
    garbage = ['na', 'n/a', 'n.a', 'n.a.', 'n a', 'nil', 'null', 'none', 'not', 'pending',
               'waiting', 'reappear', 're appear', 're-appear', 'reapear', 'reappeared',
               'reaper', 'failed', 'fail', 'back', 'supply', 'detain', 'yta',
               'result', 'declared', 'released', 'announced', 'unknown', 'don\'t', 
               'didn', 'known', 'awaited', 'good', 'rahul', 'ana', 'cgp na',
               '-', '.', '....', ' -', 'r', 'a', '-na']
    
    # Check if the entire string (stripped and lowered) matches garbage
    if sl in garbage:
        return np.nan
    # Check if it STARTS with a garbage keyword and has no digits
    if not re.search(r'\d', s):
        return np.nan
    # Check for "1st", "1st sem", "first semester", "1 sem", "1year" etc - these are semester numbers, not GPAs
    if re.match(r'^(1st|2nd|3rd|ist|first|im in|it\'s my)', sl):
        return np.nan
    if re.match(r'^[14]\s*(st|sem|year)', sl):
        return np.nan
    
    # Handle "7..6" -> "7.6"
    s_clean = re.sub(r'\.{2,}', '.', s)
    # Handle "8?28" -> "8.28"
    s_clean = re.sub(r'\?', '.', s_clean)
    
    # Try to find all numbers
    nums = re.findall(r'[\d]+\.?[\d]*', s_clean)
    if not nums:
        return np.nan
    
    # If multiple numbers separated by comma/and (like "5.63, 4.0" or "6.84 and 7.10"), take average
    if len(nums) >= 2 and (',' in s or 'and' in sl or '/' not in s):
        valid = [float(n) for n in nums if 0 <= float(n) <= 10]
        if valid:
            return round(np.mean(valid), 2)
    
    v = float(nums[0])
    # "7.04/10" pattern
    if '/' in s and len(nums) >= 2 and float(nums[1]) == 10:
        return round(v, 2) if 0 <= v <= 10 else np.nan
    # If value > 10 and <= 100, likely a percentage, convert
    if v > 10 and v <= 100:
        v = v / 10
    return round(v, 2) if 0 <= v <= 10 else np.nan


def extract_pct(val):
    """Extract percentage (0-100 scale). Handles CGPA->pct conversion."""
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    # Remove % symbols
    s = re.sub(r'[%℅℃\+]', '', s)
    
    # If it says "cgpa" or "sgpa", attempt conversion
    has_cgpa = 'cgpa' in s or 'sgpa' in s
    s_clean = re.sub(r'[a-zA-Z/\s,]+', ' ', s).strip()
    
    nums = re.findall(r'[\d]+\.?[\d]*', s_clean)
    if not nums: return np.nan
    
    # If multiple nums (like ranges), average them
    floats = [float(n) for n in nums]
    v = np.mean(floats)
    
    # If value looks like CGPA (0-10), convert to percentage
    if has_cgpa and v <= 10:
        v = v * 10
    elif v <= 10:
        v = v * 10
    
    return round(v, 2) if 0 <= v <= 100 else np.nan


def extract_score(val):
    """Extract midterm/assignment scores. Keeps numbers from messy strings like '50+', '40 45', '13-18 out of 24'."""
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    
    # Truly garbage
    reject = ["na","nil","null","none","no","fix","good","average","idk","not sure",
              "not showing", "nine"]
    if s in reject or any(s.startswith(r) for r in ["not ", "no "]):
        return np.nan
    if not re.search(r'\d', s):
        return np.nan
    
    # Remove text like "percent", "%", "+"
    s = re.sub(r'percent|%|℅|℃|\+', '', s)
    
    # Handle "out of X" patterns: "13-18 out of 24" -> average(13,18)
    # Handle "20/22" -> 20
    nums = re.findall(r'[\d]+\.?[\d]*', s)
    if not nums: return np.nan
    
    floats = [float(n) for n in nums if float(n) <= 100]
    if not floats: return np.nan
    
    # If "out of" is in the string, take the first number(s) before "out of"
    if 'out of' in s:
        before_out = s.split('out of')[0]
        before_nums = re.findall(r'[\d]+\.?[\d]*', before_out)
        if before_nums:
            floats = [float(n) for n in before_nums]
    
    v = np.mean(floats)
    if v <= 1: v *= 100
    return round(v, 2) if 0 <= v <= 100 else np.nan


def extract_hours(val):
    """Extract study hours. Handles '4-5', '2-3 hours', etc."""
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(r in s for r in ["na","fix","nothing","depends","all day","nil","null","none"]): return np.nan
    if not re.search(r'\d', s): return np.nan
    
    nums = [float(x) for x in re.findall(r'[\d]+\.?[\d]*', s) if float(x) <= 24]
    return round(np.mean(nums), 2) if nums else np.nan


def extract_backlogs(val):
    """Extract number of backlogs. '1 supply' -> 1, 'No' -> 0, 'One supply' -> 1."""
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    
    # Explicit zero
    if any(x in s for x in ["no","nil","none","zero","na","null","nill","0 backlogs","0 back"]): 
        return 0.0
    
    # Word numbers
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for word, num in word_map.items():
        if word in s:
            return float(num)
    
    nums = re.findall(r'[\d]+', s)
    return float(nums[0]) if nums else np.nan


def extract_dist(val):
    """Extract distance in KM."""
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(r in s for r in ["na","nil","null","none","hostel","walk","accommodation","on campus"]):
        return 0.0
    if not re.search(r'\d', s): return np.nan
    
    if "meter" in s:
        nm = re.findall(r'[\d]+\.?[\d]*', s)
        return round(float(nm[0]) / 1000, 2) if nm else np.nan
    nums = [float(x) for x in re.findall(r'[\d]+\.?[\d]*', s) if float(x) < 1000]
    return round(np.mean(nums), 2) if nums else np.nan


def encode_complexity(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if "1" in s or "easy" in s: return 1
    if "2" in s or "medium" in s: return 2
    if "3" in s or "hard" in s: return 3
    return np.nan

def encode_teacher_fb(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if "good" in s and "not" not in s: return 3
    if "confident" in s or "need" in s: return 2
    return 1

def encode_participation(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if "moderator" in s: return 4
    if "shares" in s or "brings" in s or "statistic" in s: return 3
    if "listener" in s: return 2
    if "less active" in s: return 1
    return 2


# ═══════════════════════════════════════════════════════════════════
# APPLY CLEANING
# ═══════════════════════════════════════════════════════════════════
for i, func in [
    (5, extract_gpa),        # Prev Prev GPA  
    (6, extract_gpa),        # Prev Semester GPA
    (7, extract_score),      # Midterm
    (8, extract_pct),        # 12th Pct
    (9, extract_hours),      # Study Hours
    (10, extract_score),     # Assignment
    (11, extract_pct),       # 10th Pct
    (12, extract_pct),       # Attendance
    (13, extract_backlogs),  # Backlogs
    (15, extract_dist),      # Distance
    (16, encode_complexity), 
    (17, encode_teacher_fb), 
    (18, encode_participation)
]:
    col_name = df_raw.columns[i]
    df_raw[col_name] = df_raw[col_name].apply(func).astype(float)

# Stress
col_stress = df_raw.columns[14]
df_raw[col_stress] = df_raw[col_stress].astype(str).str.strip().map(
    {"0": 0.0, "1": 1.0, "0.0": 0.0, "1.0": 1.0}
).astype(float)

# Quick stats
for i in [5,6,7,8,9,10,11,12,13,14,15]:
    col = df_raw.columns[i]
    total = len(df_raw)
    valid = df_raw[col].notna().sum()
    print(f"  Col {i:2d}: {valid:4d}/{total} valid  |  {col[:50]}")

df_raw.to_excel(file_path, index=False)
print("\nSUCCESS: Academic_Score_3.xlsx has been deep-cleaned with smart extractors.")
