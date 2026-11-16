"""
scripts/audit_name_merges.py
============================
Audit pairs of hoblis in data/clean/name_mapping.csv that were merged.
Prints:
1. Every pair where names differ by more than a trailing vowel or doubled letter.
2. Checks whether any pairs appear in the same year in the same taluk.
"""

import re
import pandas as pd

def normalize_vowels_and_doubles(s: str) -> str:
    s = s.lower().strip()
    # Remove trailing vowels
    s = re.sub(r'[aeiou]+$', '', s)
    # Collapse doubled consonants: e.g. tt -> t, dd -> d, kk -> k, ll -> l, rr -> r, nn -> n, pp -> p, mm -> m, bb -> b, gg -> g
    s = re.sub(r'([bcdfghjklmnpqrstvwxyz])\1+', r'\1', s)
    return s

def differs_by_more_than_vowel_or_double(raw: str, canonical: str) -> bool:
    # Also strip standard numbering suffixes for comparison
    r_base = re.sub(r'_[0-9a-zA-Z]+$', '', raw).strip().lower()
    c_base = re.sub(r'_[0-9a-zA-Z]+$', '', canonical).strip().lower()
    
    if r_base == c_base:
        return False
        
    norm_r = normalize_vowels_and_doubles(r_base)
    norm_c = normalize_vowels_and_doubles(c_base)
    
    return norm_r != norm_c

def main():
    nm = pd.read_csv('data/clean/name_mapping.csv')
    diffs = nm[nm['hobli_raw'].str.strip() != nm['canonical_hobli'].str.strip()].drop_duplicates(
        subset=['canonical_district', 'canonical_taluk', 'hobli_raw', 'canonical_hobli']
    ).copy()
    
    print("=" * 80)
    print("ALL DISTINCT RAW -> CANONICAL HOBLI MAPPINGS:")
    print("=" * 80)
    for _, r in diffs.iterrows():
        print(f"District: {r['canonical_district']} | Taluk: {r['canonical_taluk']} | Raw: '{r['hobli_raw']}' -> Canonical: '{r['canonical_hobli']}' | Match: {r['match_type']} ({r['similarity']}%)")

    print("\n" + "=" * 80)
    print("PAIRS DIFFERING BY MORE THAN A TRAILING VOWEL OR DOUBLED LETTER:")
    print("=" * 80)
    
    more_diff = []
    for _, r in diffs.iterrows():
        if differs_by_more_than_vowel_or_double(r['hobli_raw'], r['canonical_hobli']):
            more_diff.append(r)
            print(f"District: {r['canonical_district']} | Taluk: {r['canonical_taluk']}")
            print(f"  Raw:       '{r['hobli_raw']}'")
            print(f"  Canonical: '{r['canonical_hobli']}'")
            print(f"  Match Type: {r['match_type']} (Similarity: {r['similarity']}%)")
            print("-" * 60)
            
    if not more_diff:
        print("None found! All raw-to-canonical mappings differ only by suffix, trailing vowel, or doubled consonant.")
    else:
        print(f"Total pairs differing by more than trailing vowel or doubled letter: {len(more_diff)}")

    # Check for same year, same taluk collision
    print("\n" + "=" * 80)
    print("CHECKING FOR SAME-YEAR, SAME-TALUK COLLISION OF MERGED PAIRS:")
    print("=" * 80)
    
    # Check if within any single year and taluk, multiple distinct raw hoblis were mapped to the same canonical hobli
    grouped = nm.groupby(['year', 'canonical_district', 'canonical_taluk', 'canonical_hobli'])['hobli_raw'].nunique()
    collisions = grouped[grouped > 1]
    
    if len(collisions) == 0:
        print("[OK] Zero collisions! No two distinct raw hoblis within the same year and taluk were merged into the same canonical hobli.")
    else:
        print(f"[WARNING] Found {len(collisions)} instances where multiple distinct raw hoblis in the same year/taluk were merged:")
        for (yr, d, t, c_h), cnt in collisions.items():
            raw_names = nm[(nm['year'] == yr) & (nm['canonical_district'] == d) & (nm['canonical_taluk'] == t) & (nm['canonical_hobli'] == c_h)]['hobli_raw'].unique()
            print(f"  Year {yr} | {d} | {t} -> Canonical '{c_h}' merged from distinct raw: {list(raw_names)}")

if __name__ == '__main__':
    main()
