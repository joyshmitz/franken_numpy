import re

with open('crates/fnp-conformance/src/lib.rs', 'r') as f:
    code = f.read()

replacements = {
    "maxwell(1)": [1.9414318048529495e0, 7.990904302922154e-1, 1.2452585825841247e0, 1.2419575657398367e0, 7.271346312135125e-1],
    "halfnormal(1)": [6.48970595720087e-1, 1.089083499221345e0, 1.4703372914093853e0, 6.605483574733667e-1, 4.4234454827889136e-1],
    "lomax(3)": [1.450968390943416e0, 1.471122819734889e-1, 8.49492216360963e-2, 1.5602225624223753e-1, 3.0505749148711736e-1],
    "levy(0,1)": [2.374378551504659e0, 8.430971939670991e-1, 4.625578499776955e-1, 2.2918741489980268e0, 5.110679444934242e0],
    "dirichlet": [1.8801570709043375e-1, 3.8448038254055683e-1, 4.2750391036900925e-1, 1.4347162759248622e-1, 4.7960769869466385e-1, 3.7692067371285e-1],
    "noncentral_chisquare": [1.7575045860311849e0, 3.6008506631904673e0, 3.510963795806461e0, 2.3382764678059766e0, 2.777246991840461e1],
    "noncentral_f": [1.748051450280653e0, 5.010835226209491e-1, 5.105758010259412e0, 8.850261254928487e0, 3.1933776351729932e0],
    "zipf": [14.0, 1.0, 8.0, 1.0, 1.0]
}

int_replacements = {
    "multinomial": [4, 2, 4, 1, 3, 6],
    "hypergeometric": [2, 1, 1, 2, 2],
    "multivariate_hypergeometric": [1, 1, 3, 0, 3, 2]
}

for name, vals in replacements.items():
    fmt_vals = ",\n                ".join(f"{v:.17e}" for v in vals)
    replacement = f"[\n                {fmt_vals}\n            ]"
    pattern = r'("' + re.escape(name) + r'",\n[\s\S]*?)(?:\[\s*(?:[-+]?\d*\.\d+[eE][-+]?\d+\s*,\s*)*[-+]?\d*\.\d+[eE][-+]?\d+\s*\])(\n\s*\))'
    def repl(m): return m.group(1) + replacement + m.group(2)
    code, count = re.subn(pattern, repl, code, count=1)
    if count == 0: print(f"Failed {name}")

for name, vals in int_replacements.items():
    fmt_vals = ", ".join(str(v) for v in vals)
    replacement = f"[{fmt_vals}]"
    pattern = r'("' + re.escape(name) + r'",\n[\s\S]*?)(?:\[[\d\s,]+\])(\n\s*\))'
    def repl(m): return m.group(1) + replacement + m.group(2)
    code, count = re.subn(pattern, repl, code, count=1)
    if count == 0: print(f"Failed {name}")

with open('crates/fnp-conformance/src/lib.rs', 'w') as f:
    f.write(code)
print("done")
