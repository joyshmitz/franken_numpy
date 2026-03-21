import re
import numpy as np

seed = 12345
def fresh(): return np.random.Generator(np.random.PCG64DXSM(seed))

tests = [
    ("random", lambda g: g.random(5)),
    ("uniform(0,10)", lambda g: g.uniform(0.0, 10.0, 5)),
    ("standard_normal", lambda g: g.standard_normal(5)),
    ("normal(5,2)", lambda g: g.normal(5.0, 2.0, 5)),
    ("standard_exponential", lambda g: g.standard_exponential(5)),
    ("exponential(2)", lambda g: g.exponential(2.0, 5)),
    ("standard_gamma(2)", lambda g: g.standard_gamma(2.0, 5)),
    ("gamma(2,3)", lambda g: g.gamma(2.0, 3.0, 5)),
    ("beta(2,5)", lambda g: g.beta(2.0, 5.0, 5)),
    ("chisquare(3)", lambda g: g.chisquare(3.0, 5)),
    ("lognormal(0,1)", lambda g: g.lognormal(0.0, 1.0, 5)),
    ("standard_cauchy", lambda g: g.standard_cauchy(5)),
    ("standard_t(3)", lambda g: g.standard_t(3.0, 5)),
    ("f(2,5)", lambda g: g.f(2.0, 5.0, 5)),
    ("pareto(3)", lambda g: g.pareto(3.0, 5)),
    ("weibull(2)", lambda g: g.weibull(2.0, 5)),
    ("power(2)", lambda g: g.power(2.0, 5)),
    ("laplace(0,1)", lambda g: g.laplace(0.0, 1.0, 5)),
    ("logistic(0,1)", lambda g: g.logistic(0.0, 1.0, 5)),
    ("gumbel(0,1)", lambda g: g.gumbel(0.0, 1.0, 5)),
    ("wald(1,2)", lambda g: g.wald(1.0, 2.0, 5)),
    ("rayleigh(1)", lambda g: g.rayleigh(1.0, 5)),
    ("triangular(-1,0,1)", lambda g: g.triangular(-1.0, 0.0, 1.0, 5)),
    ("vonmises(0,1)", lambda g: g.vonmises(0.0, 1.0, 5)),
]

with open('crates/fnp-conformance/src/lib.rs', 'r') as f:
    code = f.read()

for name, func in tests:
    arr = func(fresh())
    
    # Format floats using .17e to match the precision needed
    fmt_vals = ",\n                ".join(f"{v:.17e}" for v in arr)
    replacement = f"[\n                {fmt_vals}\n            ]"
    
    # We replace the array inside check_seq!("name", ... )
    # regex matches: check_seq!( "name", ... [ ... ] );
    # The array part is \[\s*(?:[-+]?\d*\.\d+[eE][-+]?\d+\s*,\s*)*[-+]?\d*\.\d+[eE][-+]?\d+\s*\]
    pattern = r'(check_seq!\(\s*"' + re.escape(name) + r'".*?)(?:\[[\s\S]*?\])(\s*\))'
    
    def repl(m):
        return m.group(1) + replacement + m.group(2)
        
    new_code, count = re.subn(pattern, repl, code, count=1)
    if count == 0:
        print(f"Failed to replace {name}")
    code = new_code

with open('crates/fnp-conformance/src/lib.rs', 'w') as f:
    f.write(code)
print("Updated continuous tests.")
