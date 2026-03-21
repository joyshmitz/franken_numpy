import re
import numpy as np

seed = 12345
def fresh(): return np.random.Generator(np.random.PCG64DXSM(seed))

tests_continuous = [
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

tests_advanced = [
    ("dirichlet", lambda g: g.dirichlet([1.0, 2.0, 3.0], 2).flatten()),
    ("noncentral_chisquare", lambda g: g.noncentral_chisquare(2.0, 3.0, 5)),
    ("noncentral_f", lambda g: g.noncentral_f(2.0, 3.0, 4.0, 5)),
]

tests_remaining = [
    ("multinomial", lambda g: g.multinomial(10, [0.2, 0.3, 0.5], 2).flatten()),
    ("zipf", lambda g: g.zipf(2.0, 5)),
    ("hypergeometric", lambda g: g.hypergeometric(10, 20, 5, 5)),
    ("multivariate_hypergeometric", lambda g: g.multivariate_hypergeometric([10, 20, 30], 5, 2).flatten()),
]

with open('crates/fnp-conformance/src/lib.rs', 'r') as f:
    code = f.read()

for name, func in tests_continuous + tests_advanced + tests_remaining:
    arr = func(fresh())
    
    fmt_vals = ",\n                ".join(f"{v:.17e}" for v in arr)
    replacement = f"[\n                {fmt_vals}\n            ]"
    
    # We replace the array inside check_seq!("name", ... )
    # using a very precise string matching or regex
    # The current code looks like:
    #         check_seq!(
    #             "random",
    #             |g: &mut Generator| g.random(5),
    #             [
    #                 9.32081690319876310e-1,
    #                 ...
    #             ]
    #         );
    
    # regex matches: "name",\n            ... [ ... ]
    pattern = r'("' + re.escape(name) + r'",\n[\s\S]*?)(?:\[\s*(?:[-+]?\d*\.\d+[eE][-+]?\d+\s*,\s*)*[-+]?\d*\.\d+[eE][-+]?\d+\s*\])(\n\s*\))'
    
    def repl(m):
        return m.group(1) + replacement + m.group(2)
        
    new_code, count = re.subn(pattern, repl, code, count=1)
    
    # for ints
    if count == 0:
        fmt_vals = ", ".join(f"{int(v)}" for v in arr)
        replacement_int = f"[{fmt_vals}]"
        # some use [ 4, 2, ... ] format on a single line or multiline
        pattern_int = r'("' + re.escape(name) + r'",\n[\s\S]*?)(?:\[[\d\s,]+\])(\n\s*\))'
        def repl_int(m):
            return m.group(1) + replacement_int + m.group(2)
        new_code, count = re.subn(pattern_int, repl_int, code, count=1)
        
    if count == 0:
        print(f"Failed to replace {name}")
    code = new_code

with open('crates/fnp-conformance/src/lib.rs', 'w') as f:
    f.write(code)
print("Updated all tests.")
