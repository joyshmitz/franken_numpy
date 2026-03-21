import numpy as np

seed = 12345
def fresh(): return np.random.Generator(np.random.PCG64DXSM(seed))

tests = [
    ("random", lambda g: g.random(5)),
    ("uniform", lambda g: g.uniform(0.0, 10.0, 5)),
    ("standard_normal", lambda g: g.standard_normal(5)),
    ("normal", lambda g: g.normal(5.0, 2.0, 5)),
    ("standard_exponential", lambda g: g.standard_exponential(5)),
    ("exponential", lambda g: g.exponential(2.0, 5)),
    ("standard_gamma", lambda g: g.standard_gamma(2.0, 5)),
    ("gamma", lambda g: g.gamma(2.0, 3.0, 5)),
    ("beta", lambda g: g.beta(2.0, 5.0, 5)),
    ("chisquare", lambda g: g.chisquare(3.0, 5)),
    ("lognormal", lambda g: g.lognormal(0.0, 1.0, 5)),
    ("standard_cauchy", lambda g: g.standard_cauchy(5)),
    ("standard_t", lambda g: g.standard_t(3.0, 5)),
    ("f", lambda g: g.f(2.0, 5.0, 5)),
    ("pareto", lambda g: g.pareto(3.0, 5)),
    ("weibull", lambda g: g.weibull(2.0, 5)),
    ("power", lambda g: g.power(2.0, 5)),
    ("laplace", lambda g: g.laplace(0.0, 1.0, 5)),
    ("logistic", lambda g: g.logistic(0.0, 1.0, 5)),
    ("gumbel", lambda g: g.gumbel(0.0, 1.0, 5)),
    ("wald", lambda g: g.wald(1.0, 2.0, 5)),
    ("rayleigh", lambda g: g.rayleigh(1.0, 5)),
    ("triangular", lambda g: g.triangular(-1.0, 0.0, 1.0, 5)),
    ("vonmises", lambda g: g.vonmises(0.0, 1.0, 5)),
]

for name, func in tests:
    arr = func(fresh())
    print(f"        check_seq_placeholder!(\"{name}\",")
    print("            [")
    for i, val in enumerate(arr):
        sep = "," if i < len(arr)-1 else ""
        print(f"                {val:.17e}{sep}")
    print("            ]")
    print("        );")
