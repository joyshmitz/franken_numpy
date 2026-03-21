import re
import numpy as np

seed = 12345

def fresh():
    return np.random.Generator(np.random.PCG64DXSM(seed))

with open('crates/fnp-conformance/src/lib.rs', 'r') as f:
    code = f.read()

# Find all check_seq! blocks
# check_seq!("name", |g: &mut Generator| g.some_func(args), [ ... ])
pattern = r'check_seq!\(\s*"([^"]+)",\s*\|[^\|]+\|\s*g\.([a-zA-Z0-9_]+)\(([^)]*)\)[^,]*,[\s\S]*?\[([\s\S]*?)\]\s*\);'

matches = list(re.finditer(pattern, code))

for m in matches:
    name = m.group(1)
    func_name = m.group(2)
    args_str = m.group(3).strip()
    
    # Extract args
    # e.g., "0.0, 1.0, 5" -> [0.0, 1.0, 5]
    # "5" -> [5]
    args = []
    if args_str:
        for p in args_str.split(','):
            p = p.strip()
            if p.endswith('.0'):
                args.append(float(p))
            elif '.' in p:
                args.append(float(p))
            elif p.startswith('&['):
                # parse array e.g., &[1.0, 2.0, 3.0]
                p = p.replace('&[', '').replace(']', '')
                arr = [float(x.strip()) for x in p.split(',') if x.strip()]
                args.append(arr)
            elif p.startswith('vec!['):
                p = p.replace('vec![', '').replace(']', '')
                arr = [int(x.strip()) for x in p.split(',') if x.strip()]
                args.append(arr)
            elif p.isdigit():
                args.append(int(p))
            else:
                args.append(p)
                
    g = fresh()
    if func_name == "f_distribution": func_name = "f"
    if func_name == "random": args = [5]
    if func_name == "standard_gamma": args = [args[0], 5]
    
    try:
        func = getattr(g, func_name)
        result = func(*args)
        if isinstance(result, np.ndarray):
            result = result.flatten()
        else:
            result = [result]
            
        # Format replacement
        if result.dtype.kind in 'iu':
            fmt_vals = ",\n                ".join(str(int(v)) for v in result)
        else:
            fmt_vals = ",\n                ".join(f"{v:.17e}" for v in result)
            
        replacement = f"[\n                {fmt_vals}\n            ]"
        
        # Replace only the array part
        old_block = m.group(0)
        # regex to replace the array part
        new_block = re.sub(r'\[[\s\S]*?\]', replacement, old_block, count=1)
        code = code.replace(old_block, new_block)
        print(f"Updated {name}")
    except Exception as e:
        print(f"Skipping {name} due to error: {e}")

with open('crates/fnp-conformance/src/lib.rs', 'w') as f:
    f.write(code)
