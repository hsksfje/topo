"""Execute a validated, vectorized Python subset in a resource-limited child.

Only NumPy calls from an explicit allowlist are accessible. No IO, imports
other than numpy, arbitrary attributes, loops, comprehensions or builtins.
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

FUNCTIONS = ('arange','linspace','meshgrid','array','ones','zeros','full','clip',
             'minimum','maximum','sqrt','exp','sin','cos','abs','where','tanh',
             'logical_and','logical_or','power')
ALLOWED = (ast.Module,ast.Import,ast.alias,ast.FunctionDef,ast.arguments,ast.arg,
           ast.Assign,ast.Return,ast.Expr,ast.Constant,ast.Name,ast.Load,ast.Store,
           ast.BinOp,ast.UnaryOp,ast.Compare,ast.Call,ast.Attribute,ast.keyword,
           ast.List,ast.Tuple,ast.Subscript,ast.Slice,ast.Add,ast.Sub,ast.Mult,
           ast.Div,ast.Pow,ast.Mod,ast.USub,ast.UAdd,ast.Lt,ast.LtE,ast.Gt,
           ast.GtE,ast.Eq,ast.NotEq,ast.BitAnd,ast.BitOr,ast.Invert)


def validate(source):
    if not isinstance(source,str) or len(source)>24000:
        raise ValueError('Python source exceeds limit')
    tree=ast.parse(source)
    if len(list(ast.walk(tree)))>4000:
        raise ValueError('Python AST exceeds limit')
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1 or functions[0].name!='initialize':
        raise ValueError('Exactly one initialize function required')
    fun=functions[0]
    if [a.arg for a in fun.args.args]!=['nx','ny','volfrac'] or fun.args.defaults or fun.decorator_list or fun.args.vararg or fun.args.kwarg or fun.args.kwonlyargs or fun.args.posonlyargs or fun.returns:
        raise ValueError('Required signature: initialize(nx, ny, volfrac)')
    for n in tree.body:
        if isinstance(n,ast.Import):
            if len(n.names)!=1 or n.names[0].name!='numpy' or n.names[0].asname!='np':
                raise ValueError('Only import numpy as np permitted')
        elif n is not fun and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)):
            raise ValueError('No executable top-level statements')
    for n in ast.walk(tree):
        if not isinstance(n,ALLOWED):raise ValueError('Forbidden Python syntax: '+type(n).__name__)
        if isinstance(n,ast.Name) and (n.id.startswith('_') or n.id in ('exec','eval','open','compile','getattr','setattr')):
            raise ValueError('Forbidden identifier')
        if isinstance(n,ast.Attribute) and not (isinstance(n.value,ast.Name) and n.value.id=='np' and n.attr in FUNCTIONS+('pi',)):
            raise ValueError('Only allowlisted np attributes permitted')
        if isinstance(n,ast.Call) and not (isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='np' and n.func.attr in FUNCTIONS):
            raise ValueError('Only allowlisted NumPy calls permitted')
        if isinstance(n,ast.Assign) and any(not isinstance(t,(ast.Name,ast.Tuple,ast.List)) for t in n.targets):
            raise ValueError('No attribute or indexed assignment')
        if isinstance(n,ast.arg) and n.annotation is not None:raise ValueError('No annotations')
    # Removing import avoids exposing Python import machinery to generated code.
    tree.body=[n for n in tree.body if not isinstance(n,ast.Import)]
    return tree


def execute(source,nx,ny,volfrac,folder):
    validate(source)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'initializer.py';path.write_text(source, encoding='utf-8')
    payload=json.dumps(dict(source=source,nx=nx,ny=ny,volfrac=volfrac))
    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child'],input=payload,
                         text=True,capture_output=True,timeout=20)
    (folder/'execution.json').write_text(json.dumps(dict(returncode=child.returncode,stderr=child.stderr[-3000:]),indent=2), encoding='utf-8')
    if child.returncode:raise ValueError('Generated code failed: '+child.stderr[-600:])
    import numpy as np
    values=np.asarray(json.loads(child.stdout),dtype=float)
    if values.shape!=(ny,nx) or not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):
        raise ValueError('Code must return finite ny x nx matrix in [0,1]')
    return values


if __name__=='__main__':
    from child_limits import install_limits
    import types
    import numpy as np
    payload=json.loads(sys.stdin.read())
    tree=validate(payload['source'])
    install_limits()
    namespace={'__builtins__':{},'np':types.SimpleNamespace(**{k:getattr(np,k) for k in FUNCTIONS+('pi',)})}
    exec(compile(tree,'llm_initializer.py','exec'),namespace)
    result=np.asarray(namespace['initialize'](payload['nx'],payload['ny'],payload['volfrac']),dtype=float)
    if result.shape!=(payload['ny'],payload['nx']) or not np.isfinite(result).all() or np.any(result<0) or np.any(result>1):
        raise ValueError('Invalid generated density matrix')
    sys.stdout.write(json.dumps(result.tolist(),allow_nan=False))
