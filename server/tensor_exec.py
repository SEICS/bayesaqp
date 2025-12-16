import sys
import numpy as np
import ast

def tensor_exec(expr, context):
    "Evaluate several lines of input, returning the result of the last line"
    # print(f"expr: {expr}")
    # print(f"context: {context}")
    try:
        tree = ast.parse(expr)
        eval_expr = ast.Expression(tree.body[-1].value)
        exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
        exec(compile(exec_expr, 'file', 'exec'), context)
        return eval(compile(eval_expr, 'file', 'eval'), context)#, locals['inference_time']
    except TypeError as e: # sparse 0.17.0 isn't compatible with numpy 2.2.6
        if "Unsupported types" in str(e):
            print(f"DEBUG: Tensor type compatibility error in expression: {expr[:200]}...", file=sys.stderr)
            print(f"DEBUG: Available tensor types in context:", file=sys.stderr)
            for key, value in context.items():
                if hasattr(value, 'dtype') and hasattr(value, 'shape'):
                    print(f"  {key}: {type(value)} dtype={value.dtype} shape={value.shape}", file=sys.stderr)
        raise e