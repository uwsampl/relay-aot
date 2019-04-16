import numpy as np
import tvm
from tvm.relay import Module, GlobalVar, Function
from aot import compile

def double_example():
    # Declare a Relay module.
    mod = Module()

    # Implement the double function.
    x = var('x', shape=())
    double = GlobalVar('double')
    mod[double] = Function([x], x + x)

    # Generate a function which calls double twice.
    x = var('x', shape=())
    f = Function([x], double(double(x)))
    # Compile the function.
    cfunc = compile(f, mod)

    a = tvm.nd.array(np.array(1.5, dtype='float32'))
    print(cfunc(a).asnumpy())
