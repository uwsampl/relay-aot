from tvm import relay
from tvm.relay import var, Function, op, Module, GlobalVar
import numpy as np
import tvm
import aot

def compile(mod, f):
    return aot.compile(mod, f, ctx=tvm.cpu(0), tgt=tvm.target.create('llvm'), use_gpu=False)

def test_identity():
    mod = Module()
    x = var('x', shape=())
    func = Function([x], x)
    cfunc = compile(mod, func)
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), a.asnumpy())

def test_add():
    mod = Module()
    x = var('x', shape=())
    y = var('y', shape=())
    z = x + y
    func = Function([x, y], z)
    cfunc = compile(mod, func)
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    b = tvm.nd.array(np.array(1.0, dtype='float32'))
    c = tvm.nd.array(np.array(2.0, dtype='float32'))
    output = cfunc(a, b)
    np.testing.assert_allclose(output.asnumpy(), c.asnumpy())

def test_mult_op():
    mod = Module()
    x = var('x', shape=())
    y = var('y', shape=())
    z = x + y
    zz = op.exp(z)
    func = Function([x, y], zz)
    cfunc = compile(mod, func)
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    b = tvm.nd.array(np.array(1.0, dtype='float32'))
    output = cfunc(a, b)
    np.testing.assert_allclose(output.asnumpy(), np.exp(a.asnumpy() + b.asnumpy()))

def test_double():
    mod = Module()
    x = var('x', shape=())
    double = GlobalVar('double')
    mod[double] = Function([x], x + x)
    x = var('x', shape=())
    cfunc = compile(mod, Function([x], double(double(x))))
    a = tvm.nd.array(np.array(1.5, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), np.array(6.0, dtype='float32'))

def test_42():
    mod = Module()
    func = Function([], relay.const(42.0, 'float32'))
    cfunc = compile(mod, func)
    output = cfunc()
    np.testing.assert_allclose(output.asnumpy(), np.array(42.0, dtype='float32'))

def test_add_42():
    mod = Module()
    x = var('x', shape=())
    func = Function([x], x + relay.const(42.0, 'float32'))
    cfunc = compile(mod, func)
    a = tvm.nd.array(np.array(42.0, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), np.array(84.0, dtype='float32'))

def test_int_mult_3():
    mod = Module()
    x = var('x', dtype='int32', shape=())
    func = Function([x], x * relay.const(3, 'int32'))
    cfunc = compile(mod, func)
    a = tvm.nd.array(np.array(4, dtype='int32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), np.array(12, dtype='int32'))

def test_abs():
    mod = Module()
    x = var('x', shape=())
    func = Function([x], relay.If(op.less(x, relay.const(0.0, 'float32')), relay.const(-1.0, 'float32') * x, x))
    cfunc = compile(mod, func)
    a = tvm.nd.array(np.array(12.0, dtype='float32'))
    output = cfunc(a)
    np.testingo.assert_allclose(output.asnumpy(), np.array(12.0, dtype='float32'))
    a = tvm.nd.array(np.array(-34.0, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), np.array(34.0, dtype='float32'))

def test_recur_sum_global():
    mod = Module()
    x = var('x', dtype='int32', shape=())
    sum = GlobalVar('sum')
    c = relay.const(0, 'int32')
    mod[sum] = Function([x],
                        relay.If(op.less(x, c), c, x + sum(x - relay.const(1, 'int32'))),
                        relay.TensorType(dtype='int32', shape=()))
    cfunc = compile(mod, Function([], sum(relay.const(10, 'int32'))))
    output = cfunc()
    np.testing.assert_allclose(output.asnumpy(), np.array(55, dtype='int32'))

if __name__ == "__main__":
    test_identity()
    test_add()
    raise
    test_mult_op()
    test_double()
    test_42()
    test_add_42()
    test_int_mult_3()
    test_abs()
    test_recur_sum_global()
