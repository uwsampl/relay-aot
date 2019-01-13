# relay-aot

An experimental ahead of time compiler for Relay.

This repository contains an external library which implements ahead of time compilation
for Relay. The current approach is just a proof of concept which lowers Relay to C++,
and relies on a C++ compiler such as `gcc` or `clang` to produce an executable. 

The current library exposes a primitive `compile` which takes a `relay.Function`
and returns a closure which implements the function using native code. The
compiler lowers this function to a small C++-like IR, and then generates
a C++ program from this, which is then compiled and dynamically linked.

We extract the corresponding symbol from the dynamic library, and wrap
it in a Python closure. 
