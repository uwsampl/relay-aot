from . import little_cpp
from tvm import relay
from tvm.relay import _module
from tvm.relay.prelude import Prelude

class ExprWithStmt:
    def __init__(self, expr, stmt=""):
        assert isinstance(expr, str)
        assert isinstance(stmt, str)
        assert "ExprWithStmt" not in expr
        assert "ExprWithStmt" not in stmt
        self.expr = expr
        self.stmt = stmt

    def __str__(self):
        return f"ExprWithStmt({self.expr}, {self.stmt})"

    def __repr__(self):
        return self.__str__()

class ToSource:
    def __init__(self, gv_map):
        self.gv_map = gv_map
        self.name_counter = 0
        self.source_content = ""
        self.name_map = {}
        self.local = True
        self.declare = ""
        self.declare_map = {}
        self.input_const = []

    def fresh_global_name(self):
        name = f"global{self.name_counter}"
        self.name_counter += 1
        return name

    def fresh_local_name(self, var=None):
        if var is not None:
            name = f"local_{var.name_hint}_{self.name_counter}"
        else:
            name = f"local_{self.name_counter}"
        self.name_counter += 1
        return name

    def fresh_label_name(self):
        name = f"label_{self.name_counter}"
        self.name_counter += 1
        return name

    # return (str, str) with lhs being stmts, and rhs being expression
    def visit(self, node, local=True, name=None):
        if isinstance(node, little_cpp.PackedCall):
            res = self.visit_packed_call(node)
        elif isinstance(node, little_cpp.CPPFunction):
            res = self.visit_cpp_function(node, local, name)
        elif isinstance(node, little_cpp.Decl):
            res = self.visit_decl(node)
        elif isinstance(node, little_cpp.Invoke):
            res = self.visit_invoke(node)
        elif isinstance(node, relay.Var):
            res = ExprWithStmt(self.name_map[node])
        elif isinstance(node, relay.GlobalVar):
            res = self.visit_global_var(node)
        elif isinstance(node, relay.Constant):
            res = self.visit_constant(node)
        elif isinstance(node, little_cpp.CPPIf):
            res = self.visit_if(node)
        elif isinstance(node, little_cpp.CPPTuple):
            res = self.visit_tuple(node)
        elif isinstance(node, little_cpp.CPPConstructor):
            res = self.visit_constructor(node)
        elif isinstance(node, little_cpp.CPPMatch):
            res = self.visit_match(node)
        elif isinstance(node, little_cpp.CPPTupleGetItem):
            res = self.visit_tuple_getitem(node)
        elif isinstance(node, little_cpp.CPPRefCreate):
            res = self.visit_ref_create(node)
        elif isinstance(node, little_cpp.CPPRefRead):
            res = self.visit_ref_read(node)
        elif isinstance(node, little_cpp.CPPRefWrite):
            res = self.visit_ref_write(node)
        else:
            raise Exception(str(node))
        assert isinstance(res, ExprWithStmt)
        return res

    def visit_ref_create(self, node):
        vv = self.visit(node.value)
        return ExprWithStmt(f"RefValueNode::make({vv.expr})", vv.stmt)

    def visit_ref_read(self, node):
        vr = self.visit(node.ref)
        e = f"{vr.expr}->value"
        return ExprWithStmt(f"{self.downcast(e, self.visit_type(node.relay_type))}", vr.stmt)

    def visit_ref_write(self, node):
        vr = self.visit(node.ref)
        vv = self.visit(node.value)
        stmt = vr.stmt + vv.stmt + f"{vr.expr}->value={vv.expr};\n"
        return ExprWithStmt("TupleValueNode::make({})", stmt)

    def visit_tuple_getitem(self, node):
        vt = self.visit(node.tuple_value)
        e = f"{vt.expr}->fields[{node.index}]"
        return ExprWithStmt(f"{self.downcast(e, self.visit_type(node.relay_type))}", vt.stmt)

    def visit_constructor(self, node):
        args_str, stmt_str = self.visit_args(node.fields)
        return ExprWithStmt(f"TagToCV({node.tag}, {{{args_str}}})")

    def pattern_var(self, pat, var_set):
        if isinstance(pat, relay.PatternConstructor):
            for x in pat.patterns:
                self.pattern_var(x, var_set)
        elif isinstance(pat, relay.PatternVar):
            assert pat.var not in var_set
            var_set.add(pat.var)
        else:
            raise Exception(str(pat))

    def downcast(self, expr_str, ty_str):
        return f"Downcast<{ty_str}>({expr_str})"

    def visit_match(self, node):
        vd = self.visit(node.data)
        stmt_str = vd.stmt

        pattern_var_set = set()
        for c in node.clause:
            self.pattern_var(c[0], pattern_var_set)

        for v in pattern_var_set:
            bind_name = self.fresh_local_name()
            self.name_map[v] = bind_name
            stmt_str += f"{self.visit_type(v.checked_type)} {bind_name};\n"

        # match data_name to pat, and fill the var accordingly.
        # go to fail_label or ok_label base on failure/success.
        def visit_pattern(pat, data_name, fail_label, ok_label):
            if isinstance(pat, relay.PatternConstructor):
                ok_case = ""
                bind_names = []
                assert len(pat.constructor.inputs) == len(pat.patterns)
                for i, input_type in enumerate(pat.constructor.inputs):
                    bind_name = self.fresh_local_name()
                    bind_names.append(bind_name)
                    t = self.visit_type(input_type)
                    e = f"{data_name}->fields[{i}]"
                    ok_case += f"{t} {bind_name} = {self.downcast(e, t)};\n"
                for bind_name, p in zip(bind_names, pat.patterns):
                    next_label = self.fresh_label_name()
                    ok_case += visit_pattern(p, bind_name, fail_label, next_label)
                    ok_case += f"{next_label}:\n"
                ok_case += f"goto {ok_label};"
                return f"""
                CHECK({data_name}->constructor->tag != -1);
                if ({data_name}->constructor->tag == {pat.constructor.tag}) {{
                  {ok_case}
                }} else {{
                  goto {fail_label};
                }}
                """
            elif isinstance(pat, relay.PatternVar):
                return f"""
                {self.name_map[pat.var]} = {data_name};
                """
            else:
                raise Exception(str(pat))

        in_name = self.fresh_local_name()
        out_name = self.fresh_local_name()
        stmt_str += f"ConstructorValue {in_name} = {vd.expr};\n"
        stmt_str += f"{self.visit_type(node.relay_type)} {out_name};\n"
        match_finish_label = self.fresh_label_name()
        for c in node.clause:
            vc = self.visit(c[1])
            fail_label = self.fresh_label_name()
            ok_label = self.fresh_label_name()
            stmt_str += f"""{{
              {visit_pattern(c[0], in_name, fail_label, ok_label)}
            }}
            """
            stmt_str += f"""{{
              {ok_label}:
              {vc.stmt}
              {out_name} = {vc.expr};
              goto {match_finish_label};
            }}
            """
            stmt_str += f"{fail_label}:\n"
        stmt_str += """CHECK(false) << "does not match any";\n"""
        stmt_str += f"{match_finish_label}: ;"
        return ExprWithStmt(out_name, stmt_str)

    def visit_tuple(self, node):
        expr = []
        stmt_str = ""
        for x in node.fields:
            vx = self.visit(x)
            expr.append(vx.expr)
            stmt_str += vx.stmt
        return ExprWithStmt(f"TupleValueNode::make({{{inter(expr)}}})", stmt_str)

    def visit_if(self, node):
        vc = self.visit(node.cond)
        vt = self.visit(node.true_branch)
        vf = self.visit(node.false_branch)
        ret_name = self.fresh_local_name()
        stmt = f"{self.visit_type(node.relay_type)} {ret_name};"
        stmt += f"""
        {vc.stmt}
        if (NDToBool({vc.expr}->data)) {{
          {vt.stmt}
          {ret_name} = {vt.expr};
        }} else {{
          {vf.stmt}
          {ret_name} = {vf.expr};
        }}
        """
        return ExprWithStmt(ret_name, stmt)

    def visit_type(self, node):
        if isinstance(node, relay.TensorType):
            res = "TensorValue"
        elif isinstance(node, relay.TupleType):
            res = "TupleValue"
        elif isinstance(node, relay.TypeCall):
            res = "ConstructorValue" # typecall is only used at constructors at the moment
        elif isinstance(node, relay.TypeVar):
            res = "TensorValue"
        else:
            raise Exception(str(node))
        return res

    def visit_constant(self, const):
        if const not in self.declare_map:
            name = self.fresh_global_name()
            self.declare_map[const] = name
            self.declare += f"TensorValue {name};\n"
            self.input_const.append((name, const.data.asnumpy()))
        return ExprWithStmt(self.declare_map[const])

    def visit_global_var(self, gv):
        if gv not in self.declare_map:
            name = self.fresh_global_name()
            self.declare_map[gv] = name
            vgv = self.visit(self.gv_map[gv], local=False, name=name)
            assert vgv.stmt == ""
            assert vgv.expr == name
        return ExprWithStmt(self.declare_map[gv])

    def visit_args(self, args):
        args_str = ""
        stmt_str = ""
        for i, arg in enumerate(args):
            assert isinstance(arg, relay.Var)
            va = self.visit(arg)
            args_str += va.expr
            stmt_str += va.stmt
            if i != len(args) - 1:
                args_str += ", "
        return args_str, stmt_str

    def visit_invoke(self, invoke):
        args_str, stmt_str = self.visit_args(invoke.args)
        func = self.visit(invoke.call)
        return ExprWithStmt(f"{func.expr}({args_str})", stmt_str + func.stmt)

    def visit_decl(self, decl):
        source = ""
        for var, value in decl.bindings:
            local_name = self.fresh_local_name(var)
            self.name_map[var] = local_name
            vv = self.visit(value)
            source += vv.stmt
            source += f"auto {local_name} = {vv.expr};\n"
        vb = self.visit(decl.body)
        source += vb.stmt
        return ExprWithStmt(vb.expr, source)

    def nd_dtype(self, tt):
        assert isinstance(tt, relay.ty.TensorType)
        if tt.dtype == 'int32':
            return 'dtype_i32'
        elif tt.dtype == 'float32':
            return 'dtype_f32'
        elif tt.dtype == 'bool':
            return 'dtype_u1'
        raise Exception("unknown tensor dtype: " + str(tt))

    def nd_shape(self, tt):
        return f"{{{inter([str(s) for s in tt.shape])}}}"

    def visit_packed_call(self, call):
        decl_str = ""
        args_str = ""
        if call.args_is_tuple:
            assert len(call.args) == 1
            va = self.visit(call.args[0])
            decl_str += va.stmt
            tuple_name = self.fresh_local_name();
            decl_str += f"TupleValue {tuple_name} = {va.expr};\n"
            end = call.arity - 2
            for i in range(end + 1):
                args_str += f"ValueToND({tuple_name}->fields[{i}])"
                if i != end:
                    args_str += ", "
        else:
            end = call.arity - 2
            for i, arg in enumerate(call.args):
                va = self.visit(arg)
                decl_str += va.stmt
                args_str += f"{va.expr}->data"
                if i != end:
                    args_str += ", "

        out_name = self.fresh_local_name()
        return ExprWithStmt(out_name, f"""
            {decl_str}
            const PackedFunc *pf = runtime::Registry::Get("{call.name}");
            CHECK(pf);
            TensorValue {out_name} = TensorValueNode::make(NDArray::Empty({self.nd_shape(call.output_type)}, {self.nd_dtype(call.output_type)}, context));
            (*pf)({args_str}, {out_name}->data);
        """)

    def visit_cpp_function(self, func, local, name):
        param_str = ""

        end = len(func.params) - 1
        for i, param in enumerate(func.params):
            pname = self.fresh_local_name()
            self.name_map[param] = pname
            param_str += f"const {self.visit_type(param.type_annotation)}& {pname}"
            if i != end:
                param_str += ", "

        vb = self.visit(func.body)
        body = vb.stmt + f"""return {vb.expr};"""

        if local:
            return ExprWithStmt(f"""[=]({param_str}) {{
                {body}
            }}
            """)
        else:
            if name is None:
                name = self.fresh_global_name()
            self.declare += f"""
            {self.visit_type(func.ret_type)} {name}({param_str}) {{
                {body}
            }}
            """
            return ExprWithStmt(name)

    def mk_register_api(self, name: str, func) -> str:
        vf = self.visit(func, False)
        assert vf.stmt == ""
        source = self.declare

        args = ""
        if isinstance(func, relay.GlobalVar):
            func = self.gv_map[func]
        end = len(func.params) - 1
        init = ""
        for i, (input_name, _) in enumerate(self.input_const):
            init += f"{input_name} = args[{i}];\n"
        for i in range(len(func.params)):
            args += f"args[{i+len(self.input_const)}]"
            if i != end:
                args += ", "

        source += f"""
        TVM_REGISTER_API("{name}")
        .set_body([](TVMArgs args, TVMRetValue* ret) {{
            {init}
            *ret = {vf.expr}({args});
        }});
        """
        # print(source)
        return source

def inter(strs, sep=", "):
    ret = ""
    for i in range(len(strs)):
        ret += strs[i]
        if i != len(strs) - 1:
            ret += sep
    return ret

def mk_file(body, use_gpu):
    device_type = "DLDeviceType::kDLGPU" if use_gpu else "DLDeviceType::kDLCPU"
    return f"""
    #include <tvm/tvm.h>
    #include <tvm/api_registry.h>
    #include <tvm/relay/interpreter.h>
    #include <iostream>

    using namespace tvm;
    using namespace runtime;
    using namespace relay;

    static DLDataType dtype_f32 = DLDataType {{ .code = DLDataTypeCode::kDLFloat, .bits = 32, .lanes = 1 }};
    static DLDataType dtype_u32 = DLDataType {{ .code = DLDataTypeCode::kDLUInt, .bits = 32, .lanes = 1 }};
    static DLDataType dtype_u1 = DLDataType {{ .code = DLDataTypeCode::kDLUInt, .bits = 1, .lanes = 1 }};
    static DLDataType dtype_i32 = DLDataType {{ .code = DLDataTypeCode::kDLInt, .bits = 32, .lanes = 1 }};
    static DLContext context = DLContext {{ .device_type = {device_type}, .device_id = 0 }};
    bool NDToBool(const NDArray& nd) {{
      DLContext cpu_ctx;
      cpu_ctx.device_type = kDLCPU;
      cpu_ctx.device_id = 0;
      NDArray cpu_array = nd.CopyTo(cpu_ctx);
      CHECK_EQ(TVMType2Type(cpu_array->dtype), Bool());
      return reinterpret_cast<uint8_t*>(cpu_array->data)[0];
    }}
    NDArray ValueToND(const Value& v) {{
      const TensorValueNode* tv = v.as<TensorValueNode>();
      CHECK(tv);
      return tv->data;
    }}
    ConstructorValue TagToCV(size_t tag, const tvm::Array<Value>& fields) {{
      NodePtr<ConstructorValueNode> n = make_node<ConstructorValueNode>();
      NodePtr<ConstructorNode> con = make_node<ConstructorNode>();
      con->tag = tag;
      n->constructor = Constructor(con);
      n->fields = fields;
      return ConstructorValue(n);
    }}
    {body}
    """

def to_source(mod, gv_map, use_gpu, name, program) -> str:
    convert = ToSource(gv_map)
    ret = mk_file(convert.mk_register_api(name, program), use_gpu)
    return [value for name, value in convert.input_const], ret
