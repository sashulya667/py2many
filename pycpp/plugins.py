import ast
import functools
import io
import math
import os
import random
import sys
from typing import Callable, Dict, List, Tuple, Union

from py2many.analysis import get_id
from py2many.exceptions import AstNotImplementedError
from py2many.tracer import is_list


class CppTranspilerPlugins:
    @staticmethod
    def _container_type_info(arg_node) -> Tuple[str, object]:
        if hasattr(arg_node, "container_type"):
            container_type, element_type = arg_node.container_type
            return str(container_type).strip().lower(), element_type
        return "", None

    @staticmethod
    def _decltype_from_expr(expr: str, default_type: str = "int") -> str:
        if not expr or "decltype(auto)" in expr:
            return default_type
        return f"decltype({expr})"

    def _literal_element_type(self, node) -> str:
        if not hasattr(node, "elts") or not node.elts:
            return "int"
        first = self.visit(node.elts[0])
        return CppTranspilerPlugins._decltype_from_expr(first)

    @staticmethod
    def _map_keys_to_vector(map_expr: str, key_type: str) -> str:
        return (
            "([]() { "
            f"std::vector<{key_type}> __keys; "
            f"for (const auto& __kv : {map_expr}) __keys.push_back(__kv.first); "
            "return __keys; "
            "})()"
        )

    @staticmethod
    def _map_keys_to_set(map_expr: str, key_type: str) -> str:
        return (
            "([]() { "
            f"std::set<{key_type}> __keys; "
            f"for (const auto& __kv : {map_expr}) __keys.insert(__kv.first); "
            "return __keys; "
            "})()"
        )

    def _visit_list(self, node, vargs) -> str:
        self._usings.add("<vector>")
        if not vargs:
            return "std::vector<int>{}"
        arg = vargs[0]
        arg_node = node.args[0]
        if isinstance(arg_node, ast.List):
            element_type = CppTranspilerPlugins._literal_element_type(self, arg_node)
            return f"std::vector<{element_type}>{arg}"
        if isinstance(arg_node, ast.Dict):
            key_type = "int"
            if arg_node.keys:
                key_type = CppTranspilerPlugins._decltype_from_expr(
                    self.visit(arg_node.keys[0])
                )
            return CppTranspilerPlugins._map_keys_to_vector(arg, key_type)
        if isinstance(arg_node, ast.Set):
            element_type = self._literal_element_type(arg_node)
            return f"std::vector<{element_type}>({arg}.begin(), {arg}.end())"
        if is_list(arg_node):
            return f"decltype({arg})({arg}.begin(), {arg}.end())"
        container_type, element_type = CppTranspilerPlugins._container_type_info(
            arg_node
        )
        if container_type in {"list", "set"}:
            return f"std::vector<{element_type}>({arg}.begin(), {arg}.end())"
        if container_type == "dict":
            key_type, _ = element_type
            return CppTranspilerPlugins._map_keys_to_vector(arg, key_type)
        raise AstNotImplementedError(
            "list() conversion requires a known iterable type",
            node,
        )

    def _visit_dict(self, node, vargs) -> str:
        self._usings.add("<map>")
        if not vargs:
            return "std::map<int, int>{}"
        arg = vargs[0]
        arg_node = node.args[0]
        if isinstance(arg_node, ast.Dict):
            return arg
        if isinstance(arg_node, (ast.List, ast.Tuple)):
            if not arg_node.elts:
                return "std::map<int, int>{}"
            kv_pairs = []
            for elt in arg_node.elts:
                if not isinstance(elt, ast.Tuple) or len(elt.elts) != 2:
                    raise AstNotImplementedError(
                        "dict() iterable items must be 2-tuples",
                        node,
                    )
                key_expr = self.visit(elt.elts[0])
                value_expr = self.visit(elt.elts[1])
                kv_pairs.append(f"{{ {key_expr}, {value_expr} }}")
            key_type = CppTranspilerPlugins._decltype_from_expr(
                self.visit(arg_node.elts[0].elts[0])
            )
            value_type = CppTranspilerPlugins._decltype_from_expr(
                self.visit(arg_node.elts[0].elts[1])
            )
            return f"std::map<{key_type}, {value_type}>{{{', '.join(kv_pairs)}}}"
        container_type, _ = CppTranspilerPlugins._container_type_info(arg_node)
        if container_type == "dict":
            return f"decltype({arg})({arg}.begin(), {arg}.end())"
        raise AstNotImplementedError(
            "dict() conversion is only supported for dict/map values for now",
            node,
        )

    def _visit_set(self, node, vargs) -> str:
        self._usings.add("<set>")
        if not vargs:
            return "std::set<int>{}"
        arg = vargs[0]
        arg_node = node.args[0]
        if isinstance(arg_node, ast.Set):
            return arg
        if isinstance(arg_node, ast.List):
            element_type = CppTranspilerPlugins._literal_element_type(self, arg_node)
            return (
                "([]() { "
                f"std::vector<{element_type}> __tmp = {arg}; "
                f"return std::set<{element_type}>(__tmp.begin(), __tmp.end()); "
                "})()"
            )
        if isinstance(arg_node, ast.Dict):
            key_type = "int"
            if arg_node.keys:
                key_type = CppTranspilerPlugins._decltype_from_expr(
                    self.visit(arg_node.keys[0])
                )
            return CppTranspilerPlugins._map_keys_to_set(arg, key_type)
        container_type, element_type = CppTranspilerPlugins._container_type_info(
            arg_node
        )
        if container_type in {"list", "set"}:
            return f"std::set<{element_type}>({arg}.begin(), {arg}.end())"
        if container_type == "dict":
            key_type, _ = element_type
            return CppTranspilerPlugins._map_keys_to_set(arg, key_type)
        raise AstNotImplementedError(
            "set() conversion requires a known iterable type",
            node,
        )

    @staticmethod
    def _visit_math_pi(self, node, value, attr):
        return "3.14159265358979323846"

    @staticmethod
    def visit_ord(node, vargs) -> str:
        if not node.args:
            return "0"
        arg_node = node.args[0]
        if (
            isinstance(arg_node, ast.Constant)
            and isinstance(arg_node.value, str)
            and len(arg_node.value) == 1
        ):
            escaped = arg_node.value.replace("\\", "\\\\").replace("'", "\\'")
            return f"static_cast<int>('{escaped}')"
        return f"static_cast<int>({vargs[0]}[0])"

    def _translate_file(file: str) -> str:
        FILE_MAP: Dict[object, str] = {
            "sys.stdout": "std::cout",
            "sys.stdin": "std::cin",
            "sys.stderr": "std::cerr",
        }
        return FILE_MAP.get(file, file)

    def visit_textio_read(self, node, vargs):
        # TODO
        return None

    def visit_textio_write(self, node, vargs):
        self._usings.add("<iostream>")
        return f"{CppTranspilerPlugins._translate_file(get_id(node.func.value))} << {vargs[0]}"

    def visit_textio_flush(self, node, vargs):
        self._usings.add("<iostream>")
        return (
            f"{CppTranspilerPlugins._translate_file(get_id(node.func.value))}.flush()"
        )

    def visit_range(self, node, vargs: List[str]) -> str:
        self._usings.add("<vector>")
        if len(vargs) == 1:
            start, stop, step = "0", vargs[0], "1"
        elif len(vargs) == 2:
            start, stop, step = vargs[0], vargs[1], "1"
        elif len(vargs) == 3:
            start, stop, step = vargs[0], vargs[1], vargs[2]
        else:
            raise AstNotImplementedError("range() with invalid arity", node)
        return (
            "([](int __start, int __stop, int __step) { "
            "std::vector<int> __range; "
            "for (int __i = __start; "
            "(__step >= 0 ? (__i < __stop) : (__i > __stop)); "
            "__i += __step) { __range.push_back(__i); } "
            "return __range; "
            "})("
            f"static_cast<int>({start}), "
            f"static_cast<int>({stop}), "
            f"static_cast<int>({step})"
            ")"
        )

    def visit_print(self, node, vargs: List[str]) -> str:
        self._usings.add("<iostream>")
        buf = []
        for n in node.args:
            value = self.visit(n)
            if isinstance(n, ast.List) or isinstance(n, ast.Tuple):
                buf.append(
                    "std::cout << {};".format(
                        " << ".join([self.visit(el) for el in n.elts])
                    )
                )
            else:
                buf.append(f"std::cout << {value};")
            buf.append('std::cout << " ";')
        buf.pop()
        return "\n".join(buf) + "\nstd::cout << std::endl;"

    def visit_min_max(self, node, vargs, is_max: bool) -> str:
        min_max = "max" if is_max else "min"
        self._usings.add("<algorithm>")
        if hasattr(node.args[0], "container_type"):
            return f"*std::{min_max}_element({vargs[0]}.begin(), {vargs[0]}.end());"
        else:
            # C++ can't deal with max(1, size_t), but size_t support here has been
            # removed as size_t support here causes other problems
            all_vargs = ", ".join(vargs)
            return f"std::{min_max}({all_vargs})"

    def visit_random(self, node, vargs) -> str:
        self._usings.add("<cstdlib>")
        return "(static_cast<float>(rand()) / static_cast<float>(RAND_MAX))"

    @staticmethod
    def visit_cast(node, vargs, cast_to: str) -> str:
        return f"static_cast<{cast_to}>({vargs[0]})"

    @staticmethod
    def visit_floor(node, vargs) -> str:
        return f"static_cast<int>(floor({vargs[0]}))"


# small one liners are inlined here as lambdas
SMALL_DISPATCH_MAP = {
    "int": functools.partial(CppTranspilerPlugins.visit_cast, cast_to="int"),
    "chr": functools.partial(CppTranspilerPlugins.visit_cast, cast_to="char"),
    "ord": CppTranspilerPlugins.visit_ord,
    "str": lambda n, vargs: f"std::to_string({vargs[0]})" if vargs else '""',
    "bool": lambda n, vargs: f"static_cast<bool>({vargs[0]})" if vargs else "false",
    "len": lambda n, vargs: f"static_cast<int>({vargs[0]}.size())",
    "float": functools.partial(CppTranspilerPlugins.visit_cast, cast_to="float"),
    "floor": CppTranspilerPlugins.visit_floor,
}

SMALL_USINGS_MAP = {
    "floor": "<math.h>",
}

DISPATCH_MAP = {
    "max": functools.partial(CppTranspilerPlugins.visit_min_max, is_max=True),
    "min": functools.partial(CppTranspilerPlugins.visit_min_max, is_max=False),
    "range": CppTranspilerPlugins.visit_range,
    "xrange": CppTranspilerPlugins.visit_range,
    "list": CppTranspilerPlugins._visit_list,
    "dict": CppTranspilerPlugins._visit_dict,
    "set": CppTranspilerPlugins._visit_set,
    "print": CppTranspilerPlugins.visit_print,
}

MODULE_DISPATCH_TABLE = {
    "time": "<chrono>",
    "random": "<cstdlib>",
}

DECORATOR_DISPATCH_TABLE = {}

CLASS_DISPATCH_TABLE = {}


def emit_argv(self, node, value, attr):
    self._usings.add("<string>")
    self._usings.add("<vector>")
    return "std::vector<std::string>(argv, argv + argc)"


ATTR_DISPATCH_TABLE = {
    "math.pi": CppTranspilerPlugins._visit_math_pi,
    "sys.argv": emit_argv,
}

FuncType = Union[Callable, str]

FUNC_DISPATCH_TABLE: Dict[FuncType, Tuple[Callable, bool]] = {
    io.TextIOWrapper.read: (CppTranspilerPlugins.visit_textio_read, True),
    io.TextIOWrapper.write: (CppTranspilerPlugins.visit_textio_write, True),
    io.TextIOWrapper.flush: (CppTranspilerPlugins.visit_textio_flush, True),
    math.asin: (lambda self, node, vargs: f"std::asin({vargs[0]})", False),
    math.acos: (lambda self, node, vargs: f"std::acos({vargs[0]})", False),
    math.cos: (lambda self, node, vargs: f"std::cos({vargs[0]})", False),
    math.atan: (lambda self, node, vargs: f"std::atan({vargs[0]})", False),
    math.pow: (lambda self, node, vargs: f"std::pow({vargs[0]}, {vargs[1]})", False),
    random.random: (CppTranspilerPlugins.visit_random, False),
    random.seed: (
        lambda self, node, vargs: f"srand(static_cast<int>({vargs[0]}))",
        False,
    ),
    os.unlink: (lambda self, node, vargs: f"std::fs::remove_file({vargs[0]})", True),
    sys.exit: (lambda self, node, vargs: f"exit({vargs[0]})", True),
}
