import textwrap

import pytest

try:
    from py2many.exceptions import AstNotImplementedError
    from py2many.pycpp.transpiler import transpile
except ImportError:
    from py2many.exceptions import AstNotImplementedError
    from pycpp.transpiler import transpile


def parse(*args):
    return "\n".join(args)


def test_declare():
    source = parse("x = 3")
    cpp = transpile(source)
    assert cpp == "auto x = 3;"


def test_empty_return():
    source = parse("def foo():", "   return")
    cpp = transpile(source)
    expected = """\
    inline void foo() {
    return;}
    """
    assert cpp == textwrap.dedent(expected)


def test_print_multiple_vars():
    source = parse('print(("hi", "there" ))')
    cpp = transpile(source)
    assert cpp == parse(
        'std::cout << std::string{"hi"} << std::string{"there"};',
        "std::cout << std::endl;",
    )


def test_assert():
    source = parse("assert 1 == foo(3)")
    cpp = transpile(source, testing=True)
    assert cpp == "#include <catch2/catch_test_macros.hpp>\nREQUIRE(1 == foo(3));"


def test_augmented_assigns_with_counter():
    source = parse(
        "counter = 0", "counter += 5", "counter -= 2", "counter *= 2", "counter /= 3"
    )
    cpp = transpile(source)
    assert cpp == parse(
        "auto counter = 0;",
        "counter += 5;",
        "counter -= 2;",
        "counter *= 2;",
        "counter /= 3;",
    )


def test_declare_var_before_if_else_statements():
    source = parse("if 10:", "   x = True", "else:", "   x = False", "y = x")
    cpp = transpile(source)
    assert cpp == parse(
        "decltype(true) x;",
        "if(10) {",
        "x = true;",
        "} else {",
        "x = false;",
        "}",
        "auto y = x;",
    )


def test_declare_vars_inside_if_as_long_as_possible():
    source = parse("x = 5", "if 10:", "   y = 10", "   x *= y")
    cpp = transpile(source)
    assert cpp == parse("auto x = 5;", "if(10) {", "auto y = 10;", "x *= y;", "}")


def test_print_program_args():
    source = parse(
        'if __name__ == "__main__":', "    for arg in sys.argv:", "       print(arg)"
    )
    cpp = transpile(source)
    # Note the args and return type are missing here as this `transpile` wrapper
    # is not the main py2many wrapper, and notably doesnt use PythonMainRewriter.
    assert cpp == parse(
        "void main() {",
        "for(auto arg : std::vector<std::string>(argv, argv + argc)) {",
        "std::cout << arg;",
        "std::cout << std::endl;",
        "}}",
        "",
    )


def test_tuple_swap():
    source = parse("x = 3", "y = 1", "x, y = y, x")
    cpp = transpile(source)
    assert cpp == parse(
        "auto x = 3;", "auto y = 1;", "std::tie(x, y) = std::make_tuple(y, x);"
    )


def test_assign():
    source = parse("x = 3", "x = 1")
    cpp = transpile(source)
    assert cpp == parse("auto x = 3;", "x = 1;")


def test_function_with_return():
    source = parse("def fun(x):", "   return x")
    cpp = transpile(source)
    expected = """\
        template <typename T0>auto fun(T0 x) {
        return x;}
    """
    assert cpp == textwrap.dedent(expected)


def test_void_function():
    source = parse("def test_fun():", "   assert True")
    cpp = transpile(source, testing=False)
    expected = """\
        inline void test_fun() {
        assert(true);}
    """
    assert cpp == textwrap.dedent(expected)


def test_create_catch_test_case():
    source = parse("def test_fun():", "   assert True")
    cpp = transpile(source, testing=True)
    assert cpp == parse(
        "#include <catch2/catch_test_macros.hpp>",
        'TEST_CASE("test_fun") {',
        "REQUIRE(true);",
        "}",
    )


def test_list_as_vector():
    source = parse("values = [0, 1, 2, 3]")
    cpp = transpile(source)
    assert cpp == "std::vector<decltype(0)> values = {0, 1, 2, 3};"


def test_vector_find_out_type():
    source = parse("values = []", "values.append(1)")
    cpp = transpile(source)
    assert cpp == parse("std::vector<decltype(1)> values = {};", "values.push_back(1);")


def test_map_function():
    source = parse(
        "def map(values, fun):",
        "   results = []",
        "   for v in values:",
        "       results.append(fun(v))",
        "   return results",
    )
    cpp = transpile(source)
    expected = """\
        template <typename T0, typename T1>auto map(T0 values, T1 fun) {
        std::vector<decltype(fun(std::declval<typename decltype(values)::value_type>()))> results = {};
        for(auto v : values) {
        results.push_back(fun(v));
        }
        return results;}
    """
    assert cpp == textwrap.dedent(expected)


def test_range_with_variable_bounds_uses_parameterized_lambda():
    source = parse("def f(n):", "    for i in range(n):", "        print(i)")
    cpp = transpile(source)
    assert "([](int __start, int __stop, int __step)" in cpp
    assert "static_cast<int>(n)" in cpp


def test_append_on_typed_list_annotation_uses_push_back():
    source = parse(
        "from typing import List",
        "values: List[int] = []",
        "values.append(1)",
    )
    cpp = transpile(source)
    assert "values.push_back(1);" in cpp


def test_set_add_rewritten_to_insert():
    source = parse(
        "from typing import Set",
        "seen: Set[int] = {1, 2}",
        "seen.add(3)",
    )
    cpp = transpile(source)
    assert "seen.insert(3);" in cpp


def test_dict_keys_call_rewritten_to_vector_of_keys():
    source = parse(
        "from typing import Dict",
        "d: Dict[int, int] = {1: 2, 3: 4}",
        "ks = d.keys()",
    )
    cpp = transpile(source)
    assert "std::vector<__KeyT> __keys;" in cpp
    assert "for (const auto& __kv : __m) __keys.push_back(__kv.first);" in cpp


def test_dict_values_call_rewritten_to_vector_of_values():
    source = parse(
        "from typing import Dict",
        "d: Dict[int, int] = {1: 2, 3: 4}",
        "vs = d.values()",
    )
    cpp = transpile(source)
    assert "std::vector<__ValueT> __values;" in cpp
    assert "for (const auto& __kv : __m) __values.push_back(__kv.second);" in cpp


def test_dict_get_call_rewritten():
    source = parse(
        "from typing import Dict",
        "d: Dict[int, int] = {1: 2}",
        "x = d.get(1, 0)",
    )
    cpp = transpile(source)
    assert "auto __it = __m.find(__k);" in cpp
    assert "return __it == __m.end() ? __default : __it->second;" in cpp


def test_set_remove_rewritten_to_erase():
    source = parse(
        "from typing import Set",
        "s: Set[int] = {1, 2}",
        "s.remove(1)",
    )
    cpp = transpile(source)
    assert "s.erase(1);" in cpp


def test_contains_dunder_rewritten():
    source = parse(
        "from typing import Set",
        "s: Set[int] = {1, 2}",
        "ok = s.__contains__(1)",
    )
    cpp = transpile(source)
    assert "std::find(s.begin(), s.end(), 1)" in cpp


def test_for_over_dict_iterates_keys():
    source = parse(
        "for k in {1: 2}:",
        "    x = d[k]",
    )
    cpp = transpile(source)
    assert "for(const auto& __kv : std::map<decltype(1), decltype(2)>{{ 1, 2 }}) {" in cpp
    assert "auto k = __kv.first;" in cpp


def test_list_constructor_from_list_variable():
    source = parse("values = [1, 2]", "copied = list(values)")
    cpp = transpile(source)
    assert cpp == parse(
        "std::vector<decltype(1)> values = {1, 2};",
        "auto copied = decltype(values)(values.begin(), values.end());",
    )


def test_empty_list_constructor():
    source = parse("items = list()")
    cpp = transpile(source)
    assert cpp == "auto items = std::vector<int>{};"


def test_list_constructor_from_list_literal_supported():
    source = parse("items = list([1, 2, 3])")
    cpp = transpile(source)
    assert cpp == "auto items = std::vector<decltype(1)>{1, 2, 3};"


def test_empty_dict_constructor():
    source = parse("mapping = dict()")
    cpp = transpile(source)
    assert cpp == "auto mapping = std::map<int, int>{};"


def test_empty_set_constructor():
    source = parse("seen = set()")
    cpp = transpile(source)
    assert cpp == "auto seen = std::set<int>{};"


def test_set_constructor_from_list_literal_supported():
    source = parse("seen = set([1, 2, 2])")
    cpp = transpile(source)
    assert "std::vector<decltype(1)> __tmp = {1, 2, 2};" in cpp
    assert "return std::set<decltype(1)>(__tmp.begin(), __tmp.end());" in cpp


def test_set_constructor_from_list_variable_requires_container_type():
    source = parse(
        "from typing import List",
        "values: List[int] = [1, 2, 2]",
        "uniq = set(values)",
    )
    with pytest.raises(AstNotImplementedError):
        transpile(source)


def test_dict_constructor_from_dict_variable_requires_container_type():
    source = parse(
        "from typing import Dict",
        "a: Dict[int, int] = {1: 2, 3: 4}",
        "b = dict(a)",
    )
    with pytest.raises(AstNotImplementedError):
        transpile(source)


def test_list_constructor_from_set_variable_requires_container_type():
    source = parse(
        "from typing import Set",
        "items: Set[int] = {1, 2, 3}",
        "as_list = list(items)",
    )
    with pytest.raises(AstNotImplementedError):
        transpile(source)


def test_ord_constructor():
    source = parse("x = ord('A')")
    cpp = transpile(source)
    assert cpp == "auto x = static_cast<int>('A');"


def test_math_pi_attribute():
    source = parse("import math", "x = math.pi")
    cpp = transpile(source)
    assert cpp == parse("", "auto x = 3.14159265358979323846;")


def test_dict_constructor_from_iterable_pairs_variable_not_supported():
    source = parse("pairs = [(1, 2), (3, 4)]", "y = dict(pairs)")
    with pytest.raises(AstNotImplementedError):
        transpile(source)


def test_dict_constructor_from_iterable_pairs_literal_supported():
    source = parse("y = dict([(1, 2), (3, 4)])")
    cpp = transpile(source)
    assert (
        cpp
        == "auto y = std::map<decltype(1), decltype(2)>{{ 1, 2 }, { 3, 4 }};"
    )


def test_list_constructor_unknown_not_supported():
    source = parse("def f(x):", "    y = list(x)")
    with pytest.raises(AstNotImplementedError):
        transpile(source)


def test_set_constructor_unknown_not_supported():
    source = parse("def f(x):", "    y = set(x)")
    with pytest.raises(AstNotImplementedError):
        transpile(source)


def test_dict_constructor_untyped_variable_not_supported():
    source = parse("a = {1: 2, 3: 4}", "b = dict(a)")
    with pytest.raises(AstNotImplementedError):
        transpile(source)


def test_list_constructor_from_dict_not_supported():
    source = parse(
        "from typing import Dict",
        "a: Dict[int, int] = {1: 2, 3: 4}",
        "b = list(a)",
    )
    with pytest.raises(AstNotImplementedError):
        transpile(source)


def test_set_constructor_from_dict_not_supported():
    source = parse(
        "from typing import Dict",
        "a: Dict[int, int] = {1: 2, 3: 4}",
        "b = set(a)",
    )
    with pytest.raises(AstNotImplementedError):
        transpile(source)


def test_list_constructor_from_dict_literal_supported():
    source = parse("b = list({1: 2, 3: 4})")
    cpp = transpile(source)
    assert (
        "for (const auto& __kv : std::map<decltype(1), decltype(2)>{{ 1, 2 }, { 3, 4 }})"
        in cpp
    )
    assert "std::vector<decltype(1)> __keys;" in cpp


def test_set_constructor_from_dict_literal_supported():
    source = parse("b = set({1: 2, 3: 4})")
    cpp = transpile(source)
    assert (
        "for (const auto& __kv : std::map<decltype(1), decltype(2)>{{ 1, 2 }, { 3, 4 }})"
        in cpp
    )
    assert "std::set<decltype(1)> __keys;" in cpp
