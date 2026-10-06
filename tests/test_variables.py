import pytest

from dotenv.exceptions import InterpolationSyntaxError
from dotenv.variables import Literal, Variable, parse_variables


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", []),
        ("a", [Literal(value="a")]),
        ("${a}", [Variable(name="a", default=None)]),
        ("${a:-b}", [Variable(name="a", default="b")]),
        (
            "${a}${b}",
            [
                Variable(name="a", default=None),
                Variable(name="b", default=None),
            ],
        ),
        (
            "a${b}c${d}e",
            [
                Literal(value="a"),
                Variable(name="b", default=None),
                Literal(value="c"),
                Variable(name="d", default=None),
                Literal(value="e"),
            ],
        ),
        ("${a:=b}", [Variable(name="a", default="b", op=":=")]),
        ("${a:?b}", [Variable(name="a", default="b", op=":?")]),
        ("${a:+b}", [Variable(name="a", default="b", op=":+")]),
        ("${a-b}", [Variable(name="a", default="b", op="-")]),
        ("${a=b}", [Variable(name="a", default="b", op="=")]),
        ("${a?b}", [Variable(name="a", default="b", op="?")]),
        ("${a+b}", [Variable(name="a", default="b", op="+")]),
        ("${a:-}", [Variable(name="a", default="", op=":-")]),
        ("\\${a}", [Literal(value="${a}")]),
        ("\\$a", [Literal(value="$a")]),
        ("\\$", [Literal(value="$")]),
        ("$a", [Literal(value="$a")]),
        (
            "${a:-${b}}",
            [Variable(name="a", default="${b}", op=":-")],
        ),
        (
            "${a:-${b:-${c}}}",
            [Variable(name="a", default="${b:-${c}}", op=":-")],
        ),
        (
            "x${a:-p${b}q}y",
            [
                Literal(value="x"),
                Variable(name="a", default="p${b}q", op=":-"),
                Literal(value="y"),
            ],
        ),
    ],
)
def test_parse_variables(value, expected):
    result = parse_variables(value)

    assert list(result) == expected


@pytest.mark.parametrize(
    "value",
    [
        "${a",
        "${}",
        "${a:-${b}",
        "${a:}",
    ],
)
def test_parse_variables_syntax_error(value):
    with pytest.raises(InterpolationSyntaxError):
        list(parse_variables(value))


def test_parse_variables_nested_atoms():
    (variable,) = parse_variables("${a:-${b}x}")

    assert variable.atoms == [Variable(name="b", default=None), Literal(value="x")]
