import pytest

from dotenv.variables import (
    Alternate,
    Assign,
    InterpolationSyntaxError,
    Literal,
    Require,
    Variable,
    parse_variables,
)


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
        ("${a:=b}", [Assign(name="a", default="b")]),
        ("${a:+b}", [Alternate(name="a", alt="b")]),
        ("${a:?b}", [Require(name="a", message="b")]),
        ("${a:?}", [Require(name="a", message="")]),
        ("${a:-${b}}", [Variable(name="a", default="${b}")]),
        ("${a:=${b:-c}}", [Assign(name="a", default="${b:-c}")]),
        ("${a:+${b}${c}}", [Alternate(name="a", alt="${b}${c}")]),
        ("${a:-${b:-${c}}}", [Variable(name="a", default="${b:-${c}}")]),
        ("\\${a}", [Literal(value="${a}")]),
        ("\\$a", [Literal(value="$a")]),
        ("x\\${a}y", [Literal(value="x${a}y")]),
        ("${a:-'}'}", [Variable(name="a", default="'}'")]),
        ("${a:-'${b}'}", [Variable(name="a", default="'${b}'")]),
        (
            "${a:-\\${b}}",
            [Variable(name="a", default="\\${b"), Literal(value="}")],
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
        "${a:-b",
        "${a:-${b}",
        "${a:}",
        "${a:x}",
        "${a:~1}",
    ],
)
def test_parse_variables_syntax_error(value):
    with pytest.raises(InterpolationSyntaxError):
        list(parse_variables(value))
