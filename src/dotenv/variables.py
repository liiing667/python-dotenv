from abc import ABCMeta, abstractmethod
from typing import (
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)

from .exceptions import (
    InterpolationCycleError,
    InterpolationDepthError,
    InterpolationSyntaxError,
    InterpolationValidationError,
)

MAX_INTERPOLATION_DEPTH = 32

_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)

# Two-character operators must be listed before their one-character prefixes.
_OPERATORS = (":-", ":=", ":?", ":+", "-", "=", "?", "+")


class Atom(metaclass=ABCMeta):
    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @abstractmethod
    def resolve(self, env: Mapping[str, Optional[str]]) -> str: ...


class Literal(Atom):
    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Literal(value={self.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash((self.__class__, self.value))

    def resolve(self, env: Mapping[str, Optional[str]]) -> str:
        return self.value


class Variable(Atom):
    def __init__(
        self, name: str, default: Optional[str], op: Optional[str] = None
    ) -> None:
        self.name = name
        self.default = default
        self.op = op
        self._atoms: Optional[List[Atom]] = None

    def __repr__(self) -> str:
        return f"Variable(name={self.name}, default={self.default})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.name, self.default) == (other.name, other.default)

    def __hash__(self) -> int:
        return hash((self.__class__, self.name, self.default))

    @property
    def atoms(self) -> List[Atom]:
        """The parsed atoms of the word following the operator, if any."""
        if self._atoms is None:
            self._atoms = (
                _parse(self.default, depth=1) if self.default is not None else []
            )
        return self._atoms

    @property
    def effective_op(self) -> Optional[str]:
        if self.op is not None:
            return self.op
        # Variables constructed without an operator keep the historical
        # behavior: a default implies `:-`.
        return ":-" if self.default is not None else None

    def resolve(self, env: Mapping[str, Optional[str]]) -> str:
        assign: Optional[Callable[[str, str], None]] = None
        if isinstance(env, MutableMapping):
            assign = env.__setitem__
        resolver = _Resolver(lookup=env.get, assign=assign)
        return resolver.resolve_variable(self, depth=0)


class _Resolver:
    """Resolve atoms against a lookup function, tracking the expansion stack
    so that cyclic references and excessive nesting can be reported."""

    def __init__(
        self,
        lookup: Callable[[str], Optional[str]],
        assign: Optional[Callable[[str, str], None]] = None,
        max_depth: int = MAX_INTERPOLATION_DEPTH,
    ) -> None:
        self.lookup = lookup
        self.assign = assign
        self.max_depth = max_depth
        self.stack: List[str] = []

    def resolve_atoms(self, atoms: Iterable[Atom], depth: int) -> str:
        if depth > self.max_depth:
            raise InterpolationDepthError(
                f"variable expansion exceeds the maximum depth of {self.max_depth}"
            )
        parts: List[str] = []
        for atom in atoms:
            if isinstance(atom, Variable):
                parts.append(self.resolve_variable(atom, depth))
            else:
                parts.append(atom.resolve({}))
        return "".join(parts)

    def resolve_variable(self, var: Variable, depth: int) -> str:
        value = self.lookup(var.name)
        if var.name in self.stack and value is None:
            # The variable is already being expanded and its value cannot be
            # determined without finishing that expansion: this is a cycle.
            # A self-reference whose value is already known (e.g.
            # `${VAR:+${VAR}}` with `VAR` set) is not cyclic.
            cycle = self.stack[self.stack.index(var.name) :] + [var.name]
            raise InterpolationCycleError(
                "cyclic variable reference: " + " -> ".join(cycle)
            )
        self.stack.append(var.name)
        try:
            is_set = value is not None
            is_nonempty = is_set and value != ""
            op = var.effective_op

            if op is None:
                return value if value is not None else ""
            if op in (":-", "-"):
                if (op == ":-" and not is_nonempty) or (op == "-" and not is_set):
                    return self.resolve_atoms(var.atoms, depth + 1)
                return value or ""
            if op in (":=", "="):
                if (op == ":=" and not is_nonempty) or (op == "=" and not is_set):
                    result = self.resolve_atoms(var.atoms, depth + 1)
                    if self.assign is not None:
                        self.assign(var.name, result)
                    return result
                return value or ""
            if op in (":?", "?"):
                if (op == ":?" and not is_nonempty) or (op == "?" and not is_set):
                    message = self.resolve_atoms(var.atoms, depth + 1)
                    if not message:
                        message = "parameter null or not set"
                    raise InterpolationValidationError(f"{var.name}: {message}")
                return value or ""
            if op in (":+", "+"):
                if (op == ":+" and is_nonempty) or (op == "+" and is_set):
                    return self.resolve_atoms(var.atoms, depth + 1)
                return ""
            raise InterpolationSyntaxError(
                f"unsupported operator {op!r} for variable {var.name!r}"
            )
        finally:
            self.stack.pop()


def parse_variables(value: str) -> Iterator[Atom]:
    return iter(_parse(value, depth=0))


def _parse(value: str, depth: int) -> List[Atom]:
    atoms: List[Atom] = []
    literal: List[str] = []
    cursor = 0
    length = len(value)
    while cursor < length:
        if value.startswith("\\$", cursor):
            # An escaped dollar is kept as a literal `$` and never expanded.
            literal.append("$")
            cursor += 2
        elif value.startswith("${", cursor):
            if literal:
                atoms.append(Literal(value="".join(literal)))
                literal = []
            variable, cursor = _parse_variable(value, cursor, depth)
            atoms.append(variable)
        else:
            literal.append(value[cursor])
            cursor += 1
    if literal:
        atoms.append(Literal(value="".join(literal)))
    return atoms


def _parse_variable(value: str, start: int, depth: int) -> Tuple[Variable, int]:
    """Parse a single `${...}` expression starting at `start` (which points at
    the `$`). Return the variable and the position right after the closing
    brace."""
    cursor = start + 2
    length = len(value)
    name_start = cursor
    while cursor < length and value[cursor] in _NAME_CHARS:
        cursor += 1
    name = value[name_start:cursor]
    if not name:
        raise InterpolationSyntaxError(
            f"missing variable name in expression {value[start:]!r}"
        )
    if cursor < length and value[cursor] == "}":
        return Variable(name=name, default=None), cursor + 1
    op = next(
        (candidate for candidate in _OPERATORS if value.startswith(candidate, cursor)),
        None,
    )
    if op is None:
        raise InterpolationSyntaxError(
            f"invalid or missing operator in expression {value[start:]!r}"
        )
    cursor += len(op)
    word, cursor = _extract_word(value, cursor)
    if depth + 1 > MAX_INTERPOLATION_DEPTH:
        raise InterpolationDepthError(
            f"variable expression nesting exceeds the maximum depth of "
            f"{MAX_INTERPOLATION_DEPTH}"
        )
    variable = Variable(name=name, default=word, op=op)
    variable._atoms = _parse(word, depth + 1)
    return variable, cursor + 1


def _extract_word(value: str, start: int) -> Tuple[str, int]:
    """Extract the raw word following an operator, up to the closing brace
    that matches the opening brace of the current expression. Nested `${...}`
    expressions are skipped over, and `\\$` escapes are preserved verbatim."""
    cursor = start
    length = len(value)
    nested = 0
    while cursor < length:
        if value.startswith("\\$", cursor):
            cursor += 2
        elif value.startswith("${", cursor):
            nested += 1
            cursor += 2
        elif value[cursor] == "}":
            if nested == 0:
                return value[start:cursor], cursor
            nested -= 1
            cursor += 1
        else:
            cursor += 1
    raise InterpolationSyntaxError(
        f"missing closing '}}' in variable expression {value!r}"
    )
