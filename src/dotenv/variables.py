from abc import ABCMeta, abstractmethod
from typing import Iterator, List, MutableMapping, Optional, Tuple

# Maximum number of nested variable expressions expanded while resolving a
# single value. Keeps pathological input from exhausting the call stack.
MAX_EXPANSION_DEPTH = 100


class InterpolationError(ValueError):
    """Base class for errors raised while expanding variables."""


class InterpolationSyntaxError(InterpolationError):
    """A variable expression is malformed (e.g. unterminated or bad operator)."""


class CyclicVariableError(InterpolationSyntaxError):
    """Expanding a variable requires expanding the same variable again."""


class MissingVariableError(InterpolationError):
    """A variable required by ``${NAME:?message}`` is missing or empty."""


class Atom(metaclass=ABCMeta):
    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @abstractmethod
    def resolve(
        self,
        env: MutableMapping[str, Optional[str]],
        depth: int = 0,
        stack: Tuple[str, ...] = (),
    ) -> str: ...


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

    def resolve(
        self,
        env: MutableMapping[str, Optional[str]],
        depth: int = 0,
        stack: Tuple[str, ...] = (),
    ) -> str:
        return self.value


class Variable(Atom):
    """``${NAME}`` and ``${NAME:-default}``."""

    def __init__(self, name: str, default: Optional[str]) -> None:
        self.name = name
        self.default = default

    def __repr__(self) -> str:
        return f"Variable(name={self.name}, default={self.default})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.name, self.default) == (other.name, other.default)

    def __hash__(self) -> int:
        return hash((self.__class__, self.name, self.default))

    def resolve(
        self,
        env: MutableMapping[str, Optional[str]],
        depth: int = 0,
        stack: Tuple[str, ...] = (),
    ) -> str:
        if self.name in env:
            result = env[self.name]
            return result if result is not None else ""
        if self.default is None:
            return ""
        return _expand_argument(self.name, self.default, env, depth, stack)


class Assign(Atom):
    """``${NAME:=default}``: use and remember ``default`` when NAME is missing."""

    def __init__(self, name: str, default: str) -> None:
        self.name = name
        self.default = default

    def __repr__(self) -> str:
        return f"Assign(name={self.name}, default={self.default})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.name, self.default) == (other.name, other.default)

    def __hash__(self) -> int:
        return hash((self.__class__, self.name, self.default))

    def resolve(
        self,
        env: MutableMapping[str, Optional[str]],
        depth: int = 0,
        stack: Tuple[str, ...] = (),
    ) -> str:
        result = env.get(self.name)
        if result is not None:
            return result
        result = _expand_argument(self.name, self.default, env, depth, stack)
        env[self.name] = result
        return result


class Alternate(Atom):
    """``${NAME:+alternate}``: use ``alternate`` only when NAME is non-empty."""

    def __init__(self, name: str, alt: str) -> None:
        self.name = name
        self.alt = alt

    def __repr__(self) -> str:
        return f"Alternate(name={self.name}, alt={self.alt})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.name, self.alt) == (other.name, other.alt)

    def __hash__(self) -> int:
        return hash((self.__class__, self.name, self.alt))

    def resolve(
        self,
        env: MutableMapping[str, Optional[str]],
        depth: int = 0,
        stack: Tuple[str, ...] = (),
    ) -> str:
        if env.get(self.name):
            return _expand_argument(self.name, self.alt, env, depth, stack)
        return ""


class Require(Atom):
    """``${NAME:?message}``: raise when NAME is missing or empty."""

    def __init__(self, name: str, message: str) -> None:
        self.name = name
        self.message = message

    def __repr__(self) -> str:
        return f"Require(name={self.name}, message={self.message})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.name, self.message) == (other.name, other.message)

    def __hash__(self) -> int:
        return hash((self.__class__, self.name, self.message))

    def resolve(
        self,
        env: MutableMapping[str, Optional[str]],
        depth: int = 0,
        stack: Tuple[str, ...] = (),
    ) -> str:
        result = env.get(self.name)
        if result:
            return result
        if self.message:
            message = _expand_argument(self.name, self.message, env, depth, stack)
        else:
            message = "required variable is missing or empty"
        raise MissingVariableError(f"{self.name}: {message}")


def _guard_expansion(
    name: str,
    depth: int,
    stack: Tuple[str, ...],
) -> Tuple[int, Tuple[str, ...]]:
    if name in stack:
        cycle = stack[stack.index(name) :] + (name,)
        raise CyclicVariableError("cyclic variable reference: " + " -> ".join(cycle))
    if depth + 1 > MAX_EXPANSION_DEPTH:
        raise InterpolationSyntaxError(
            f"maximum expansion depth ({MAX_EXPANSION_DEPTH}) exceeded "
            f"while expanding ${{{name}}}"
        )
    return depth + 1, stack + (name,)


def _expand_argument(
    name: str,
    argument: str,
    env: MutableMapping[str, Optional[str]],
    depth: int,
    stack: Tuple[str, ...],
) -> str:
    next_depth, next_stack = _guard_expansion(name, depth, stack)
    return _expand(argument, env, next_depth, next_stack)


def _expand(
    value: str,
    env: MutableMapping[str, Optional[str]],
    depth: int = 0,
    stack: Tuple[str, ...] = (),
) -> str:
    return "".join(
        atom.resolve(env, depth, stack) for atom in _parse_atoms(value, quoted=True)
    )


def _find_closing_brace(value: str, pos: int) -> int:
    """Return the position of the ``}`` closing the expression opened at pos."""
    depth = 1
    length = len(value)
    while pos < length:
        char = value[pos]
        if char == "\\" and value[pos + 1 : pos + 2] == "$":
            pos += 2
        elif char == "'":
            end = value.find("'", pos + 1)
            if end == -1:
                pos += 1
            else:
                pos = end + 1
        elif char == "$" and value[pos + 1 : pos + 2] == "{":
            depth += 1
            pos += 2
        elif char == "}":
            depth -= 1
            if depth == 0:
                return pos
            pos += 1
        else:
            pos += 1
    raise InterpolationSyntaxError(
        f"unterminated variable expression: {value!r} is missing a closing '}}'"
    )


def _parse_variable(value: str, start: int) -> Tuple[Atom, int]:
    # value[start] == "$" and value[start + 1] == "{"
    length = len(value)
    cursor = start + 2
    name_start = cursor
    while cursor < length and value[cursor] not in ":}":
        cursor += 1
    if cursor == length:
        raise InterpolationSyntaxError(
            f"unterminated variable expression: {value[start:]!r} is missing '}}'"
        )
    name = value[name_start:cursor]
    if value[cursor] == "}":
        return Variable(name=name, default=None), cursor + 1

    # value[cursor] == ":"
    operator_start = cursor + 1
    if operator_start >= length or value[operator_start] not in "-=+?":
        raise InterpolationSyntaxError(
            f"invalid operator in variable expression: "
            f"{value[start : operator_start + 1]!r}; expected one of ':-', ':=', "
            f"':+' or ':?'"
        )
    operator = value[operator_start]
    argument_start = operator_start + 1
    end = _find_closing_brace(value, argument_start)
    argument = value[argument_start:end]

    if operator == "-":
        atom: Atom = Variable(name=name, default=argument)
    elif operator == "=":
        atom = Assign(name=name, default=argument)
    elif operator == "+":
        atom = Alternate(name=name, alt=argument)
    else:
        atom = Require(name=name, message=argument)
    return atom, end + 1


def _parse_atoms(value: str, quoted: bool) -> List[Atom]:
    atoms: List[Atom] = []
    literal: List[str] = []
    cursor = 0
    length = len(value)

    def flush_literal() -> None:
        if literal:
            atoms.append(Literal(value="".join(literal)))
            literal.clear()

    while cursor < length:
        char = value[cursor]
        if char == "\\" and value[cursor + 1 : cursor + 2] == "$":
            literal.append("$")
            cursor += 2
        elif quoted and char == "'":
            end = value.find("'", cursor + 1)
            if end == -1:
                literal.append(char)
                cursor += 1
            else:
                literal.append(value[cursor + 1 : end])
                cursor = end + 1
        elif char == "$" and value[cursor + 1 : cursor + 2] == "{":
            flush_literal()
            atom, cursor = _parse_variable(value, cursor)
            atoms.append(atom)
        else:
            literal.append(char)
            cursor += 1

    flush_literal()
    return atoms


def parse_variables(value: str) -> Iterator[Atom]:
    return iter(_parse_atoms(value, quoted=False))
