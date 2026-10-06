class InterpolationError(ValueError):
    """Base class for errors raised while interpolating variables."""


class InterpolationSyntaxError(InterpolationError):
    """A ``${...}`` variable expression is malformed."""


class InterpolationCycleError(InterpolationError):
    """Expanding a variable directly or indirectly requires the variable itself."""


class InterpolationDepthError(InterpolationError):
    """Variable expansion exceeded the maximum supported nesting depth."""


class InterpolationValidationError(InterpolationError):
    """A ``${VAR:?message}`` or ``${VAR?message}`` validation check failed."""
