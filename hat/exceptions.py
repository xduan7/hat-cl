"""Exceptions and warnings for easier logging and debugging."""


class HATInitializationError(Exception):
    """Raised when the HAT model is not initialized properly and therefore
    cannot get complete information about the masks.

    """

    pass


class MaskerLockedError(Exception):
    """Raised when the masker is locked and therefore cannot be modified."""

    pass


class MaskDimensionMismatchError(Exception):
    """Raised when the shape of mask is not compatible with the shape of
    the data.
    """

    pass


class ModuleConversionWarning(UserWarning):
    """Raised when a module is unable to be converted to a task dependent
    module or vice versa.
    """

    pass


class LearningSuppressedWarning(UserWarning):
    """Raised when the learning ability of a hard attention masked module
    is suppressed.
    """

    pass


class MaskDimensionInferenceWarning(UserWarning):
    """Raised when the dimension of the mask cannot be inferred from the
    data.
    """

    pass


class NoParameterToForgetWarning(UserWarning):
    """Raised when there is no parameter to forget by the given task id and
    the task dependent module.

    """

    pass


class InsufficientMaskWarning(UserWarning):
    """Raised when the mask is not sufficient for all the tasks."""

    pass
