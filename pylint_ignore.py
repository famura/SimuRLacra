""" Gratefully copied from https://github.com/PyCQA/pylint/issues/2686#issuecomment-621927895. """

from pylint.utils import utils


class PylintIgnorePaths:  # pylint: disable=too-few-public-methods
    """Helper class for ignoring directories in PyLint."""

    def __init__(self, *paths):
        self.paths = paths
        self.original_expand_modules = utils.expand_modules
        utils.expand_modules = self.patched_expand

    def patched_expand(self, *args, **kwargs):  # pylint: disable=missing-function-docstring
        result, errors = self.original_expand_modules(*args, **kwargs)
        result = list(filter(lambda item: not any(True for path in self.paths if path in item["path"]), result))
        return result, errors
