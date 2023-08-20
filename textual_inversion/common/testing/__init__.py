import os
from pathlib import Path

from tango.common.testing import TangoTestCase


class TextualInversionTestCase(TangoTestCase):
    PROJECT_ROOT = (Path(__file__).parent / ".." / ".." / "..").resolve()
    """
    Root of the git repository.
    """

    # to run test suite with finished package, which does not contain
    # tests & fixtures, we must be able to look them up somewhere else
    PROJECT_ROOT_FALLBACK = (
        # users wanting to run test suite for installed package
        Path(os.environ["TEXTUAL_INVERSION_SRC_DIR"])
        if "TEXTUAL_INVERSION_SRC_DIR" in os.environ
        else (
            # fallback for conda packaging
            Path(os.environ["SRC_DIR"])
            if "CONDA_BUILD" in os.environ
            # stay in-tree
            else PROJECT_ROOT
        )
    )

    MODULE_ROOT = PROJECT_ROOT_FALLBACK / "textual_inversion"
    """
    Root of the tango module.
    """

    TESTS_ROOT = PROJECT_ROOT_FALLBACK / "tests"
    """
    Root of the tests directory.
    """

    FIXTURES_ROOT = PROJECT_ROOT_FALLBACK / "test_fixtures"
    """
    Root of the test fixtures directory.
    """
