from optkit.compat import *

import os

class TempFileContext:
    def __init__(self, build_file):
        self._file = None
        try:
            self._file = build_file()
        except:
            raise

    def __enter__(self):
        return self._file

    def __exit__(self, *exc):
        if self._file is not None:
            if os.path.exists(self._file):
                os.remove(self._file)