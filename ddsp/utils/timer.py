#!/usr/bin/env python

# IMPORTS

import time
from typing import Callable


# IMPLEMENTATION


class Timer(object):

    def __init__(self, precision: int = 1, print_fn: Callable = print):
        self.precision = precision
        self._print_fn = print_fn

    def __enter__(self):
        self.start_time = time.time()
        self.print_fn("-" * 80)  # Section break indicator
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        timestr = f"{self.elapsed_time:.{self.precision}f}"
        self.print_fn(f"Elapsed time: {timestr} seconds")


# Usage example
with Timer() as timer:
    # Place the code you want to measure here
    for _ in range(1000000):
        pass
