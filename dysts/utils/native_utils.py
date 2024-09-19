"""Native python utilities"""

import gzip
import importlib
import json
import os
import threading
import warnings


def has_module(module_name: str) -> bool:
    """Check if a module is installed"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def convert_json_to_gzip(fpath, encoding="utf-8", delete_original=False):
    """
    Convert a json file to a gzip file in a format that can be easily read by the
    `dysts` package. By default, the gzip file will be saved with the same name and
    in the same directory as the json file, but with a ".gz" extension.

    Args:
        fpath (str): Path to the json file to be converted
        encoding (str): Encoding to use when writing the gzip file
        delete_original (bool): Whether to delete the original json file after
            conversion. Default is False.

    Returns:
        None

    """
    if os.path.splitext(fpath)[1] == ".gz":
        warnings.warn("File already gzipped, exiting without conversion")
        return None

    with open(fpath, "r") as file:
        data = json.load(file)

    with gzip.open(fpath + ".gz", "wt", encoding=encoding) as file:
        json.dump(data, file, indent=4)

    if delete_original:
        os.remove(fpath)


def group_consecutives(vals, step=1):
    """
    Return list of consecutive lists of numbers from vals (number list).

    References:
        Modified from the following
        https://stackoverflow.com/questions/7352684/
        how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    """
    run = list()
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


class ComputationHolder:
    """
    A wrapper class to force a computation to stop after a timeout.

    Parameters
        func (callable): the function to run
        args (tuple): the arguments to pass to the function
        kwargs (dict): the keyword arguments to pass to the function
        timeout (int): the timeout in seconds. If None is passed, the computation
            will run indefinitely until it finishes.

    Example
        >>> def my_func():
        ...     while True:
        ...         print("hello")
        ...         time.sleep(8)
        >>> ch = ComputationHolder(my_func, timeout=3)
        >>> ch.run()
        hello
        hello
        hello
        None

    """

    def __init__(self, func=None, *args, timeout=10, **kwargs):
        self.sol = None
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout

        def func_wrapped():
            self.sol = self.func(*self.args, **self.kwargs)

        self.func_wrapped = func_wrapped

    def run(self):
        my_thread = threading.Thread(target=self.func_wrapped)
        my_thread.start()
        my_thread.join(self.timeout)  # kill the thread after `timeout` seconds

        if self.sol is None:
            return None
        else:
            return self.sol
