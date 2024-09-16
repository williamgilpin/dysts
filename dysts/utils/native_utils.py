"""Native python utilities"""

import threading


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
