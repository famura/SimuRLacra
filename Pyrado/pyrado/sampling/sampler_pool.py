# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import traceback
from copy import deepcopy
from enum import Enum, auto
from queue import Empty
from time import sleep

import torch.multiprocessing as mp
from tqdm import tqdm

import pyrado


class GlobalNamespace:
    """Type of the worker's global namespace"""


_CMD_STOP = "stop"
_RES_SUCCESS = "success"
_RES_ERROR = "error"
_RES_FATAL = "fatal"


def _pool_worker(from_master, to_master):
    """
    Two queues: from master and to master

    :param from_master: tuple (func, args) or the special _CMD_STOP
    :param to_master: tuple (success, obj), where obj is the result on success=True and an error message on success=False
    """
    # Use a custom global namespace. This 'trick'
    G = GlobalNamespace()

    while True:
        # Get command from master, should be a tuple
        cmd = from_master.get()
        if cmd == _CMD_STOP:
            # Terminate, no need to send a message
            break

        # CMD is a tuple of (func, args)
        func, args, kwargs = cmd

        # Invoke func
        # noinspection PyBroadException
        try:
            res = func(G, *args, **kwargs)
        except Exception:  # pylint: disable=broad-except
            msg = traceback.format_exc()
            to_master.put((_RES_ERROR, msg))
        except:
            # Generic exception: still report, but also terminate the worker
            msg = traceback.format_exc()
            to_master.put((_RES_FATAL, msg))
            raise
        else:
            to_master.put((_RES_SUCCESS, res))


class _OpState(Enum):
    PENDING = auto()
    ERRORED = auto()
    DONE = auto()


class _WorkerInfo:
    """
    Internal class managing a single worker process in the sampler pool.
    """

    def __init__(self, num):
        self._to_slave = mp.Queue()
        self._from_slave = mp.Queue()

        # Create the process
        self._process = mp.Process(
            target=_pool_worker,
            args=(self._to_slave, self._from_slave),
            name=f"Sampler-Worker-{num}",
        )
        self._process.daemon = True
        # Start it
        self._process.start()

        # Track pending invocations
        self._pending = False

        # Stores early retrieved result
        self._result = None

    def invoke_start(self, func, *args, **kwargs):
        if not self._process.is_alive():
            raise RuntimeError("Worker has terminated")
        if self._pending:
            raise RuntimeError("There is still a pending call waiting for completion.")
        # Send to slave
        self._to_slave.put((func, args, kwargs))
        self._pending = True

    def operation_state(self):
        """
        Try to retrieve the result
        :return: False if there was an error result and the loop should be cancelled
        """
        if not self._process.is_alive():
            raise RuntimeError("Worker has terminated")
        if not self._pending:
            return _OpState.PENDING

        if self._result is None:
            # Try to pull
            try:
                self._result = self._from_slave.get(block=False)
            except Empty:
                # Nothing there yet
                return _OpState.PENDING

        # Check result
        if self._result[0] == _RES_SUCCESS:
            return _OpState.DONE
        else:
            return _OpState.ERRORED

    def invoke_wait(self):
        if not self._process.is_alive():
            raise RuntimeError("Worker has terminated")
        if not self._pending:
            raise RuntimeError("There is no pending call.")

        if self._result is None:
            # Await result if not done yet
            res = self._from_slave.get()
        else:
            # Clear stored result
            res = self._result
            self._result = None
        self._pending = False

        # Interpret result
        stat, value = res

        if stat == _RES_SUCCESS:
            return value
        elif stat == _RES_ERROR:
            raise RuntimeError(f"Caught error in {self._process.name}:\n{value}")
        elif stat == _RES_FATAL:
            # Mark as failed. For now, there is no way to recover, we just get an error on the next start method.
            raise RuntimeError(f"Fatal error in {self._process.name}:\n{value}")
        raise pyrado.ValueErr(given=stat, eq_constraint="_RES_SUCCESS, _RES_ERROR, or _RES_FATAL")

    def stop(self):
        if self._pending:
            raise RuntimeError("There is still a pending call waiting for completion.")
        # Send stop signal
        self._to_slave.put(_CMD_STOP)

        # Wait a bit for the process to die
        self._process.join(2)
        # Check if stopped
        if self._process.is_alive():
            # send SIGTERM in case there was a problem with the graceful stop
            self._process.terminate()
            # Wait for the process to die from the SIGTERM
            self._process.join(2)

            # Check if stopped
            if self._process.is_alive():
                # Sigterm didn't work, so break out the big guns with SIGKILL
                self._process.kill()


def _run_set_seed(seed):
    """Ignore global space, and forward to `pyrado.set_seed()`"""
    pyrado.set_seed(seed)


def _run_collect(G, counter, run_counter, lock, n, min_runs, func, args, kwargs):
    """Worker function for run_collect"""
    result = []
    while True:
        # Invoke once
        res, n_done = func(G, *args, **kwargs)

        # Add to result and record increment
        result.append(res)
        with lock:
            # decrement work counter
            counter.value += n_done
            run_counter.value += 1
            # Check if done
            if counter.value >= n and (min_runs is None or run_counter.value >= min_runs):
                break
    return result


def _run_map(G, func, argqueue):
    result = []
    while True:
        try:
            index, arg = argqueue.get(block=False)
        except Empty:
            break
        result.append((index, func(G, arg)))
    return result


class SamplerPool:
    """
    A process pool capable of executing operations in parallel. This differs from the multiprocessing.Pool class in
    that it explicitly incorporates process-local state.

    Every parallel function gets a GlobalNamespace object as first argument, which can hold arbitrary worker-local
    state. This allows for certain optimizations. For example, when the parallel operation requires an object that is
    expensive to transmit, we can create this object once in each process, store it in the namespace, and then use it
    in every map function call.

    This class also contains additional methods to call a function exactly once in each worker, to setup worker-local
    state.
    """

    def __init__(self, num_threads: int):
        if not isinstance(num_threads, int):
            raise pyrado.TypeErr(given=num_threads, expected_type=int)
        if num_threads < 1:
            raise pyrado.ValueErr(given=num_threads, ge_constraint="1")

        self._n_threads = num_threads
        if num_threads > 1:
            # Create workers
            self._workers = [_WorkerInfo(i + 1) for i in range(num_threads)]
            self._manager = mp.Manager()
        self._G = GlobalNamespace()

    def stop(self):
        """Terminate all workers."""
        if self._n_threads > 1:
            for w in self._workers:
                w.stop()

    def _start(self, func, *args, **kwargs):
        # Start invocation
        for w in self._workers:
            w.invoke_start(func, *args, **kwargs)

    def _await_result(self):
        return [w.invoke_wait() for w in self._workers]

    def _operation_in_progress(self):
        done = True
        for w in self._workers:
            s = w.operation_state()
            if s == _OpState.ERRORED:
                return False
            done = done and s == _OpState.DONE
        return not done

    def invoke_all(self, func, *args, **kwargs):
        """
        Invoke func on all workers using the same argument values.
        The return values are collected into a list.

        :param func: the first argument of func will be a worker-local namespace
        """
        if self._n_threads == 1:
            return [func(self._G, *args, **kwargs)]

        # Start invocation
        for w in self._workers:
            w.invoke_start(func, *args, **kwargs)
        # Await results
        return self._await_result()

    def invoke_all_map(self, func, arglist):
        """
        Invoke func(arg) on all workers using one argument from the list for each ordered worker.
        The length of the argument list must match the number of workers.
        The first argument of func will be a worker-local namespace.
        The return values are collected into a list.
        """
        assert self._n_threads == len(arglist)

        if self._n_threads == 1:
            return [func(self._G, arglist[0])]

        # Start invocation
        for w, arg in zip(self._workers, arglist):
            w.invoke_start(func, arg)
        # Await results
        return self._await_result()

    def run_map(self, func, arglist: list, progressbar: tqdm = None):
        """
        A parallel version of `[func(G, arg) for arg in arglist]`.
        There is no deterministic assignment of workers to arglist elements. Optionally runs with progress bar.

        :param func: mapper function, must be pickleable
        :param arglist: list of function args
        :param progressbar: optional progress bar from the `tqdm` library
        :return: list of results
        """
        # Set max on progress bar
        if progressbar is not None:
            progressbar.total = len(arglist)

        # Single thread optimization
        if self._n_threads == 1:
            res = []
            for arg in arglist:
                res.append(func(self._G, deepcopy(arg)))  # numpy arrays and others are passed by reference
                if progressbar:
                    progressbar.update(1)
            return res

        # Put args into a parallel queue
        argqueue = self._manager.Queue(maxsize=len(arglist))

        # Fill the queue, must be done fist to avoid race conditions
        # Add the original argument index to be able to restore it later
        for indexedArg in enumerate(arglist):
            argqueue.put(indexedArg)

        # Start workers
        self._start(_run_map, func, argqueue)

        # show progress bar if any
        if progressbar is not None:
            while self._operation_in_progress():
                # Retrieve number of remaining jobs
                remaining = argqueue.qsize()
                if remaining == 0:
                    break

                done = len(arglist) - remaining

                # Update progress (need to subtract since it's incremental)
                progressbar.update(done - progressbar.n)
                sleep(0.1)

        # Collect results in one list
        allres = self._await_result()
        result = [item for res in allres for item in res]
        # Sort results by index to ensure consistent order with args
        result.sort(key=lambda t: t[0])
        return [item for _, item in result]

    def run_collect(self, n, func, *args, collect_progressbar: tqdm = None, min_runs=None, **kwargs) -> tuple:
        """
        Collect at least n samples from func, where the number of samples per run can vary.

        This is done by calling res, ns = func(G, *args, **kwargs) until the sum of ns exceeds n.

        This is intended for situations like reinforcement learning runs. If the environment ends up
        in an error state, you get less samples per run. To ensure a stable learning behaviour, you can
        specify the minimum amount of samples to collect before returning.

        Since the workers can only check the amount of samples before starting a run, you will likely
        get more samples than the minimum. No generated samples are dropped; if that is desired,
        do so manually.

        :param n: minimum number of samples to collect
        :param func: sampler function. Must be pickleable.
        :param args: remaining positional args are passed to the function
        :param collect_progressbar: tdqm progress bar to use; default None
        :param min_runs: optionally specify a minimum amount of runs to be executed before returning
        :param kwargs: remaining keyword args are passed to the function
        :return: list of all results
        :return: total number of samples
        """

        # Set total on progress bar
        if collect_progressbar is not None:
            collect_progressbar.total = n

        if self._n_threads == 1:
            # Do locally
            result = []
            counter = 0
            while counter < n or (min_runs is not None and len(result) < min_runs):
                # Invoke once
                res, n_done = func(self._G, *args, **kwargs)

                # Add to result and record increment
                result.append(res)
                counter += n_done
                if collect_progressbar is not None:
                    collect_progressbar.update(n_done)
                # return result and total
            return result, counter

        # Create counter + counter-lock as shared vars (counter has no own lock since there is an explicit one).
        counter = self._manager.Value("i", 0)
        run_counter = self._manager.Value("i", 0)
        lock = self._manager.RLock()

        # Start async computation
        self._start(_run_collect, counter, run_counter, lock, n, min_runs, func, args, kwargs)

        # show progress bar
        if collect_progressbar is not None:
            while self._operation_in_progress():
                # Retrieve current counter value
                with lock:
                    cnt = counter.value
                if cnt >= n:
                    break

                # Update progress (need to subtract since it's incremental)
                collect_progressbar.update(cnt - collect_progressbar.n)
                sleep(0.1)

        # Collect results in one list
        allres = self._await_result()
        result = [item for res in allres for item in res]

        return result, counter.value

    def set_seed(self, seed):
        """
        Set a deterministic seed on all workers.

        :param seed: seed value for the random number generators
        """
        self.invoke_all_map(_run_set_seed, [seed + i for i in range(self._n_threads)])

    def __reduce__(self):
        # We cannot really pickle this object since it has a lot of hidden state in the worker processes
        raise RuntimeError("The sampler pool is not serializable!")

    def __del__(self):
        # stop the workers as soon as the pool is not referenced anymore
        self.stop()
