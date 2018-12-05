from collections.abc import Mapping
import shelve
import random
import time


class ConcurrentShelf(Mapping):
    def __init__(self, file_name, time_out_seconds=60):
        self._file_name = file_name
        self._time_out_seconds = time_out_seconds
        self._locked_shelf = None
        shelf = self._open(write=True)
        shelf.close()

    def __del__(self):
        if self._locked_shelf is not None:
            self._locked_shelf.close()

    def _open(self, write=False):
        flag = 'c' if write else 'r'
        start = time.time()
        while True:
            if time.time() - start > self._time_out_seconds:
                raise RuntimeError('ConcurrentShelf time out, cannot gain access to shelf on disk')
            try:
                shelf = shelve.open(self._file_name, flag=flag)
                return shelf
            except Exception as e:
                if '[Errno 11] Resource temporarily unavailable' in str(e):
                    # print('Shelf locked, waiting...')
                    time.sleep(random.uniform(0.01, 0.250))
                    next
                else:
                    raise e

    def lock(self, write=True):
        self._locked_shelf = self._open(write=write)

    def unlock(self):
        if self._locked_shelf is not None:
            self._locked_shelf.close()
            self._locked_shelf = None

    def __getitem__(self, key):
        if self._locked_shelf is not None:
            return self._locked_shelf[key]
        else:
            shelf = self._open()
            try:
                value = shelf[key]
                shelf.close()
            except Exception as e:
                shelf.close()
                raise e
            return value

    def __setitem__(self, key, value):
        if self._locked_shelf is not None:
            self._locked_shelf[key] = value
        else:
            shelf = self._open(write=True)
            try:
                shelf[key] = value
                shelf.close()
            except Exception as e:
                shelf.close()
                raise e

    def __iter__(self):
        if self._locked_shelf is not None:
            for value in self._locked_shelf:
                yield value
        else:
            shelf = self._open()
            try:
                for value in shelf:
                    yield value
                shelf.close()
            except Exception as e:
                shelf.close()
                raise e

    def __len__(self):
        if self._locked_shelf is not None:
            return len(self._locked_shelf)
        else:
            shelf = self._open()
            try:
                value = len(shelf)
                shelf.close()
            except Exception as e:
                shelf.close()
                raise e
            return value
