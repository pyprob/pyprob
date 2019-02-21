from collections.abc import Mapping
import shelve
import random
import time
import collections


class ConcurrentShelf(Mapping):
    capacity = 32
    cache = collections.OrderedDict()
    def __init__(self, file_name, time_out_seconds=60):
        self._file_name = file_name

    def _open(self):
# idea from https://www.kunxi.org/2014/05/lru-cache-in-python
        try:
            shelf = ConcurrentShelf.cache.pop(self._file_name)
            # it was in the cache, put it back on the front
            ConcurrentShelf.cache[self._file_name] = shelf
            return shelf
        except KeyError:
            # not in the cache
            if len(ConcurrentShelf.cache) >= ConcurrentShelf.capacity:
                # cache is full, delete the last entry
                _,s = ConcurrentShelf.cache.popitem(last=False)
                s.close()
            shelf = shelve.open(self._file_name, flag='r')
            ConcurrentShelf.cache[self._file_name] = shelf
            return shelf

    def lock(self, write=True):
        pass

    def unlock(self):
        pass

    def __getitem__(self, key):
        shelf = self._open()
        return shelf[key]

    def __iter__(self):
        shelf = self._open()
        for value in shelf:
                yield value
    def __len__(self):
        shelf = self._open()
        return len(shelf)
