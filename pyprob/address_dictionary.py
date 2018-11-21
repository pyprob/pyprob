from functools import lru_cache

from .concurrency import ConcurrentShelf


class AddressDictionary():
    def __init__(self, file_name):
        self._file_name = file_name
        self._closed = False
        self._shelf = None
        self._shelf = ConcurrentShelf(file_name)
        self._shelf.lock()
        if '__last_id' not in self._shelf:
            self._shelf['__last_id'] = 0
        self._shelf.unlock()

    @lru_cache(maxsize=4096)
    def address_to_id(self, address):
        address_key = '__address__' + address
        if address_key in self._shelf:
            return self._shelf[address_key]
        else:
            self._shelf.lock()
            new_id = self._shelf['__last_id'] + 1
            self._shelf['__last_id'] = new_id
            new_id = '__A{}'.format(new_id)
            self._shelf[address_key] = new_id
            id_key = '__id__' + new_id
            self._shelf[id_key] = address
            self._shelf.unlock()
            return new_id

    @lru_cache(maxsize=4096)
    def id_to_address(self, id):
        id_key = '__id__' + id
        return self._shelf[id_key]
