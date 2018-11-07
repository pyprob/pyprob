import shelve


class AddressDictionary():
    def __init__(self, file_name):
        self._file_name = file_name
        self._closed = False
        self._shelf = shelve.open(self._file_name, writeback=True)
        if '__last_id' not in self._shelf:
            self._shelf['__last_id'] = 0
            self._length = 0

    def __len__(self):
        return self._length

    @property
    def length(self):
        return self._length

    def __del__(self):
        self.close()

    def close(self):
        if not self._closed:
            self._shelf.close()
            self._closed = True

    def address_to_id(self, address):
        address_key = '__address_' + address
        if address_key in self._shelf:
            return self._shelf[address_key]
        else:
            new_id = self._shelf['__last_id'] + 1
            self._length = new_id
            self._shelf['__last_id'] = new_id
            new_id = 'A{}'.format(new_id)
            self._shelf[address_key] = new_id
            id_key = '__id_' + new_id
            self._shelf[id_key] = address
            self._shelf.sync()
            return new_id

    def id_to_address(self, id):
        id_key = '__id_' + id
        return self._shelf[id_key]
