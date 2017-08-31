#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util
import sys
import zmq
from termcolor import colored

class Replier(object):
    def __init__(self, server_address):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(server_address)
        util.logger.log(colored('Protocol: zmq.REP socket bound to ' + server_address, 'yellow', attrs=['bold']))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if not self.socket.closed:
            self.socket.close()
            self.context.term()
            util.logger.log(colored('Protocol: zmq.REP socket disconnected', 'yellow', attrs=['bold']))

    def send_reply(self, reply):
        self.socket.send(reply)

    def receive_request(self):
        return self.socket.recv()


class Requester(object):
    def __init__(self, server_address):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(server_address)
        util.logger.log(colored('Protocol: zmq.REQ socket connected to server ' + server_address, 'yellow', attrs=['bold']))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if not self._socket.closed:
            self._socket.close()
            self._context.term()
            util.logger.log(colored('Protocol: zmq.REQ socket disconnected', 'yellow', attrs=['bold']))

    def send_request(self, request):
        self._socket.send(request)

    def receive_reply(self, discard_source=True):
        return self._socket.recv()
