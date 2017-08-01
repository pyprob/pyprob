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
        util.log_print(colored('Protocol: zmq.REP socket bound to ' + server_address, 'yellow', attrs=['bold']))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        self.socket.close()
        self.context.term()
        util.log_print(colored('Protocol: zmq.REP socket disconnected', 'yellow', attrs=['bold']))

    def send_reply(self, reply):
        self.socket.send(reply)

    def receive_request(self):
        return self.socket.recv()


class Requester(object):
    def __init__(self, server_address):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(server_address)
        util.log_print(colored('Protocol: zmq.REQ socket connected to server ' + server_address, 'yellow', attrs=['bold']))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        self.socket.close()
        self.context.term()
        util.log_print(colored('Protocol: zmq.REQ socket disconnected', 'yellow', attrs=['bold']))

    def send_request(self, request):
        self.socket.send(request)

    def receive_reply(self, discard_source=True):
        return self.socket.recv()
