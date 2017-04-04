#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import util
from modules import Sample, Trace

import sys
import zmq
import msgpack

class Requester(object):
    def __init__(self, server_address):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(server_address)
        util.log_print('Protocol: connected to server ' + server_address)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.socket.close()
        self.context.term()
        util.log_print('Protocol: disconnected')

    def send_request(self, request):
        self.socket.send(msgpack.packb(request))

    def receive_reply(self):
        return msgpack.unpackb(self.socket.recv(), encoding='utf-8')

    def get_batch(self, data, standardize):
        traces = []
        for i in range(len(data)):
            trace = Trace()
            data_i = data[i]
            obs_shape = data_i['observes']['shape']
            obs_data = data_i['observes']['data']
            obs = util.Tensor(obs_data).view(obs_shape)
            if standardize:
                obs = util.standardize(obs)
            trace.set_observes(obs)

            for timeStep in range(len(data_i['samples'])):
                samples_timeStep = data_i['samples'][timeStep]

                address = samples_timeStep['sample-address']
                instance = samples_timeStep['sample-instance']
                proposal_type = samples_timeStep['proposal-name']
                value = samples_timeStep['value']
                if type(value) != int:
                    util.log_error('Unsupported sample value type: ' + str(type(value)))
                value = util.Tensor([value])
                sample = Sample(address, instance, value, proposal_type)
                if proposal_type == 'discreteminmax':
                    sample.proposal_min = samples_timeStep['proposal-extra-params'][0]
                    sample.proposal_max = samples_timeStep['proposal-extra-params'][1]
                else:
                    util.log_error('Unsupported proposal distribution type: ' + proposal_type)
                trace.add_sample(sample)

            traces.append(trace)
        return traces

    def get_sub_batches(self, batch):
        sb = {}
        for trace in batch:
            h = hash(str(trace))
            if not h in sb:
                sb[h] = []
            sb[h].append(trace)
        ret = []
        for _, t in sb.items():
            ret.append(t)
        return ret

    def request_batch(self, n):
        self.send_request({'command':'new-batch', 'command-param':n})

    def receive_batch(self, standardize=True):
        sys.stdout.write('Waiting for new batch...                                 \r')
        sys.stdout.flush()
        data = self.receive_reply()
        sys.stdout.write('New batch received, processing...                        \r')
        sys.stdout.flush()
        b = self.get_batch(data, standardize)
        sys.stdout.write('New batch received, splitting into sub-batches...        \r')
        sys.stdout.flush()
        bs = self.get_sub_batches(b)
        sys.stdout.write('                                                         \r')
        sys.stdout.flush()
        return bs
