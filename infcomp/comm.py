#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- May 2017
#

import infcomp
import infcomp.zmq
import infcomp.pool
from infcomp import util
from infcomp.probprog import Sample, Trace, UniformDiscrete, Normal, Flip, Discrete, Categorical, UniformContinuous
import infcomp.protocol.Message
import infcomp.protocol.MessageBody
import infcomp.protocol.TracesFromPriorRequest
import infcomp.protocol.TracesFromPriorReply
import infcomp.protocol.ObservesInitRequest
import infcomp.protocol.ObservesInitReply
import infcomp.protocol.ProposalRequest
import infcomp.protocol.ProposalReply
import infcomp.protocol.Trace
import infcomp.protocol.NDArray
import infcomp.protocol.Distribution
import infcomp.protocol.UniformDiscrete
import infcomp.protocol.Normal
import infcomp.protocol.Flip
import infcomp.protocol.Discrete
import infcomp.protocol.Categorical
import infcomp.protocol.UniformContinuous

import flatbuffers
import sys
import numpy as np
import time
from collections import deque

def NDArray_to_Tensor(ndarray):
    b = ndarray._tab.Bytes
    o = flatbuffers.number_types.UOffsetTFlags.py_type(ndarray._tab.Offset(4))
    offset = ndarray._tab.Vector(o) if o != 0 else 0
    length = ndarray.DataLength()
    data_np = np.frombuffer(b, offset=offset, dtype=np.dtype('float64'), count=length)

    o = flatbuffers.number_types.UOffsetTFlags.py_type(ndarray._tab.Offset(6))
    offset = ndarray._tab.Vector(o) if o != 0 else 0
    length = ndarray.ShapeLength()
    shape_np = np.frombuffer(b, offset=offset, dtype=np.dtype('int32'), count=length)

    # print('data:', data_np)
    # print('shape', shape_np)

    data = data_np.reshape(shape_np)
    return util.Tensor(data)


def Tensor_to_NDArray(builder, tensor):
    tensor_numpy = tensor.cpu().numpy()
    data = tensor_numpy.tolist()
    shape = list(tensor_numpy.shape)

    infcomp.protocol.NDArray.NDArrayStartDataVector(builder, len(data))
    for d in reversed(data):
        builder.PrependFloat64(d)
    data = builder.EndVector(len(data))

    infcomp.protocol.NDArray.NDArrayStartShapeVector(builder, len(shape))
    for s in reversed(shape):
        builder.PrependInt32(s)
    shape = builder.EndVector(len(shape))

    infcomp.protocol.NDArray.NDArrayStart(builder)
    infcomp.protocol.NDArray.NDArrayAddData(builder, data)
    infcomp.protocol.NDArray.NDArrayAddShape(builder, shape)
    return infcomp.protocol.NDArray.NDArrayEnd(builder)

def get_message_body(message_buffer):
    message = infcomp.protocol.Message.Message.GetRootAsMessage(message_buffer, 0)
    body_type = message.BodyType()
    if body_type == infcomp.protocol.MessageBody.MessageBody().TracesFromPriorReply:
        message_body = infcomp.protocol.TracesFromPriorReply.TracesFromPriorReply()
    elif body_type == infcomp.protocol.MessageBody.MessageBody().ObservesInitRequest:
        message_body = infcomp.protocol.ObservesInitRequest.ObservesInitRequest()
    elif body_type == infcomp.protocol.MessageBody.MessageBody().ProposalRequest:
        message_body = infcomp.protocol.ProposalRequest.ProposalRequest()
    else:
        log_error('Unexpected body: MessageBody id: {0}'.format(bodyType))
    message_body.Init(message.Body().Bytes, message.Body().Pos)
    return message_body

def get_sample(s):
    sample = Sample()
    sample.address = s.Address().decode("utf-8")
    sample.instance = s.Instance()
    value = s.Value()
    if not value is None:
        sample.value = NDArray_to_Tensor(value)
    distribution_type = s.DistributionType()
    if distribution_type != infcomp.protocol.Distribution.Distribution().NONE:
        if distribution_type == infcomp.protocol.Distribution.Distribution().UniformDiscrete:
            p = infcomp.protocol.UniformDiscrete.UniformDiscrete()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = UniformDiscrete(p.PriorMin(), p.PriorSize())
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Normal:
            p = infcomp.protocol.Normal.Normal()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = Normal(p.PriorMean(), p.PriorStd())
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Flip:
            # p = infcomp.protocol.Flip.Flip()
            # p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = Flip()
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Discrete:
            p = infcomp.protocol.Discrete.Discrete()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = Discrete(p.PriorSize())
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Categorical:
            p = infcomp.protocol.Categorical.Categorical()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = Categorical(p.PriorSize())
        elif distribution_type == infcomp.protocol.Distribution.Distribution().UniformContinuous:
            p = infcomp.protocol.UniformContinuous.UniformContinuous()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = UniformContinuous(p.PriorMin(), p.PriorMax())
        else:
            util.log_error('Unknown distribution:Distribution id: {0}.'.format(distribution_type))
    return sample

class BatchRequester(object):
    def __init__(self, data_source, standardize=False, batch_pool=False, request_ahead=128):
        if batch_pool:
            self.requester = infcomp.pool.Requester(data_source)
        else:
            self.requester = infcomp.zmq.Requester(data_source)
        self.standardize = standardize
        self.request_ahead = request_ahead
        self.queue = deque([])
        self.request_traces(self.request_ahead)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.requester.close()

    def get_traces(self, n):
        t0 = time.time()
        while len(self.queue) < n:
            self.receive_traces_to_queue()
            self.request_traces(n)
        ret = [self.queue.popleft() for i in range(n)]
        return ret, time.time() - t0

    def receive_traces_to_queue(self):
        sys.stdout.write('Waiting for traces from prior...                         \r')
        sys.stdout.flush()
        data = self.requester.receive_reply()
        sys.stdout.write('Processing new traces from prior...                      \r')
        sys.stdout.flush()
        traces = self.read_traces(data)
        sys.stdout.write('                                                         \r')
        sys.stdout.flush()
        self.queue.extend(traces)

    def read_traces(self, data):
        message_body = get_message_body(data)
        if not isinstance(message_body, infcomp.protocol.TracesFromPriorReply.TracesFromPriorReply):
            util.log_error('Expecting a TracesFromPriorReply, but received {0}'.format(message_body))

        traces_length = message_body.TracesLength()
        traces = []
        for i in range(traces_length):
            trace = Trace()

            t = message_body.Traces(i)
            obs = NDArray_to_Tensor(t.Observes())
            if self.standardize:
                obs = util.standardize(obs)
            trace.set_observes(obs)

            samples_length = t.SamplesLength()
            for timeStep in range(samples_length):
                s = t.Samples(timeStep)
                sample = get_sample(s)
                trace.add_sample(sample)

            traces.append(trace)

        return traces

    def request_traces(self, n):
        # allocate buffer
        builder = flatbuffers.Builder(64) # actual message is around 36 bytes

        # construct message body
        infcomp.protocol.TracesFromPriorRequest.TracesFromPriorRequestStart(builder)
        infcomp.protocol.TracesFromPriorRequest.TracesFromPriorRequestAddNumTraces(builder, n)
        message_body = infcomp.protocol.TracesFromPriorRequest.TracesFromPriorRequestEnd(builder)

        # construct message
        infcomp.protocol.Message.MessageStart(builder)
        infcomp.protocol.Message.MessageAddBodyType(builder, infcomp.protocol.MessageBody.MessageBody().TracesFromPriorRequest)
        infcomp.protocol.Message.MessageAddBody(builder, message_body)
        message = infcomp.protocol.Message.MessageEnd(builder)
        builder.Finish(message)

        message = builder.Output()
        self.requester.send_request(message)


class ProposalReplier(object):
    def __init__(self, server_address):
        self.replier = infcomp.zmq.Replier(server_address)
        self.new_trace = False
        self.observes = None
        self.current_sample = None
        self.previous_sample = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.replier.close()

    def receive_request(self, standardize):
        data = self.replier.receive_request()
        message_body = get_message_body(data)
        if isinstance(message_body, infcomp.protocol.ObservesInitRequest.ObservesInitRequest):
            self.observes = NDArray_to_Tensor(message_body.Observes())
            if standardize:
                self.observes = util.standardize(self.observes)
            self.new_trace = True
        elif isinstance(message_body, infcomp.protocol.ProposalRequest.ProposalRequest):
            current_sample = message_body.CurrentSample()
            previous_sample = message_body.PreviousSample()
            self.current_sample = get_sample(current_sample)
            self.previous_sample = get_sample(previous_sample)
            self.new_trace = False
        else:
            util.log_error('Expecting ObservesInitRequest or ProposalRequest, but received {0}'.format(message_body))

    def reply_observes_received(self):
        # allocate buffer
        builder = flatbuffers.Builder(64)

        # construct message body
        infcomp.protocol.ObservesInitReply.ObservesInitReplyStart(builder)
        infcomp.protocol.ObservesInitReply.ObservesInitReplyAddSuccess(builder, True)
        message_body = infcomp.protocol.ObservesInitReply.ObservesInitReplyEnd(builder)

        # construct message
        infcomp.protocol.Message.MessageStart(builder)
        infcomp.protocol.Message.MessageAddBodyType(builder, infcomp.protocol.MessageBody.MessageBody().ObservesInitReply)
        infcomp.protocol.Message.MessageAddBody(builder, message_body)
        message = infcomp.protocol.Message.MessageEnd(builder)
        builder.Finish(message)

        message = builder.Output()
        self.replier.send_reply(message)

    def reply_proposal(self, success, p):
        # allocate buffer
        builder = flatbuffers.Builder(64)

        if not success:
            infcomp.protocol.ProposalReply.ProposalReplyStart(builder)
            infcomp.protocol.ProposalReply.ProposalReplyAddSuccess(builder, False)
            message_body = infcomp.protocol.ProposalReply.ProposalReplyEnd(builder)
        else:
            if isinstance(p, UniformDiscrete):
                # construct probabilities
                proposal_probabilities = Tensor_to_NDArray(builder, p.proposal_probabilities)
                # construct UniformDiscrete
                infcomp.protocol.UniformDiscrete.UniformDiscreteStart(builder)
                infcomp.protocol.UniformDiscrete.UniformDiscreteAddPriorMin(builder, p.prior_min)
                infcomp.protocol.UniformDiscrete.UniformDiscreteAddPriorSize(builder, p.prior_size)
                infcomp.protocol.UniformDiscrete.UniformDiscreteAddProposalProbabilities(builder, proposal_probabilities)
                distribution = infcomp.protocol.UniformDiscrete.UniformDiscreteEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().UniformDiscrete
            elif isinstance(p, Normal):
                # construct Normal
                infcomp.protocol.Normal.NormalStart(builder)
                infcomp.protocol.Normal.NormalAddPriorMean(builder, p.prior_mean)
                infcomp.protocol.Normal.NormalAddPriorStd(builder, p.prior_std)
                infcomp.protocol.Normal.NormalAddProposalMean(builder, p.proposal_mean)
                infcomp.protocol.Normal.NormalAddProposalStd(builder, p.proposal_std)
                distribution = infcomp.protocol.Normal.NormalEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Normal
            elif isinstance(p, Flip):
                # construct Flip
                infcomp.protocol.Flip.FlipStart(builder)
                infcomp.protocol.Flip.FlipAddProposalProbability(builder, p.proposal_probability)
                distribution = infcomp.protocol.Flip.FlipEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Flip
            elif isinstance(p, Discrete):
                # construct probabilities
                proposal_probabilities = Tensor_to_NDArray(builder, p.proposal_probabilities)
                # construct Discrete
                infcomp.protocol.Discrete.DiscreteStart(builder)
                infcomp.protocol.Discrete.DiscreteAddPriorSize(builder, p.prior_size)
                infcomp.protocol.Discrete.DiscreteAddProposalProbabilities(builder, proposal_probabilities)
                distribution = infcomp.protocol.Discrete.DiscreteEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Discrete
            elif isinstance(p, Categorical):
                # construct probabilities
                proposal_probabilities = Tensor_to_NDArray(builder, p.proposal_probabilities)
                # construct Categorical
                infcomp.protocol.Categorical.CategoricalStart(builder)
                infcomp.protocol.Categorical.CategoricalAddPriorSize(builder, p.prior_size)
                infcomp.protocol.Categorical.CategoricalAddProposalProbabilities(builder, proposal_probabilities)
                distribution = infcomp.protocol.Categorical.CategoricalEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Categorical
            elif isinstance(p, UniformContinuous):
                # construct UniformContinuous
                infcomp.protocol.UniformContinuous.UniformContinuousStart(builder)
                infcomp.protocol.UniformContinuous.UniformContinuousAddPriorMin(builder, p.prior_min)
                infcomp.protocol.UniformContinuous.UniformContinuousAddPriorMax(builder, p.prior_max)
                infcomp.protocol.UniformContinuous.UniformContinuousAddProposalMode(builder, p.proposal_mode)
                infcomp.protocol.UniformContinuous.UniformContinuousAddProposalCertainty(builder, p.proposal_certainty)
                distribution = infcomp.protocol.UniformContinuous.UniformContinuousEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().UniformContinuous
            else:
                util.log_error('Unsupported proposal distribution: {0}'.format(p))

            # construct message body (ProposalReply)
            infcomp.protocol.ProposalReply.ProposalReplyStart(builder)
            infcomp.protocol.ProposalReply.ProposalReplyAddSuccess(builder, True)
            infcomp.protocol.ProposalReply.ProposalReplyAddDistributionType(builder, distribution_type)
            infcomp.protocol.ProposalReply.ProposalReplyAddDistribution(builder, distribution)
            message_body = infcomp.protocol.ProposalReply.ProposalReplyEnd(builder)

        # construct message
        infcomp.protocol.Message.MessageStart(builder)
        infcomp.protocol.Message.MessageAddBodyType(builder, infcomp.protocol.MessageBody.MessageBody().ProposalReply)
        infcomp.protocol.Message.MessageAddBody(builder, message_body)
        message = infcomp.protocol.Message.MessageEnd(builder)

        builder.Finish(message)
        message = builder.Output()
        self.replier.send_reply(message)
