#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
import pyprob.zmq
import pyprob.pool
from pyprob import util
from pyprob.trace import Sample, Trace
from pyprob.distributions import UniformDiscrete, Normal, Flip, Discrete, Categorical, UniformContinuous, UniformContinuousAlt, Laplace, Gamma, Beta, MultivariateNormal
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
import infcomp.protocol.UniformContinuousAlt
import infcomp.protocol.Laplace
import infcomp.protocol.Gamma
import infcomp.protocol.Beta
import infcomp.protocol.MultivariateNormal

import flatbuffers
import sys
import numpy as np
import time
from collections import deque

def NDArray_to_Tensor(ndarray):
    if ndarray is None:
        # util.log_warning('NDArray_to_Tensor: empty NDArray received, returning empty tensor')
        return util.Tensor()
    else:
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
    data = tensor_numpy.flatten().tolist()
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
        log_error('get_message_body: Unexpected body: MessageBody id: {0}'.format(bodyType))
    message_body.Init(message.Body().Bytes, message.Body().Pos)
    return message_body

def get_sample(s):
    address = s.Address().decode("utf-8")
    distribution = None
    # sample.instance = s.Instance()
    value = NDArray_to_Tensor(s.Value())
    distribution_type = s.DistributionType()
    if distribution_type != infcomp.protocol.Distribution.Distribution().NONE:
        if distribution_type == infcomp.protocol.Distribution.Distribution().UniformDiscrete:
            p = infcomp.protocol.UniformDiscrete.UniformDiscrete()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            distribution = UniformDiscrete(p.PriorMin(), p.PriorSize())
            if value.dim() > 0:
                value = util.one_hot(distribution.prior_size, int(value[0]) - distribution.prior_min)
        elif distribution_type == infcomp.protocol.Distribution.Distribution().MultivariateNormal:
            p = infcomp.protocol.MultivariateNormal.MultivariateNormal()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            distribution = MultivariateNormal(NDArray_to_Tensor(p.PriorMean()), NDArray_to_Tensor(p.PriorCov()))
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Normal:
            p = infcomp.protocol.Normal.Normal()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            distribution = Normal(p.PriorMean(), p.PriorStd())
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Flip:
            distribution = Flip()
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Discrete:
            p = infcomp.protocol.Discrete.Discrete()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            distribution = Discrete(p.PriorSize())
            if value.dim() > 0:
                value = util.one_hot(distribution.prior_size, int(value[0]))
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Categorical:
            p = infcomp.protocol.Categorical.Categorical()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            distribution = Categorical(p.PriorSize())
            if value.dim() > 0:
                value = util.one_hot(distribution.prior_size, int(value[0]))
        elif distribution_type == infcomp.protocol.Distribution.Distribution().UniformContinuous:
            p = infcomp.protocol.UniformContinuous.UniformContinuous()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            distribution = UniformContinuous(p.PriorMin(), p.PriorMax())
        elif distribution_type == infcomp.protocol.Distribution.Distribution().UniformContinuousAlt:
            p = infcomp.protocol.UniformContinuousAlt.UniformContinuousAlt()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            distribution = UniformContinuousAlt(p.PriorMin(), p.PriorMax())
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Laplace:
            p = infcomp.protocol.Laplace.Laplace()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            distribution = Laplace(p.PriorLocation(), p.PriorScale())
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Gamma:
            distribution = Gamma()
        elif distribution_type == infcomp.protocol.Distribution.Distribution().Beta:
            distribution = Beta()
        else:
            util.log_error('get_sample: Unknown distribution:Distribution id: {0}.'.format(distribution_type))
    sample = Sample(address, distribution, value)
    return sample

class BatchRequester(object):
    def __init__(self, data_source, standardize=False, batch_pool=False, request_ahead=256):
        if batch_pool:
            self._requester = pyprob.pool.Requester(data_source)
        else:
            self._requester = pyprob.zmq.Requester(data_source)
        self._standardize = standardize
        self._request_ahead = request_ahead
        self._queue = deque([])
        self.request_traces(self._request_ahead)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._requester.close()

    def __del__(self):
        self._requester.close()

    def get_traces(self, n, discard_source=False):
        t0 = time.time()
        while len(self._queue) < n:
            self.receive_traces_to_queue(discard_source)
            self.request_traces(n)
        ret = [self._queue.popleft() for i in range(n)]
        return ret, time.time() - t0

    def receive_traces_to_queue(self, discard_source=False):
        sys.stdout.write('Waiting for traces from prior...                         \r')
        sys.stdout.flush()
        data = self._requester.receive_reply(discard_source)
        sys.stdout.write('Processing new traces from prior...                      \r')
        sys.stdout.flush()
        traces = self.read_traces(data)
        sys.stdout.write('                                                         \r')
        sys.stdout.flush()
        self._queue.extend(traces)

    def read_traces(self, data):
        message_body = get_message_body(data)
        if not isinstance(message_body, infcomp.protocol.TracesFromPriorReply.TracesFromPriorReply):
            util.logger.log_error('read_traces: Expecting a TracesFromPriorReply, but received {0}'.format(message_body))

        traces_length = message_body.TracesLength()
        traces = []
        for i in range(traces_length):
            trace = Trace()

            t = message_body.Traces(i)
            obs = NDArray_to_Tensor(t.Observes())
            if self._standardize:
                obs = util.standardize(obs)
            trace.set_observes_tensor(obs)

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
        self._requester.send_request(message)

class ProposalReplier(object):
    def __init__(self, server_address):
        self._replier = pyprob.zmq.Replier(server_address)
        self.new_trace = False
        self.observes = None
        self.current_sample = None
        self.previous_sample = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._replier.close()

    def __del__(self):
        self._replier.close()

    def receive_request(self, standardize):
        data = self._replier.receive_request()
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
            util.logger.log_error('receive_request: Expecting ObservesInitRequest or ProposalRequest, but received {0}'.format(message_body))

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
        self._replier.send_reply(message)

    def reply_proposal(self, proposal=None):
        # allocate buffer
        builder = flatbuffers.Builder(64)

        if proposal is None:
            infcomp.protocol.ProposalReply.ProposalReplyStart(builder)
            infcomp.protocol.ProposalReply.ProposalReplyAddSuccess(builder, False)
            message_body = infcomp.protocol.ProposalReply.ProposalReplyEnd(builder)
        else:
            if isinstance(proposal, UniformDiscrete):
                # construct probabilities
                proposal_probabilities = Tensor_to_NDArray(builder, proposal.proposal_probabilities)
                # construct UniformDiscrete
                infcomp.protocol.UniformDiscrete.UniformDiscreteStart(builder)
                infcomp.protocol.UniformDiscrete.UniformDiscreteAddPriorMin(builder, proposal.prior_min)
                infcomp.protocol.UniformDiscrete.UniformDiscreteAddPriorSize(builder, proposal.prior_size)
                infcomp.protocol.UniformDiscrete.UniformDiscreteAddProposalProbabilities(builder, proposal_probabilities)
                distribution = infcomp.protocol.UniformDiscrete.UniformDiscreteEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().UniformDiscrete
            elif isinstance(proposal, Normal):
                # construct Normal
                infcomp.protocol.Normal.NormalStart(builder)
                infcomp.protocol.Normal.NormalAddPriorMean(builder, proposal.prior_mean)
                infcomp.protocol.Normal.NormalAddPriorStd(builder, proposal.prior_std)
                infcomp.protocol.Normal.NormalAddProposalMean(builder, proposal.proposal_mean)
                infcomp.protocol.Normal.NormalAddProposalStd(builder, proposal.proposal_std)
                distribution = infcomp.protocol.Normal.NormalEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Normal
            elif isinstance(proposal, Flip):
                # construct Flip
                infcomp.protocol.Flip.FlipStart(builder)
                infcomp.protocol.Flip.FlipAddProposalProbability(builder, proposal.proposal_probability)
                distribution = infcomp.protocol.Flip.FlipEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Flip
            elif isinstance(proposal, Discrete):
                # construct probabilities
                proposal_probabilities = Tensor_to_NDArray(builder, proposal.proposal_probabilities)
                # construct Discrete
                infcomp.protocol.Discrete.DiscreteStart(builder)
                infcomp.protocol.Discrete.DiscreteAddPriorSize(builder, proposal.prior_size)
                infcomp.protocol.Discrete.DiscreteAddProposalProbabilities(builder, proposal_probabilities)
                distribution = infcomp.protocol.Discrete.DiscreteEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Discrete
            elif isinstance(proposal, Categorical):
                # construct probabilities
                proposal_probabilities = Tensor_to_NDArray(builder, proposal.proposal_probabilities)
                # construct Categorical
                infcomp.protocol.Categorical.CategoricalStart(builder)
                infcomp.protocol.Categorical.CategoricalAddPriorSize(builder, proposal.prior_size)
                infcomp.protocol.Categorical.CategoricalAddProposalProbabilities(builder, proposal_probabilities)
                distribution = infcomp.protocol.Categorical.CategoricalEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Categorical
            elif isinstance(proposal, UniformContinuous):
                # construct UniformContinuous
                infcomp.protocol.UniformContinuous.UniformContinuousStart(builder)
                infcomp.protocol.UniformContinuous.UniformContinuousAddPriorMin(builder, proposal.prior_min)
                infcomp.protocol.UniformContinuous.UniformContinuousAddPriorMax(builder, proposal.prior_max)
                infcomp.protocol.UniformContinuous.UniformContinuousAddProposalMode(builder, proposal.proposal_mode)
                infcomp.protocol.UniformContinuous.UniformContinuousAddProposalCertainty(builder, proposal.proposal_certainty)
                distribution = infcomp.protocol.UniformContinuous.UniformContinuousEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().UniformContinuous
            elif isinstance(proposal, UniformContinuousAlt):
                # construct proposal parameters
                # print('means, stds, coeffs')
                # print(proposal.proposal_means)
                # print(proposal.proposal_stds)
                # print(proposal.proposal_coeffs)
                proposal_means = Tensor_to_NDArray(builder, proposal.proposal_means)
                proposal_stds = Tensor_to_NDArray(builder, proposal.proposal_stds)
                proposal_coeffs = Tensor_to_NDArray(builder, proposal.proposal_coeffs)
                # construct UniformContinuousAlt
                infcomp.protocol.UniformContinuousAlt.UniformContinuousAltStart(builder)
                infcomp.protocol.UniformContinuousAlt.UniformContinuousAltAddPriorMin(builder, proposal.prior_min)
                infcomp.protocol.UniformContinuousAlt.UniformContinuousAltAddPriorMax(builder, proposal.prior_max)
                infcomp.protocol.UniformContinuousAlt.UniformContinuousAltAddProposalMeans(builder, proposal_means)
                infcomp.protocol.UniformContinuousAlt.UniformContinuousAltAddProposalStds(builder, proposal_stds)
                infcomp.protocol.UniformContinuousAlt.UniformContinuousAltAddProposalCoeffs(builder, proposal_coeffs)
                distribution = infcomp.protocol.UniformContinuousAlt.UniformContinuousAltEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().UniformContinuousAlt
            elif isinstance(proposal, Laplace):
                # construct Laplace
                infcomp.protocol.Laplace.LaplaceStart(builder)
                infcomp.protocol.Laplace.LaplaceAddPriorLocation(builder, proposal.prior_location)
                infcomp.protocol.Laplace.LaplaceAddPriorScale(builder, proposal.prior_scale)
                infcomp.protocol.Laplace.LaplaceAddProposalLocation(builder, proposal.proposal_location)
                infcomp.protocol.Laplace.LaplaceAddProposalScale(builder, proposal.proposal_scale)
                distribution = infcomp.protocol.Laplace.LaplaceEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Laplace
            elif isinstance(proposal, Gamma):
                # construct Gamma
                infcomp.protocol.Gamma.GammaStart(builder)
                infcomp.protocol.Gamma.GammaAddProposalLocation(builder, proposal.proposal_location)
                infcomp.protocol.Gamma.GammaAddProposalScale(builder, proposal.proposal_scale)
                distribution = infcomp.protocol.Gamma.GammaEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Gamma
            elif isinstance(proposal, Beta):
                # construct Beta
                infcomp.protocol.Beta.BetaStart(builder)
                infcomp.protocol.Beta.BetaAddProposalMode(builder, proposal.proposal_mode)
                infcomp.protocol.Beta.BetaAddProposalCertainty(builder, proposal.proposal_certainty)
                distribution = infcomp.protocol.Beta.BetaEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().Beta
            elif isinstance(proposal, MultivariateNormal):
                # construct prior_mean, prior_cov, proposal_mean, proposal_vars
                prior_mean = Tensor_to_NDArray(builder, proposal.prior_mean)
                prior_cov = Tensor_to_NDArray(builder, proposal.prior_cov)
                proposal_mean = Tensor_to_NDArray(builder, proposal.proposal_mean)
                proposal_vars = Tensor_to_NDArray(builder, proposal.proposal_vars)
                # construct MultivariateNormal
                infcomp.protocol.MultivariateNormal.MultivariateNormalStart(builder)
                infcomp.protocol.MultivariateNormal.MultivariateNormalAddPriorMean(builder, prior_mean)
                infcomp.protocol.MultivariateNormal.MultivariateNormalAddPriorCov(builder, prior_cov)
                infcomp.protocol.MultivariateNormal.MultivariateNormalAddProposalMean(builder, proposal_mean)
                infcomp.protocol.MultivariateNormal.MultivariateNormalAddProposalVars(builder, proposal_vars)
                distribution = infcomp.protocol.MultivariateNormal.MultivariateNormalEnd(builder)
                distribution_type = infcomp.protocol.Distribution.Distribution().MultivariateNormal
            else:
                util.logger.log_error('reply_proposal: Unsupported proposal distribution: {0}'.format(proposal))

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
        self._replier.send_reply(message)
