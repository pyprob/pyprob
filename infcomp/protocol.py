import infcomp
import infcomp.zmq
from infcomp import util
from infcomp.probprog import Sample, Trace, UniformDiscrete, Normal, Flip, Discrete, Categorical
import infcomp.flatbuffers.Message
import infcomp.flatbuffers.MessageBody
import infcomp.flatbuffers.TracesFromPriorRequest
import infcomp.flatbuffers.TracesFromPriorReply
import infcomp.flatbuffers.ObservesInitRequest
import infcomp.flatbuffers.ObservesInitReply
import infcomp.flatbuffers.ProposalRequest
import infcomp.flatbuffers.ProposalReply
import infcomp.flatbuffers.Trace
import infcomp.flatbuffers.NDArray
import infcomp.flatbuffers.Distribution
import infcomp.flatbuffers.UniformDiscrete
import infcomp.flatbuffers.Normal
import infcomp.flatbuffers.Flip
import infcomp.flatbuffers.Discrete
import infcomp.flatbuffers.Categorical

import flatbuffers
import sys

def NDArray_to_Tensor(ndarray):
    data = [ndarray.Data(i) for i in range(ndarray.DataLength())]
    shape = [ndarray.Shape(i) for i in range(ndarray.ShapeLength())]
    return util.Tensor(data).view(shape)

def Tensor_to_NDArray(builder, tensor):
    tensor_numpy = tensor.cpu().numpy()
    data = tensor_numpy.tolist()
    shape = list(tensor_numpy.shape)

    infcomp.flatbuffers.NDArray.NDArrayStartDataVector(builder, len(data))
    for d in reversed(data):
        builder.PrependFloat64(d)
    data = builder.EndVector(len(data))

    infcomp.flatbuffers.NDArray.NDArrayStartShapeVector(builder, len(shape))
    for s in reversed(shape):
        builder.PrependInt32(s)
    shape = builder.EndVector(len(shape))

    infcomp.flatbuffers.NDArray.NDArrayStart(builder)
    infcomp.flatbuffers.NDArray.NDArrayAddData(builder, data)
    infcomp.flatbuffers.NDArray.NDArrayAddShape(builder, shape)
    return infcomp.flatbuffers.NDArray.NDArrayEnd(builder)

def get_message_body(message_buffer):
    message = infcomp.flatbuffers.Message.Message.GetRootAsMessage(message_buffer, 0)
    body_type = message.BodyType()
    if body_type == infcomp.flatbuffers.MessageBody.MessageBody().TracesFromPriorReply:
        message_body = infcomp.flatbuffers.TracesFromPriorReply.TracesFromPriorReply()
    elif body_type == infcomp.flatbuffers.MessageBody.MessageBody().ObservesInitRequest:
        message_body = infcomp.flatbuffers.ObservesInitRequest.ObservesInitRequest()
    elif body_type == infcomp.flatbuffers.MessageBody.MessageBody().ProposalRequest:
        message_body = infcomp.flatbuffers.ProposalRequest.ProposalRequest()
    else:
        log_error('Unexpected body:MessageBody id: {0}'.format(bodyType))
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
    if distribution_type != infcomp.flatbuffers.Distribution.Distribution().NONE:
        if distribution_type == infcomp.flatbuffers.Distribution.Distribution().UniformDiscrete:
            p = infcomp.flatbuffers.UniformDiscrete.UniformDiscrete()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = UniformDiscrete(p.PriorMin(), p.PriorSize())
        elif distribution_type == infcomp.flatbuffers.Distribution.Distribution().Normal:
            # p = infcomp.flatbuffers.Normal.Normal()
            # p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = Normal()
        elif distribution_type == infcomp.flatbuffers.Distribution.Distribution().Flip:
            # p = infcomp.flatbuffers.Flip.Flip()
            # p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = Flip()
        elif distribution_type == infcomp.flatbuffers.Distribution.Distribution().Discrete:
            p = infcomp.flatbuffers.Discrete.Discrete()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = Discrete(p.PriorSize())
        elif distribution_type == infcomp.flatbuffers.Distribution.Distribution().Categorical:
            p = infcomp.flatbuffers.Categorical.Categorical()
            p.Init(s.Distribution().Bytes, s.Distribution().Pos)
            sample.distribution = Categorical(p.PriorSize())
        else:
            util.log_error('Unknown distribution:Distribution id: {0}.'.format(distribution_type))
    return sample

class BatchRequester(object):
    def __init__(self, server_address):
        self.requester = infcomp.zmq.Requester(server_address)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.requester.close()

    def get_batch(self, data, standardize):
        message_body = get_message_body(data)
        if not isinstance(message_body, infcomp.flatbuffers.TracesFromPriorReply.TracesFromPriorReply):
            util.log_error('Expecting a TracesFromPriorReply, but received {0}'.format(message_body))

        traces_length = message_body.TracesLength()
        traces = []
        for i in range(traces_length):
            trace = Trace()

            t = message_body.Traces(i)
            obs = NDArray_to_Tensor(t.Observes())
            if standardize:
                obs = util.standardize(obs)
            trace.set_observes(obs)

            samples_length = t.SamplesLength()
            for timeStep in range(samples_length):
                s = t.Samples(timeStep)
                sample = get_sample(s)
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
        # allocate buffer
        builder = flatbuffers.Builder(64) # actual message is around 36 bytes

        # construct message body
        infcomp.flatbuffers.TracesFromPriorRequest.TracesFromPriorRequestStart(builder)
        infcomp.flatbuffers.TracesFromPriorRequest.TracesFromPriorRequestAddNumTraces(builder, n)
        message_body = infcomp.flatbuffers.TracesFromPriorRequest.TracesFromPriorRequestEnd(builder)

        # construct message
        infcomp.flatbuffers.Message.MessageStart(builder)
        infcomp.flatbuffers.Message.MessageAddBodyType(builder, infcomp.flatbuffers.MessageBody.MessageBody().TracesFromPriorRequest)
        infcomp.flatbuffers.Message.MessageAddBody(builder, message_body)
        message = infcomp.flatbuffers.Message.MessageEnd(builder)
        builder.Finish(message)

        message = builder.Output()
        self.requester.send_request(message)

    def receive_batch(self, standardize=False):
        sys.stdout.write('Waiting for new batch...                                 \r')
        sys.stdout.flush()
        data = self.requester.receive_reply()
        sys.stdout.write('New batch received, processing...                        \r')
        sys.stdout.flush()
        b = self.get_batch(data, standardize)
        sys.stdout.write('New batch received, splitting into sub-batches...        \r')
        sys.stdout.flush()
        bs = self.get_sub_batches(b)
        sys.stdout.write('                                                         \r')
        sys.stdout.flush()
        return bs


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
        if isinstance(message_body, infcomp.flatbuffers.ObservesInitRequest.ObservesInitRequest):
            self.observes = NDArray_to_Tensor(message_body.Observes())
            if standardize:
                self.observes = util.standardize(self.observes)
            self.new_trace = True
        elif isinstance(message_body, infcomp.flatbuffers.ProposalRequest.ProposalRequest):
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
        infcomp.flatbuffers.ObservesInitReply.ObservesInitReplyStart(builder)
        infcomp.flatbuffers.ObservesInitReply.ObservesInitReplyAddSuccess(builder, True)
        message_body = infcomp.flatbuffers.ObservesInitReply.ObservesInitReplyEnd(builder)

        # construct message
        infcomp.flatbuffers.Message.MessageStart(builder)
        infcomp.flatbuffers.Message.MessageAddBodyType(builder, infcomp.flatbuffers.MessageBody.MessageBody().ObservesInitReply)
        infcomp.flatbuffers.Message.MessageAddBody(builder, message_body)
        message = infcomp.flatbuffers.Message.MessageEnd(builder)
        builder.Finish(message)

        message = builder.Output()
        self.replier.send_reply(message)

    def reply_proposal(self, p):
        # allocate buffer
        builder = flatbuffers.Builder(64)

        if isinstance(p, UniformDiscrete):
            # construct probabilities
            proposal_probabilities = Tensor_to_NDArray(builder, p.proposal_probabilities)

            # construct UniformDiscrete
            infcomp.flatbuffers.UniformDiscrete.UniformDiscreteStart(builder)
            infcomp.flatbuffers.UniformDiscrete.UniformDiscreteAddProposalProbabilities(builder, proposal_probabilities)
            distribution = infcomp.flatbuffers.UniformDiscrete.UniformDiscrete(builder)

            # construct message body (ProposalReply)
            infcomp.flatbuffers.ProposalReply.ProposalReplyStart(builder)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddSuccess(builder, True)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddDistributionType(builder, infcomp.flatbuffers.Distribution.Distribution().UniformDiscrete)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddDistribution(builder, distribution)
            message_body = infcomp.flatbuffers.ProposalReply.ProposalReplyEnd(builder)
        elif isinstance(p, Normal):
            # construct Normal
            infcomp.flatbuffers.Normal.NormalStart(builder)
            infcomp.flatbuffers.Normal.NormalAddProposalMean(builder, p.proposal_mean)
            infcomp.flatbuffers.Normal.NormalAddProposalStd(builder, p.proposal_std)
            distribution = infcomp.flatbuffers.Normal.NormalEnd(builder)

            # construct message body (ProposalReply)
            infcomp.flatbuffers.ProposalReply.ProposalReplyStart(builder)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddSuccess(builder, True)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddDistributionType(builder, infcomp.flatbuffers.Distribution.Distribution().Normal)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddDistribution(builder, distribution)
            message_body = infcomp.flatbuffers.ProposalReply.ProposalReplyEnd(builder)
        elif isinstance(p, Flip):
            infcomp.flatbuffers.Flip.FlipStart(builder)
            infcomp.flatbuffers.Flip.FlipAddProposalProbability(builder, p.proposal_probability)
            distribution = infcomp.flatbuffers.Flip.FlipEnd(builder)

            infcomp.flatbuffers.ProposalReply.ProposalReplyStart(builder)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddSuccess(builder, True)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddDistributionType(builder, infcomp.flatbuffers.Distribution.Distribution().Flip)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddDistribution(builder, distribution)
            message_body = infcomp.flatbuffers.ProposalReply.ProposalReplyEnd(builder)
        elif isinstance(p, Discrete):
            # construct probabilities
            proposal_probabilities = Tensor_to_NDArray(builder, p.proposal_probabilities)

            # construct Discrete
            infcomp.flatbuffers.Discrete.DiscreteStart(builder)
            infcomp.flatbuffers.Discrete.DiscreteAddProposalProbabilities(builder, proposal_probabilities)
            distribution = infcomp.flatbuffers.Discrete.DiscreteEnd(builder)

            # construct message body (ProposalReply)
            infcomp.flatbuffers.ProposalReply.ProposalReplyStart(builder)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddSuccess(builder, True)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddDistributionType(builder, infcomp.flatbuffers.Distribution.Distribution().Discrete)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddDistribution(builder, distribution)
            message_body = infcomp.flatbuffers.ProposalReply.ProposalReplyEnd(builder)
        elif isinstance(p, Categorical):
            # construct probabilities
            proposal_probabilities = Tensor_to_NDArray(builder, p.proposal_probabilities)

            # construct Categorical
            infcomp.flatbuffers.Categorical.CategoricalStart(builder)
            infcomp.flatbuffers.Categorical.CategoricalAddProposalProbabilities(builder, proposal_probabilities)
            distribution = infcomp.flatbuffers.Categorical.CategoricalEnd(builder)

            # construct message body (ProposalReply)
            infcomp.flatbuffers.ProposalReply.ProposalReplyStart(builder)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddSuccess(builder, True)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddDistributionType(builder, infcomp.flatbuffers.Distribution.Distribution().Categorical)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddDistribution(builder, distribution)
            message_body = infcomp.flatbuffers.ProposalReply.ProposalReplyEnd(builder)
        else:
            util.log_error('Unsupported proposal distribution: {0}'.format(p))

        # construct message
        infcomp.flatbuffers.Message.MessageStart(builder)
        infcomp.flatbuffers.Message.MessageAddBodyType(builder, infcomp.flatbuffers.MessageBody.MessageBody().ProposalReply)
        infcomp.flatbuffers.Message.MessageAddBody(builder, message_body)
        message = infcomp.flatbuffers.Message.MessageEnd(builder)

        builder.Finish(message)
        message = builder.Output()
        self.replier.send_reply(message)
