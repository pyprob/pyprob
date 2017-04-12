import infcomp
import infcomp.zmq
from infcomp import util
from infcomp.probprog import Sample, Trace, UniformDiscreteProposal
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
import infcomp.flatbuffers.ProposalDistribution
import infcomp.flatbuffers.UniformDiscreteProposal

import flatbuffers
import sys

def NDArray_to_Tensor(ndarray):
    def make_array(indexer_func, length):
        ret = []
        for i in range(length):
            ret.append(indexer_func(i))
        return ret
    data = make_array(ndarray.Data, ndarray.DataLength())
    shape = make_array(ndarray.Shape, ndarray.ShapeLength())
    return util.Tensor(data).view(shape)

def Tensor_to_NDArray(builder, tensor):
    tensor_numpy = tensor.cpu().numpy()
    data = tensor_numpy.tolist()
    shape = list(tensor_numpy.shape)
    infcomp.flatbuffers.NDArray.NDArrayStart(builder)
    infcomp.flatbuffers.NDArray.NDArrayStartDataVector(builder, len(data))
    for d in reversed(data):
        infcomp.flatbuffers.NDArray.NDArrayAddData(builder, d)
    infcomp.flatbuffers.NDArray.NDArrayStartShapeVector(builder, len(shape))
    for s in reversed(shape):
        infcomp.flatbuffers.NDArray.NDArrayAddShape(builder, s)
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
    proposal_type = s.ProposalType()
    if not proposal_type is None:
        if proposal_type == infcomp.flatbuffers.ProposalDistribution.ProposalDistribution().UniformDiscreteProposal:
            p = infcomp.flatbuffers.UniformDiscreteProposal.UniformDiscreteProposal()
            p.Init(s.Proposal().Bytes, s.Proposal().Pos)
            sample.proposal = UniformDiscreteProposal(p.Min(), p.Max()) # Note: p.Probabilities() is not used in TracesFromPriorReply
        else:
            util.log_error('Unknown proposal:ProposalDistribution id: {0}.'.format(proposal_type))
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
            print(current_sample)
            print(previous_sample)
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
        infcomp.flatbuffers.ObservesInitReply.ObservesInitReplyAddOk(builder, True)
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

        if isinstance(p, UniformDiscreteProposal):
            # construct probabilities
            probabilities = Tensor_to_NDArray(builder, p.probabilities)

            # construct proposal UniformDiscreteProposal
            infcomp.flatbuffers.UniformDiscreteProposal.UniformDiscreteProposalStart(builder)
            infcomp.flatbuffers.UniformDiscreteProposal.UniformDiscreteProposalAddProbabilities(builder, probabilities)
            proposal = infcomp.flatbuffers.UniformDiscreteProposal.UniformDiscreteProposalEnd(builder)

            # construct message body (ProposalReply)
            infcomp.flatbuffers.ProposalReply.ProposalReplyStart(builder)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddProposalType(builder, infcomp.flatbuffers.ProposalDistribution.ProposalDistribution().UniformDiscreteProposal)
            infcomp.flatbuffers.ProposalReply.ProposalReplyAddProposal(builder, proposal)
            message_body = infcomp.flatbuffers.ProposalReply.ProposalReplyEnd(builder)

            # construct message
            infcomp.flatbuffers.Message.MessageStart(builder)
            infcomp.flatbuffers.Message.MessageAddBodyType(builder, infcomp.flatbuffers.MessageBody.MessageBody().ProposalReply)
            infcomp.flatbuffers.Message.MessageAddBody(builder, message_body)
        else:
            util.log_error('Unsupported proposal distribution: {0}'.format(p))

        builder.Finish(message)
        message = builder.Output()
        self.replier.send_reply(message)
