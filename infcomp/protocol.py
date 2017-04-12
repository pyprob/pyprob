import infcomp
import infcomp.zmq
from infcomp import util
from infcomp.probprog import Sample, Trace, UniformDiscreteProposal
import infcomp.flatbuffers.Message
import infcomp.flatbuffers.MessageBody
import infcomp.flatbuffers.TracesFromPriorRequest
import infcomp.flatbuffers.TracesFromPriorReply
import infcomp.flatbuffers.Trace
import infcomp.flatbuffers.NDArray
import infcomp.flatbuffers.ProposalDistribution
import infcomp.flatbuffers.UniformDiscreteProposal

import flatbuffers
import sys

class BatchRequester(object):
    def __init__(self, server_address):
        self.requester = infcomp.zmq.Requester(server_address)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.requester.close()

    def get_batch(self, data, standardize):
        message = infcomp.flatbuffers.Message.Message.GetRootAsMessage(data, 0)
        body_type = message.BodyType()
        if body_type == infcomp.flatbuffers.MessageBody.MessageBody().TracesFromPriorReply:
            reply = infcomp.flatbuffers.TracesFromPriorReply.TracesFromPriorReply()
            reply.Init(message.Body().Bytes, message.Body().Pos)
        else:
            util.log_error('Unknown reply with body:MessageBody type: {0}. Expecting a TracesFromPriorReply.'.format(bodyType))

        traces_length = reply.TracesLength()
        traces = []
        for i in range(traces_length):
            trace = Trace()

            t = reply.Traces(i)
            obs = util.NDArray_to_Tensor(t.Observes())
            if standardize:
                obs = util.standardize(obs)
            trace.set_observes(obs)

            samples_length = t.SamplesLength()
            for timeStep in range(samples_length):
                s = t.Samples(timeStep)
                address = s.Address().decode("utf-8")
                instance = s.Instance()
                value = util.NDArray_to_Tensor(s.Value())
                proposal_type = s.ProposalType()

                if proposal_type == infcomp.flatbuffers.ProposalDistribution.ProposalDistribution().UniformDiscreteProposal:
                    p = infcomp.flatbuffers.UniformDiscreteProposal.UniformDiscreteProposal()
                    p.Init(s.Proposal().Bytes, s.Proposal().Pos)
                    proposal = UniformDiscreteProposal(p.Min(), p.Max()) # Note: p.Probabilities() is not used in TracesFromPriorReply
                else:
                    util.log_error('Unknown reply with proposal:ProposalDistribution id: {0}.'.format(proposal_type))

                sample = Sample(address, instance, value, proposal)
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
        #self.requester.send_request({'command':'new-batch', 'command-param':n})
        # allocate buffer for the request
        builder = flatbuffers.Builder(64) # actual message is 36 bytes

        # construct the request
        infcomp.flatbuffers.TracesFromPriorRequest.TracesFromPriorRequestStart(builder)
        infcomp.flatbuffers.TracesFromPriorRequest.TracesFromPriorRequestAddNumTraces(builder, n)
        request = infcomp.flatbuffers.TracesFromPriorRequest.TracesFromPriorRequestEnd(builder)

        # construct message
        infcomp.flatbuffers.Message.MessageStart(builder)
        infcomp.flatbuffers.Message.MessageAddBodyType(builder, infcomp.flatbuffers.MessageBody.MessageBody().TracesFromPriorRequest)
        infcomp.flatbuffers.Message.MessageAddBody(builder, request)
        message = infcomp.flatbuffers.Message.MessageEnd(builder)
        builder.Finish(message)

        message = builder.Output()
        self.requester.send_request(message)

    def receive_batch(self, standardize=True):
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
