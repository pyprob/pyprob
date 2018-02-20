import torch
import zmq
import flatbuffers
from termcolor import colored

from . import util, state, __version__
from .distributions import Uniform, Normal
from .PPLProtocol import Message as PPLProtocol_Message
from .PPLProtocol import MessageBody as PPLProtocol_MessageBody
from .PPLProtocol import ProtocolTensor as PPLProtocol_ProtocolTensor
from .PPLProtocol import Distribution as PPLProtocol_Distribution
from .PPLProtocol import Uniform as PPLProtocol_Uniform
from .PPLProtocol import Normal as PPLProtocol_Normal
from .PPLProtocol import Handshake as PPLProtocol_Handshake
from .PPLProtocol import HandshakeResult as PPLProtocol_HandshakeResult
from .PPLProtocol import Run as PPLProtocol_Run
from .PPLProtocol import RunResult as PPLProtocol_RunResult
from .PPLProtocol import Sample as PPLProtocol_Sample
from .PPLProtocol import SampleResult as PPLProtocol_SampleResult
from .PPLProtocol import Observe as PPLProtocol_Observe
from .PPLProtocol import ObserveResult as PPLProtocol_ObserveResult


class Requester(object):
    def __init__(self, server_address):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(server_address)
        print('Protocol (Python): zmq.REQ socket connected to server {}'.format(server_address))

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
            print('Protocol (Python): zmq.REQ socket disconnected')

    def send_request(self, request):
        self._socket.send(request)

    def receive_reply(self):
        return self._socket.recv()


class ModelServer(object):
    def __init__(self, server_address):
        self._requester = Requester(server_address)
        self.system_name, self.model_name = self._handshake()
        print('Protocol (Python): this system        : {}'.format(colored('pyprob {}'.format(__version__), 'green')))
        print('Protocol (Python): connected to system: {}'.format(colored(self.system_name, 'green')))
        print('Protocol (Python): model name         : {}'.format(colored(self.model_name, 'green', attrs=['bold'])))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self._requester.close()

    def _protocol_tensor_to_variable(self, protocol_tensor):
        data = protocol_tensor.DataAsNumpy()
        shape = protocol_tensor.ShapeAsNumpy()
        t = torch.from_numpy(data)
        if len(shape) != 0:
            t = t.view(shape.tolist())
        return util.to_variable(t)

    def _variable_to_protocol_tensor(self, builder, variable):
        if variable is None:
            variable = util.to_variable(torch.zeros(0))
        variable_numpy = util.to_numpy(variable)
        data = variable_numpy.flatten().tolist()
        shape = list(variable_numpy.shape)

        # pack data
        PPLProtocol_ProtocolTensor.ProtocolTensorStartDataVector(builder, len(data))
        for d in reversed(data):
            builder.PrependFloat64(d)
        data = builder.EndVector(len(data))

        # pack shape
        PPLProtocol_ProtocolTensor.ProtocolTensorStartShapeVector(builder, len(shape))
        for s in reversed(shape):
            builder.PrependInt32(s)
        shape = builder.EndVector(len(shape))

        PPLProtocol_ProtocolTensor.ProtocolTensorStart(builder)
        PPLProtocol_ProtocolTensor.ProtocolTensorAddData(builder, data)
        PPLProtocol_ProtocolTensor.ProtocolTensorAddShape(builder, shape)
        return PPLProtocol_ProtocolTensor.ProtocolTensorEnd(builder)

    def _get_message_body(self, message_buffer):
        message = PPLProtocol_Message.Message.GetRootAsMessage(message_buffer, 0)
        body_type = message.BodyType()
        if body_type == PPLProtocol_MessageBody.MessageBody().HandshakeResult:
            message_body = PPLProtocol_HandshakeResult.HandshakeResult()
        elif body_type == PPLProtocol_MessageBody.MessageBody().RunResult:
            message_body = PPLProtocol_RunResult.RunResult()
        elif body_type == PPLProtocol_MessageBody.MessageBody().Sample:
            message_body = PPLProtocol_Sample.Sample()
        elif body_type == PPLProtocol_MessageBody.MessageBody().Observe:
            message_body = PPLProtocol_Observe.Observe()
        else:
            raise RuntimeError('Received unexpected message body type: {}'.format(body_type))
        message_body.Init(message.Body().Bytes, message.Body().Pos)
        return message_body

    def _handshake(self):
        builder = flatbuffers.Builder(64)
        # consturct MessageBody
        system_name = builder.CreateString('pyprob {}'.format(__version__))
        PPLProtocol_Handshake.HandshakeStart(builder)
        PPLProtocol_Handshake.HandshakeAddSystemName(builder, system_name)
        message_body = PPLProtocol_Handshake.HandshakeEnd(builder)

        # construct Message
        PPLProtocol_Message.MessageStart(builder)
        PPLProtocol_Message.MessageAddBodyType(builder, PPLProtocol_MessageBody.MessageBody().Handshake)
        PPLProtocol_Message.MessageAddBody(builder, message_body)
        message = PPLProtocol_Message.MessageEnd(builder)
        builder.Finish(message)

        message = builder.Output()
        self._requester.send_request(message)

        reply = self._requester.receive_reply()
        message_body = self._get_message_body(reply)
        if isinstance(message_body, PPLProtocol_HandshakeResult.HandshakeResult):
            system_name = message_body.SystemName().decode('utf-8')
            model_name = message_body.ModelName().decode('utf-8')
            return system_name, model_name
        else:
            raise RuntimeError('Unexpected resply to handshake.')

    def forward(self, observation):
        builder = flatbuffers.Builder(64)

        # construct ProtocolTensor
        observation = self._variable_to_protocol_tensor(builder, observation)

        # construct MessageBody
        PPLProtocol_Run.RunStart(builder)
        PPLProtocol_Run.RunAddObservation(builder, observation)
        message_body = PPLProtocol_Run.RunEnd(builder)

        # construct Message
        PPLProtocol_Message.MessageStart(builder)
        PPLProtocol_Message.MessageAddBodyType(builder, PPLProtocol_MessageBody.MessageBody().Run)
        PPLProtocol_Message.MessageAddBody(builder, message_body)
        message = PPLProtocol_Message.MessageEnd(builder)
        builder.Finish(message)

        message = builder.Output()
        self._requester.send_request(message)

        while True:
            reply = self._requester.receive_reply()
            message_body = self._get_message_body(reply)

            if isinstance(message_body, PPLProtocol_RunResult.RunResult):
                result = self._protocol_tensor_to_variable(message_body.Result())
                return result
            elif isinstance(message_body, PPLProtocol_Sample.Sample):
                address = message_body.Address().decode('utf-8')
                control = bool(message_body.Control())
                record_last_only = bool(message_body.RecordLastOnly())
                distribution_type = message_body.DistributionType()
                if distribution_type == PPLProtocol_Distribution.Distribution().Uniform:
                    uniform = PPLProtocol_Uniform.Uniform()
                    uniform.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    low = uniform.Low()
                    high = uniform.High()
                    dist = Uniform(low, high)
                elif distribution_type == PPLProtocol_Distribution.Distribution().Normal:
                    normal = PPLProtocol_Normal.Normal()
                    normal.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    mean = normal.Mean()
                    stddev = normal.Stddev()
                    dist = Normal(mean, stddev)
                else:
                    raise RuntimeError('Sample from an unexpected distribution requested.')
                result = state.sample(dist, control, record_last_only, address)
                builder = flatbuffers.Builder(64)
                result = self._variable_to_protocol_tensor(builder, result)
                PPLProtocol_SampleResult.SampleResultStart(builder)
                PPLProtocol_SampleResult.SampleResultAddResult(builder, result)
                message_body = PPLProtocol_SampleResult.SampleResultEnd(builder)

                # construct Message
                PPLProtocol_Message.MessageStart(builder)
                PPLProtocol_Message.MessageAddBodyType(builder, PPLProtocol_MessageBody.MessageBody().SampleResult)
                PPLProtocol_Message.MessageAddBody(builder, message_body)
                message = PPLProtocol_Message.MessageEnd(builder)
                builder.Finish(message)

                message = builder.Output()
                self._requester.send_request(message)
            elif isinstance(message_body, PPLProtocol_Observe.Observe):
                value = self._protocol_tensor_to_variable(message_body.Value())
                distribution_type = message_body.DistributionType()
                if distribution_type == PPLProtocol_Distribution.Distribution().Uniform:
                    uniform = PPLProtocol_Uniform.Uniform()
                    uniform.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    low = uniform.Low()
                    high = uniform.High()
                    dist = Uniform(low, high)
                elif distribution_type == PPLProtocol_Distribution.Distribution().Normal:
                    normal = PPLProtocol_Normal.Normal()
                    normal.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    mean = normal.Mean()
                    stddev = normal.Stddev()
                    dist = Normal(mean, stddev)
                else:
                    raise RuntimeError('Sample from an unexpected distribution requested.')
                state.observe(dist, value)
                builder = flatbuffers.Builder(64)
                PPLProtocol_ObserveResult.ObserveResultStart(builder)
                message_body = PPLProtocol_ObserveResult.ObserveResultEnd(builder)

                # construct Message
                PPLProtocol_Message.MessageStart(builder)
                PPLProtocol_Message.MessageAddBodyType(builder, PPLProtocol_MessageBody.MessageBody().ObserveResult)
                PPLProtocol_Message.MessageAddBody(builder, message_body)
                message = PPLProtocol_Message.MessageEnd(builder)
                builder.Finish(message)

                message = builder.Output()
                self._requester.send_request(message)
            else:
                raise RuntimeError('Received unexpected message.')
