import torch
import zmq
import flatbuffers
from termcolor import colored

from . import util, state, __version__
from .distributions import Uniform, Normal, Categorical, Poisson
from .PPLProtocol import Message as PPLProtocol_Message
from .PPLProtocol import MessageBody as PPLProtocol_MessageBody
from .PPLProtocol import ProtocolTensor as PPLProtocol_ProtocolTensor
from .PPLProtocol import Distribution as PPLProtocol_Distribution
from .PPLProtocol import Uniform as PPLProtocol_Uniform
from .PPLProtocol import Normal as PPLProtocol_Normal
from .PPLProtocol import Categorical as PPLProtocol_Categorical
from .PPLProtocol import Poisson as PPLProtocol_Poisson
from .PPLProtocol import Handshake as PPLProtocol_Handshake
from .PPLProtocol import HandshakeResult as PPLProtocol_HandshakeResult
from .PPLProtocol import Run as PPLProtocol_Run
from .PPLProtocol import RunResult as PPLProtocol_RunResult
from .PPLProtocol import Sample as PPLProtocol_Sample
from .PPLProtocol import SampleResult as PPLProtocol_SampleResult
from .PPLProtocol import Observe as PPLProtocol_Observe
from .PPLProtocol import ObserveResult as PPLProtocol_ObserveResult
from .PPLProtocol import Reset as PPLProtocol_Reset


class Requester(object):
    def __init__(self, server_address):
        self._server_address = server_address
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 100)
        print('PPLProtocol (Python): zmq.REQ socket connecting to server {}'.format(self._server_address))
        self._socket.connect(self._server_address)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if not self._socket.closed:
            self._socket.close()
            self._context.destroy()
            print('PPLProtocol (Python): zmq.REQ socket disconnected from server {}'.format(self._server_address))

    def send_request(self, request):
        self._socket.send(request)

    def receive_reply(self):
        return self._socket.recv()


class ModelServer(object):
    def __init__(self, server_address):
        self._requester = Requester(server_address)
        self.system_name, self.model_name = self._handshake()
        print('PPLProtocol (Python): This system        : {}'.format(colored('pyprob {}'.format(__version__), 'green')))
        print('PPLProtocol (Python): Connected to system: {}'.format(colored(self.system_name, 'green')))
        print('PPLProtocol (Python): Model name         : {}'.format(colored(self.model_name, 'green', attrs=['bold'])))

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
        if len(data) == 0:
            return None
        else:
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
        elif body_type == PPLProtocol_MessageBody.MessageBody().Reset:
            message_body = PPLProtocol_Reset.Reset()
        else:
            raise RuntimeError('PPLProtocol (Python): Received unexpected message body type: {}'.format(body_type))
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
            raise RuntimeError('PPLProtocol (Python): Unexpected reply to handshake.')

    def forward(self, observation=None):
        builder = flatbuffers.Builder(64)

        if observation is not None:
            # construct ProtocolTensor
            observation = self._variable_to_protocol_tensor(builder, observation)

        # construct MessageBody
        PPLProtocol_Run.RunStart(builder)
        if observation is not None:
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
                replace = bool(message_body.Replace())
                distribution_type = message_body.DistributionType()
                if distribution_type == PPLProtocol_Distribution.Distribution().Uniform:
                    uniform = PPLProtocol_Uniform.Uniform()
                    uniform.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    low = self._protocol_tensor_to_variable(uniform.Low())
                    high = self._protocol_tensor_to_variable(uniform.High())
                    dist = Uniform(low, high)
                elif distribution_type == PPLProtocol_Distribution.Distribution().Normal:
                    normal = PPLProtocol_Normal.Normal()
                    normal.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    mean = self._protocol_tensor_to_variable(normal.Mean())
                    stddev = self._protocol_tensor_to_variable(normal.Stddev())
                    dist = Normal(mean, stddev)
                elif distribution_type == PPLProtocol_Distribution.Distribution().Categorical:
                    categorical = PPLProtocol_Categorical.Categorical()
                    categorical.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    probs = self._protocol_tensor_to_variable(categorical.Probs())
                    dist = Categorical(probs)
                elif distribution_type == PPLProtocol_Distribution.Distribution().Poisson:
                    poisson = PPLProtocol_Poisson.Poisson()
                    poisson.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    rate = self._protocol_tensor_to_variable(poisson.Rate())
                    dist = Poisson(rate)
                else:
                    raise RuntimeError('PPLProtocol (Python): Sample from an unexpected distribution requested.')
                result = state.sample(dist, control, replace, address)
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
                address = message_body.Address().decode('utf-8')
                value = self._protocol_tensor_to_variable(message_body.Value())
                distribution_type = message_body.DistributionType()
                if distribution_type == PPLProtocol_Distribution.Distribution().Uniform:
                    uniform = PPLProtocol_Uniform.Uniform()
                    uniform.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    low = self._protocol_tensor_to_variable(uniform.Low())
                    high = self._protocol_tensor_to_variable(uniform.High())
                    dist = Uniform(low, high)
                elif distribution_type == PPLProtocol_Distribution.Distribution().Normal:
                    normal = PPLProtocol_Normal.Normal()
                    normal.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    mean = self._protocol_tensor_to_variable(normal.Mean())
                    stddev = self._protocol_tensor_to_variable(normal.Stddev())
                    dist = Normal(mean, stddev)
                elif distribution_type == PPLProtocol_Distribution.Distribution().Categorical:
                    categorical = PPLProtocol_Categorical.Categorical()
                    categorical.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    probs = self._protocol_tensor_to_variable(categorical.Probs())
                    dist = Categorical(probs)
                elif distribution_type == PPLProtocol_Distribution.Distribution().Poisson:
                    poisson = PPLProtocol_Poisson.Poisson()
                    poisson.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    rate = self._protocol_tensor_to_variable(poisson.Rate())
                    dist = Poisson(rate)
                else:
                    raise RuntimeError('PPLProtocol (Python): Sample from an unexpected distribution requested.')
                if value is None:
                    print('PPLProtocol (Python): Warning: observed None value.')
                else:
                    state.observe(dist, value, address)
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
            elif isinstance(message_body, PPLProtocol_Reset.Reset):
                raise RuntimeError('PPLProtocol (Python): Received a reset request. Protocol out of sync.')
            else:
                raise RuntimeError('PPLProtocol (Python): Received unexpected message.')
