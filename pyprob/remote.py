import torch
import zmq
import flatbuffers
from termcolor import colored

from . import util, state, __version__
from .distributions import Uniform, Normal, Categorical, Poisson
from .ppx import Message as ppx_Message
from .ppx import MessageBody as ppx_MessageBody
from .ppx import Tensor as ppx_Tensor
from .ppx import Distribution as ppx_Distribution
from .ppx import Uniform as ppx_Uniform
from .ppx import Normal as ppx_Normal
from .ppx import Categorical as ppx_Categorical
from .ppx import Poisson as ppx_Poisson
from .ppx import Handshake as ppx_Handshake
from .ppx import HandshakeResult as ppx_HandshakeResult
from .ppx import Run as ppx_Run
from .ppx import RunResult as ppx_RunResult
from .ppx import Sample as ppx_Sample
from .ppx import SampleResult as ppx_SampleResult
from .ppx import Observe as ppx_Observe
from .ppx import ObserveResult as ppx_ObserveResult
from .ppx import Tag as ppx_Tag
from .ppx import TagResult as ppx_TagResult
from .ppx import Reset as ppx_Reset


class ZMQRequester():
    def __init__(self, server_address):
        self._server_address = server_address
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 100)
        print('ppx (Python): zmq.REQ socket connecting to server {}'.format(self._server_address))
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
            self._context.term()
            print('ppx (Python): zmq.REQ socket disconnected from server {}'.format(self._server_address))

    def send_request(self, request):
        self._socket.send(request)

    def receive_reply(self):
        return self._socket.recv()


class ModelServer(object):
    def __init__(self, server_address):
        self._requester = ZMQRequester(server_address)
        self.system_name, self.model_name = self._handshake()
        print('ppx (Python): This system        : {}'.format(colored('pyprob {}'.format(__version__), 'green')))
        print('ppx (Python): Connected to system: {}'.format(colored(self.system_name, 'green')))
        print('ppx (Python): Model name         : {}'.format(colored(self.model_name, 'green', attrs=['bold'])))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self._requester.close()

    def _protocol_tensor_to_variable(self, protocol_tensor):
        if protocol_tensor is None:
            return None
        data = protocol_tensor.DataAsNumpy()
        shape = protocol_tensor.ShapeAsNumpy()
        if len(data) == 0:
            return None
        else:
            t = torch.from_numpy(data)
        if len(shape) != 0:
            t = t.view(shape.tolist())
        return util.to_tensor(t)

    def _variable_to_protocol_tensor(self, builder, variable):
        if variable is None:
            variable = util.to_tensor(torch.zeros(0))
        variable_numpy = util.to_numpy(variable)
        data = variable_numpy.flatten().tolist()
        shape = list(variable_numpy.shape)

        # pack data
        ppx_Tensor.TensorStartDataVector(builder, len(data))
        for d in reversed(data):
            builder.PrependFloat64(d)
        data = builder.EndVector(len(data))

        # pack shape
        ppx_Tensor.TensorStartShapeVector(builder, len(shape))
        for s in reversed(shape):
            builder.PrependInt32(s)
        shape = builder.EndVector(len(shape))

        ppx_Tensor.TensorStart(builder)
        ppx_Tensor.TensorAddData(builder, data)
        ppx_Tensor.TensorAddShape(builder, shape)
        return ppx_Tensor.TensorEnd(builder)

    def _get_message_body(self, message_buffer):
        message = ppx_Message.Message.GetRootAsMessage(message_buffer, 0)
        body_type = message.BodyType()
        if body_type == ppx_MessageBody.MessageBody().HandshakeResult:
            message_body = ppx_HandshakeResult.HandshakeResult()
        elif body_type == ppx_MessageBody.MessageBody().RunResult:
            message_body = ppx_RunResult.RunResult()
        elif body_type == ppx_MessageBody.MessageBody().Sample:
            message_body = ppx_Sample.Sample()
        elif body_type == ppx_MessageBody.MessageBody().Observe:
            message_body = ppx_Observe.Observe()
        elif body_type == ppx_MessageBody.MessageBody().Tag:
            message_body = ppx_Tag.Tag()
        elif body_type == ppx_MessageBody.MessageBody().Reset:
            message_body = ppx_Reset.Reset()
        else:
            raise RuntimeError('ppx (Python): Received unexpected message body type: {}'.format(body_type))
        message_body.Init(message.Body().Bytes, message.Body().Pos)
        return message_body

    def _handshake(self):
        builder = flatbuffers.Builder(64)
        # consturct MessageBody
        system_name = builder.CreateString('pyprob {}'.format(__version__))
        ppx_Handshake.HandshakeStart(builder)
        ppx_Handshake.HandshakeAddSystemName(builder, system_name)
        message_body = ppx_Handshake.HandshakeEnd(builder)

        # construct Message
        ppx_Message.MessageStart(builder)
        ppx_Message.MessageAddBodyType(builder, ppx_MessageBody.MessageBody().Handshake)
        ppx_Message.MessageAddBody(builder, message_body)
        message = ppx_Message.MessageEnd(builder)
        builder.Finish(message)

        message = builder.Output()
        self._requester.send_request(message)

        reply = self._requester.receive_reply()
        message_body = self._get_message_body(reply)
        if isinstance(message_body, ppx_HandshakeResult.HandshakeResult):
            system_name = message_body.SystemName().decode('utf-8')
            model_name = message_body.ModelName().decode('utf-8')
            return system_name, model_name
        else:
            raise RuntimeError('ppx (Python): Unexpected reply to handshake.')

    def forward(self):
        builder = flatbuffers.Builder(64)

        # construct MessageBody
        ppx_Run.RunStart(builder)
        message_body = ppx_Run.RunEnd(builder)

        # construct Message
        ppx_Message.MessageStart(builder)
        ppx_Message.MessageAddBodyType(builder, ppx_MessageBody.MessageBody().Run)
        ppx_Message.MessageAddBody(builder, message_body)
        message = ppx_Message.MessageEnd(builder)
        builder.Finish(message)

        message = builder.Output()
        self._requester.send_request(message)

        while True:
            reply = self._requester.receive_reply()
            message_body = self._get_message_body(reply)

            if isinstance(message_body, ppx_RunResult.RunResult):
                result = self._protocol_tensor_to_variable(message_body.Result())
                return result
            elif isinstance(message_body, ppx_Sample.Sample):
                address = message_body.Address().decode('utf-8')
                name = message_body.Name().decode('utf-8')
                if name == '':
                    name = None
                control = bool(message_body.Control())
                replace = bool(message_body.Replace())
                distribution_type = message_body.DistributionType()
                if distribution_type == ppx_Distribution.Distribution().Uniform:
                    uniform = ppx_Uniform.Uniform()
                    uniform.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    low = self._protocol_tensor_to_variable(uniform.Low())
                    high = self._protocol_tensor_to_variable(uniform.High())
                    dist = Uniform(low, high)
                elif distribution_type == ppx_Distribution.Distribution().Normal:
                    normal = ppx_Normal.Normal()
                    normal.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    mean = self._protocol_tensor_to_variable(normal.Mean())
                    stddev = self._protocol_tensor_to_variable(normal.Stddev())
                    dist = Normal(mean, stddev)
                elif distribution_type == ppx_Distribution.Distribution().Categorical:
                    categorical = ppx_Categorical.Categorical()
                    categorical.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    probs = self._protocol_tensor_to_variable(categorical.Probs())
                    dist = Categorical(probs)
                elif distribution_type == ppx_Distribution.Distribution().Poisson:
                    poisson = ppx_Poisson.Poisson()
                    poisson.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    rate = self._protocol_tensor_to_variable(poisson.Rate())
                    dist = Poisson(rate)
                else:
                    raise RuntimeError('ppx (Python): Sample from an unexpected distribution requested.')
                result = state.sample(distribution=dist, control=control, replace=replace, name=name, address=address)
                builder = flatbuffers.Builder(64)
                result = self._variable_to_protocol_tensor(builder, result)
                ppx_SampleResult.SampleResultStart(builder)
                ppx_SampleResult.SampleResultAddResult(builder, result)
                message_body = ppx_SampleResult.SampleResultEnd(builder)

                # construct Message
                ppx_Message.MessageStart(builder)
                ppx_Message.MessageAddBodyType(builder, ppx_MessageBody.MessageBody().SampleResult)
                ppx_Message.MessageAddBody(builder, message_body)
                message = ppx_Message.MessageEnd(builder)
                builder.Finish(message)

                message = builder.Output()
                self._requester.send_request(message)
            elif isinstance(message_body, ppx_Observe.Observe):
                address = message_body.Address().decode('utf-8')
                name = message_body.Name().decode('utf-8')
                if name == '':
                    name = None
                value = self._protocol_tensor_to_variable(message_body.Value())
                distribution_type = message_body.DistributionType()
                if distribution_type == ppx_Distribution.Distribution().NONE:
                    dist = None
                elif distribution_type == ppx_Distribution.Distribution().Uniform:
                    uniform = ppx_Uniform.Uniform()
                    uniform.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    low = self._protocol_tensor_to_variable(uniform.Low())
                    high = self._protocol_tensor_to_variable(uniform.High())
                    dist = Uniform(low, high)
                elif distribution_type == ppx_Distribution.Distribution().Normal:
                    normal = ppx_Normal.Normal()
                    normal.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    mean = self._protocol_tensor_to_variable(normal.Mean())
                    stddev = self._protocol_tensor_to_variable(normal.Stddev())
                    dist = Normal(mean, stddev)
                elif distribution_type == ppx_Distribution.Distribution().Categorical:
                    categorical = ppx_Categorical.Categorical()
                    categorical.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    probs = self._protocol_tensor_to_variable(categorical.Probs())
                    dist = Categorical(probs)
                elif distribution_type == ppx_Distribution.Distribution().Poisson:
                    poisson = ppx_Poisson.Poisson()
                    poisson.Init(message_body.Distribution().Bytes, message_body.Distribution().Pos)
                    rate = self._protocol_tensor_to_variable(poisson.Rate())
                    dist = Poisson(rate)
                else:
                    raise RuntimeError('ppx (Python): Sample from an unexpected distribution requested: {}'.format(distribution_type))

                state.observe(distribution=dist, value=value, name=name, address=address)
                builder = flatbuffers.Builder(64)
                ppx_ObserveResult.ObserveResultStart(builder)
                message_body = ppx_ObserveResult.ObserveResultEnd(builder)

                # construct Message
                ppx_Message.MessageStart(builder)
                ppx_Message.MessageAddBodyType(builder, ppx_MessageBody.MessageBody().ObserveResult)
                ppx_Message.MessageAddBody(builder, message_body)
                message = ppx_Message.MessageEnd(builder)
                builder.Finish(message)

                message = builder.Output()
                self._requester.send_request(message)
            elif isinstance(message_body, ppx_Tag.Tag):
                address = message_body.Address().decode('utf-8')
                name = message_body.Name().decode('utf-8')
                if name == '':
                    name = None
                value = self._protocol_tensor_to_variable(message_body.Value())
                state.tag(value=value, name=name, address=address)
                builder = flatbuffers.Builder(64)
                ppx_TagResult.TagResultStart(builder)
                message_body = ppx_TagResult.TagResultEnd(builder)

                # construct Message
                ppx_Message.MessageStart(builder)
                ppx_Message.MessageAddBodyType(builder, ppx_MessageBody.MessageBody().TagResult)
                ppx_Message.MessageAddBody(builder, message_body)
                message = ppx_Message.MessageEnd(builder)
                builder.Finish(message)

                message = builder.Output()
                self._requester.send_request(message)
            elif isinstance(message_body, ppx_Reset.Reset):
                raise RuntimeError('ppx (Python): Received a reset request. Protocol out of sync.')
            else:
                raise RuntimeError('ppx (Python): Received unexpected message.')
