# automatically generated by the FlatBuffers compiler, do not modify

# namespace: ppx

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Reset(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Reset()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReset(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ResetBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x50\x50\x58\x46", size_prefixed=size_prefixed)

    # Reset
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def Start(builder): builder.StartObject(0)
def ResetStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def End(builder): return builder.EndObject()
def ResetEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)