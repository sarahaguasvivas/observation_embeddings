import numpy as np
import struct
from bitarray.util import serialize, deserialize, ba2int, vl_decode
from bitarray import bitarray
import typing
from nptyping import Float16
from codecs import decode

# NOTE(sarahaguasvivas): Bitset represents the
#           keys to my hash maps using the encodings.
#           It is a container class (similar to an union in
#           embedded)
class BitSet:
    def __init__(self, float_val : np.float16 = 0.0):
        self.floating_point = None
        self.binary : bitarray = None
        self.update_float(float_val)

    def update_float(self, x : np.float16 = 0.0):
        self.floating_point = np.float16(x)
        self.binary = bitarray('0' + bin(np.float16(x).view('H'))[2:])

    def update_binary(self, x : bitarray):
        self.binary = x
        self.floating_point = np.float16(int(x.to01()[::-1], 2))

if __name__ == '__main__':
    # declare my float as a bitset
    bs1 = BitSet(2.0)
    bs2 = BitSet(4.0)
    print(bs1.binary)
    print(bs2.binary)
    bs1.update_binary(bs1.binary)
    print(bs1.floating_point)