import numpy as np
import struct
from bitarray.util import  count_xor
from bitarray import bitarray
import typing
from nptyping import Float16
from codecs import decode

def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]


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
        bin_str = float_to_bin(float(x))
        self.binary = bitarray(bin_str, endian = 'little')

    def update_binary(self, x : bitarray):
        self.binary = x
        self.floating_point = bin_to_float(x.to01())

    def compute_hamming_weight(self, b):
        # b is another object of type BitSet
        return count_xor(self.binary, b.binary)

if __name__ == '__main__':
    # declare my float as a bitset
    bs1 = BitSet(2345.34456)
    bs2 = BitSet(4.0)
    print(bs1.binary)
    print(bs2.binary)
    bs1.update_binary(bs1.binary)
    print(bs1.floating_point)
    print(bs1.binary ^ bs2.binary)
    print(count_xor(bs1.binary, bs2.binary))