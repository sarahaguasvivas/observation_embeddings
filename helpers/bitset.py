import numpy as np
import typing
from nptyping import Float16

# NOTE(sarahaguasvivas): Bitset represents the
#           keys to my hash maps using the encodings.
#           It is a container class (similar to an union in
#           embedded)
class BitSet:
    def __init__(self, number : Float16):

        # this is the human-readable value
        self.float_val = number

        # this is the serialized bitset as a string
        self.bitset_ser = bin(np.float16(number).view('H'))[2:]\
                            .zfill(16)

        # this is the bitset as a byte array
        self.bitarray = ' '.join(map(bin,bytearray(self.bitset_ser,'utf8')))

if __name__ == '__main__':
    # declare my float as a bitset
    o = BitSet(21.5)
    print(o.bitset_ser)
    print(o.bitarray)