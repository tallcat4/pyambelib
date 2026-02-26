import ctypes
import os
import glob
import sys

_lib_dir = os.path.dirname(__file__)
_lib_pattern = os.path.join(_lib_dir, "_libambe*")
_found_libs = glob.glob(_lib_pattern)

if not _found_libs:
    raise ImportError(
        "Compiled C extension '_libambe' not found. "
        "Please install the package using 'pip install .'"
    )

_lib_path = _found_libs[0]

try:
    libambe = ctypes.CDLL(_lib_path)
except OSError as e:
    raise ImportError(f"Failed to load shared library '{_lib_path}': {e}")


class DecoderContext(ctypes.Structure):
    pass

libambe.create_context.restype = ctypes.POINTER(DecoderContext)
libambe.free_context.argtypes = [ctypes.POINTER(DecoderContext)]

libambe.process_ambe2450.argtypes = [
    ctypes.POINTER(DecoderContext),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_short)
]
libambe.process_ambe2450.restype = ctypes.c_int

libambe.process_ambe3600.argtypes = [
    ctypes.POINTER(DecoderContext),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_short)
]
libambe.process_ambe3600.restype = ctypes.c_int

libambe.fec_demod_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_char)
]

class AmbeDecoder:
    """AMBE Codec Wrapper"""
    
    def __init__(self):
        self._ctx = libambe.create_context()
        if not self._ctx:
            raise MemoryError("Failed to allocate decoder context")
    
    def __del__(self):
        if self._ctx:
            libambe.free_context(self._ctx)
            self._ctx = None

    def decode_2450(self, payload_bytes: bytes) -> list[int]:
        """
        Decode 7 bytes (AMBE2450/ThumbDV) to PCM samples.
        Returns a list of 160 signed 16-bit integers.
        """
        if len(payload_bytes) != 7:
            raise ValueError(f"Payload must be 7 bytes, got {len(payload_bytes)}")
        
        c_payload = (ctypes.c_uint8 * 7)(*payload_bytes)
        pcm_buffer = (ctypes.c_short * 160)()
        
        libambe.process_ambe2450(self._ctx, c_payload, pcm_buffer)
        return list(pcm_buffer)

    def decode_3600(self, payload_bytes: bytes) -> list[int]:
        """
        Decode 9 bytes (AMBE3600x2450) to PCM samples.
        Returns a list of 160 signed 16-bit integers.
        """
        if len(payload_bytes) != 9:
            raise ValueError(f"Payload must be 9 bytes, got {len(payload_bytes)}")
        
        c_payload = (ctypes.c_uint8 * 9)(*payload_bytes)
        pcm_buffer = (ctypes.c_short * 160)()
        
        libambe.process_ambe3600(self._ctx, c_payload, pcm_buffer)
        return list(pcm_buffer)

def fec_demod(input_bytes_9: bytes) -> list[int]:
    """
    Apply FEC demodulation to convert 9-byte 3600 data to 49-bit 2450 data.
    Returns: List of 49 integers (0 or 1).
    """
    if len(input_bytes_9) != 9:
         raise ValueError(f"Input must be 9 bytes, got {len(input_bytes_9)}")
         
    c_payload = (ctypes.c_uint8 * 9)(*input_bytes_9)
    out_ambe_d = (ctypes.c_char * 49)()
    
    libambe.fec_demod_wrapper(c_payload, out_ambe_d)
    
    return [
        out_ambe_d[i][0] if isinstance(out_ambe_d[i], bytes) else out_ambe_d[i] 
        for i in range(49)
    ]