from importlib.metadata import version
try:
    __version__ = version(__name__)
except:
    pass

from silero_vad.model import load_silero_vad
from silero_vad.utils_vad import (get_speech_timestamps,
                                  get_speech_timestamps_from_probs,
                                  save_audio,
                                  read_audio,
                                  VADIterator,
                                  collect_chunks,
                                  drop_chunks)
from silero_vad.sequence_vad import (SileroVADSequence,
                                     get_speech_timestamps_sequence)
