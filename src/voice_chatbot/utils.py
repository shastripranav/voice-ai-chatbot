import io
import struct

# 16-bit PCM at 16kHz is the standard for speech processing
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_BITS_PER_SAMPLE = 16


def create_silent_wav(
    duration_seconds: float = 1.0, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> bytes:
    """Generate a valid WAV file containing silence."""
    num_samples = int(sample_rate * duration_seconds)
    bits_per_sample = DEFAULT_BITS_PER_SAMPLE
    channels = DEFAULT_CHANNELS

    data_size = num_samples * channels * (bits_per_sample // 8)
    buf = io.BytesIO()

    # WAV header is always 44 bytes for PCM format
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")

    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * channels * (bits_per_sample // 8)))
    buf.write(struct.pack("<H", channels * (bits_per_sample // 8)))
    buf.write(struct.pack("<H", bits_per_sample))

    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(b"\x00" * data_size)

    return buf.getvalue()


def get_audio_duration_wav(audio_bytes: bytes) -> float | None:
    """Estimate duration from WAV bytes. Returns None if not a valid WAV."""
    # FIXME: handle audio formats other than wav/mp3 more gracefully
    if len(audio_bytes) < 44 or audio_bytes[:4] != b"RIFF":
        return None

    try:
        channels = struct.unpack_from("<H", audio_bytes, 22)[0]
        sample_rate = struct.unpack_from("<I", audio_bytes, 24)[0]
        bits_per_sample = struct.unpack_from("<H", audio_bytes, 34)[0]
        data_size = struct.unpack_from("<I", audio_bytes, 40)[0]

        if sample_rate == 0 or channels == 0 or bits_per_sample == 0:
            return None

        bytes_per_sample = channels * (bits_per_sample // 8)
        num_samples = data_size // bytes_per_sample
        return num_samples / sample_rate
    except struct.error:
        return None


def validate_audio_bytes(audio_bytes: bytes) -> bool:
    if not audio_bytes or len(audio_bytes) < 12:
        return False
    # check for WAV, OGG, or MP3 magic bytes
    if audio_bytes[:4] == b"RIFF":
        return True
    if audio_bytes[:4] == b"OggS":
        return True
    if audio_bytes[:2] == b"\xff\xfb" or audio_bytes[:3] == b"ID3":
        return True
    return False
