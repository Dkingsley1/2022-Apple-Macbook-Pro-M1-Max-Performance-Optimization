import numpy as np
import pyaudio
from pedalboard import Pedalboard, Limiter
import sys

# --- MASTERING CONFIGURATION ---
THRESHOLD = -1.0  # Professional ceiling for maximum loudness
RELEASE = 200.0  # Smooth release to protect spatial imaging
RATE = 96000  # Your Audient sample rate
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 2


def get_audient_index(pa_instance):
    """Finds the Audient index using the passed-in PyAudio instance."""
    for i in range(pa_instance.get_device_count()):
        dev = pa_instance.get_device_info_by_index(i)
        if "audient" in dev['name'].lower():
            return i
    return None


def run_mastering_limiter():
    pa = pyaudio.PyAudio()
    stream = None

    try:
        device_index = get_audient_index(pa)

        if device_index is None:
            print("❌ Audient interface not found. Check connection.")
            return

        # Pedalboard Limiter setup
        board = Pedalboard([Limiter(threshold_db=THRESHOLD, release_ms=RELEASE)])

        print(f"✅ Mastering Limiter ACTIVE")
        print(f"Ceiling: {THRESHOLD}dB | Sample Rate: {RATE}Hz")

        stream = pa.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         output=True,
                         input_device_index=device_index,
                         output_device_index=device_index,
                         frames_per_buffer=CHUNK)

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32).reshape(CHUNK, CHANNELS)

            # Limiting both channels together preserves the stereo image
            limited_audio = board(audio_data.T, RATE).T

            stream.write(limited_audio.astype(np.float32).tobytes())

    except KeyboardInterrupt:
        print("\nStopping Limiter...")
    except Exception as e:
        print(f"⚠️ Error: {e}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()


if __name__ == "__main__":
    run_mastering_limiter()