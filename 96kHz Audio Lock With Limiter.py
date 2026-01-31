import numpy as np
import pyaudio
from pedalboard import Pedalboard, Limiter
import time

# --- CONFIGURATION ---
TARGET_RATE = 96000
THRESHOLD = -1.0
RELEASE = 200.0
CHUNK = 2048
CHANNELS = 2


def run_limiter():
    pa = pyaudio.PyAudio()
    board = Pedalboard([Limiter(threshold_db=THRESHOLD, release_ms=RELEASE)])

    while True:  # Infinite loop to keep service alive
        try:
            # Auto-find Audient
            device_index = None
            for i in range(pa.get_device_count()):
                if "audient" in pa.get_device_info_by_index(i)['name'].lower():
                    device_index = i
                    break

            if device_index is None:
                time.sleep(5)  # Wait 5 seconds and try again if Audient is off
                continue

            stream = pa.open(format=pyaudio.paFloat32, channels=CHANNELS,
                             rate=TARGET_RATE, input=True, output=True,
                             input_device_index=device_index,
                             output_device_index=device_index,
                             frames_per_buffer=CHUNK)

            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32).reshape(CHUNK, CHANNELS)
                limited_audio = board(audio_data.T, TARGET_RATE).T
                stream.write(limited_audio.astype(np.float32).tobytes())

        except Exception:
            time.sleep(2)  # Brief pause on error before restarting
            continue


if __name__ == "__main__":
    run_limiter()