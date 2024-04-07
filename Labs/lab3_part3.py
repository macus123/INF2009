from pydub import AudioSegment
from pydub.effects import reverb, delay
import numpy as np
from scikits.audiolab import Sndfile

# Load the original audio file
audio = AudioSegment.from_file("input_audio.wav")

# Apply reverb effect
audio_with_reverb = reverb(audio, reverberance=50)

# Apply echo effect
audio_with_echo = delay(audio, delay_time=500, gain=0.2)

# Export the modified audio with reverb
audio_with_reverb.export("output_audio_with_reverb.wav", format="wav")

# Export the modified audio with echo
audio_with_echo.export("output_audio_with_echo.wav", format="wav")

# Load the original audio file for pitch alteration
f = Sndfile('input_audio.wav', 'r')
data = f.read_frames(f.nframes)

# Change pitch by a factor (higher values increase pitch)
pitch_factor = 1.5
data_with_altered_pitch = data.copy()
data_with_altered_pitch[:, 0] = np.interp(np.arange(0, len(data_with_altered_pitch)), 
                                          np.arange(0, len(data_with_altered_pitch), 1/pitch_factor),
                                          data_with_altered_pitch[:, 0])
data_with_altered_pitch[:, 1] = np.interp(np.arange(0, len(data_with_altered_pitch)), 
                                          np.arange(0, len(data_with_altered_pitch), 1/pitch_factor),
                                          data_with_altered_pitch[:, 1])

# Save the modified audio with altered pitch
f = Sndfile('output_audio_with_altered_pitch.wav', 'w', f.format, f.channels, f.samplerate)
f.write_frames(data_with_altered_pitch)
f.close()

print("All effects applied and saved successfully!")
