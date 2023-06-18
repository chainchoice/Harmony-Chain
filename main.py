import gradio as gr
import pypianoroll
import matplotlib.pyplot as plt
from basic_pitch.inference import predict
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd

def convert_audio_to_midi_and_visualize(file_path):
    # Generate MIDI data from audio
    model_output, midi_data, note_events = predict(file_path)

    # Save MIDI data temporarily as a MIDI file
    temp_midi_path = 'temp_output.mid'
    midi_data.write(temp_midi_path)
    
    # Read MIDI data using pypianoroll
    multitrack = pypianoroll.read(temp_midi_path, resolution=12)
    multitrack_numpy = multitrack.stack()
    print(multitrack_numpy)
    print(multitrack_numpy[0])
    filename = "output.csv"
    np.savetxt(filename, multitrack_numpy[0], delimiter=",")
    
    # Visualize numpy array
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(multitrack_numpy[0].T, aspect='auto', cmap='viridis')
    ax.set_xlabel("Time step")
    ax.set_ylabel("Pitch")
    ax.figure.colorbar(ax.imshow(multitrack_numpy[0].T, aspect='auto', cmap='viridis'), ax=ax, label="Velocity")

    # Save image in bytes
    image_byte = BytesIO()
    plt.savefig(image_byte, format='png')
    image_byte.seek(0)

    # Convert byte image to PIL
    pil_image = Image.open(image_byte)
    
    # Estimate chords based on the MIDI data
    chord_names = estimate_chords(multitrack_numpy[0])
    
    return pil_image, chord_names, temp_midi_path


def estimate_chords(midi_array):
    # Extended chord estimation logic (major, minor, augmented, major 7th, dominant 7th, sus4, etc.)
    # This is a basic example and can be improved.
    
    chord_names = []
    
    # Define basic chords
    major = [4, 7]
    minor = [3, 7]
    augmented = [4, 8]
    
    # Define seventh chords and others
    major_seventh = [4, 7, 11]
    dominant_seventh = [4, 7, 10]
    minor_seventh = [3, 7, 10]
    sus4 = [5, 7]
    
    # Tensions (extensions)
    ninth = [14]
    eleventh = [17]
    thirteenth = [21]
    
    for time_step in midi_array:
        notes = np.where(time_step > 0)[0]
        if len(notes) < 3:
            chord_names.append("N/A")
            continue

        intervals = np.sort(notes[1:] - notes[0])
        
        # Check basic triads
        if np.array_equal(intervals[:2], major):
            chord_names.append("Major")
        elif np.array_equal(intervals[:2], minor):
            chord_names.append("Minor")
        elif np.array_equal(intervals[:2], augmented):
            chord_names.append("Augmented")
        # Check seventh chords
        elif np.array_equal(intervals[:3], major_seventh):
            chord_names.append("Major Seventh")
        elif np.array_equal(intervals[:3], dominant_seventh):
            chord_names.append("Dominant Seventh")
        elif np.array_equal(intervals[:3], minor_seventh):
            chord_names.append("Minor Seventh")
        # Check sus4
        elif np.array_equal(intervals[:2], sus4):
            chord_names.append("Sus4")
        # Check tensions
        else:
            base_chord = "Unknown"
            if np.array_equal(intervals[:3], dominant_seventh):
                base_chord = "Dominant"
            elif np.array_equal(intervals[:3], minor_seventh):
                base_chord = "Minor"
                
            if len(intervals) > 3 and intervals[3] in ninth:
                chord_names.append(f"{base_chord} Ninth")
            elif len(intervals) > 4 and intervals[4] in eleventh:
                chord_names.append(f"{base_chord} Eleventh")
            elif len(intervals) > 5 and intervals[5] in thirteenth:
                chord_names.append(f"{base_chord} Thirteenth")
            else:
                chord_names.append("Other")
    
    return chord_names



# Define interface to accept uploaded files
inputs = gr.inputs.Audio(type="filepath", label="Upload a WAV file")

# The outputs are the visualized numpy array as an image, an array of chord names, and the converted MIDI file
outputs = [
    gr.outputs.Image(type="pil", label="Visualized MIDI Data"),
    gr.outputs.Textbox(label="Chord Names"),
    gr.outputs.File(label="Converted MIDI File")
]

# Launch the interface
gr.Interface(fn=convert_audio_to_midi_and_visualize, inputs=inputs, outputs=outputs).launch()

