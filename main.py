import gradio as gr
import pypianoroll
import matplotlib.pyplot as plt
from basic_pitch.inference import predict
from io import BytesIO
from PIL import Image
import numpy as np
import collections
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import json
import os
import tempfile
import urllib.request
import firebase_admin
from firebase_admin import credentials, storage
import uuid
import shutil


# Firebase Admin SDK initialization
cred = credentials.Certificate('XXX')
firebase_admin.initialize_app(cred, {'storageBucket': 'XXX'})

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Dictionary of chords and their relative note structures
chord_types = {
    'Major Triad': [0, 4, 7],
    'Minor Triad': [0, 3, 7],
    'Diminished Triad': [0, 3, 6],
    'Augmented Triad': [0, 4, 8],
    'Dominant Seventh': [0, 4, 7, 10],
    'Diminished Seventh': [0, 3, 6, 9],
    'Major Seventh': [0, 4, 7, 11],
    'Minor Seventh': [0, 3, 7, 10],
    'Minor Seventh Flat Five': [0, 3, 6, 10],
    'Augmented Seventh': [0, 4, 8, 10],
    'Augmented Major Seventh': [0, 4, 8, 11],
    'Suspended Fourth': [0, 5, 7],
    'Sixth Chord': [0, 4, 7, 9],
    'Flat Ninth': [0, 4, 7, 10, 13],
    'Ninth': [0, 4, 7, 10, 14],
    'Sharp Ninth': [0, 4, 7, 10, 15],
    'Eleventh': [0, 4, 7, 10, 14, 17],
    'Sharp Eleventh': [0, 4, 7, 10, 14, 18],
    'Flat Thirteenth': [0, 4, 7, 10, 14, 17, 20],
    'Sharp Thirteenth': [0, 4, 7, 10, 14, 18, 21]
}

# Initialize LLM wrapper
llm = OpenAI(temperature=0.7)

# Create prompt template
prompt = PromptTemplate(
    input_variables=["chord","concept"],
    template="「楽曲中に出現したコードの種類と頻度」:{chord}「楽曲のコンセプト」:{concept} ［１］：「楽曲中に出現したコードの種類と頻度」を基に「楽曲のおしゃれさ」「楽曲の明るさ」「楽曲の暗さ」「切なさ」「複雑さ」を100段階でスコアリングしてください。［2］：［１］で抽出した得点および「楽曲のコンセプト」をもとにこの楽曲のコンセプトアートとして適切な画像のイメージを描写してください。［3］：［2］で描写した適切な画像のイメージを300文字の英語で抽出してください。［4］[1]～[3]を順に実施し返答してください。ただし出力は[3]で生成した300文字の英語のみとしてください。",
)

# Create LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

def upload_to_firebase_storage(local_file_path):
    # Create a unique ID for the file
    unique_id = str(uuid.uuid4())
    
    # Create destination file path in the storage
    dest_file_path = 'generated_images/{}.jpg'.format(unique_id)
    
    # Upload the file
    bucket = storage.bucket()
    blob = bucket.blob(dest_file_path)
    blob.upload_from_filename(local_file_path)
    
    # Make the file publicly accessible
    blob.make_public()
    
    # Return the public URL of the file
    return blob.public_url

def identify_chord(notes_present):
    chords = []
    for root_note in range(12):
        for chord_name, structure in chord_types.items():
            if all((note + root_note) % 12 in notes_present for note in structure):
                chords.append(chord_name)
    return chords

class CustomLLMChain:
    def __init__(self, api_key, temperature=0.7):
        self.llm = OpenAI(temperature=temperature, api_key=api_key)
        self.prompt = PromptTemplate(
            input_variables=["chord","concept"],
            template="「楽曲中に出現したコードの種類と頻度」:{chord}「楽曲のコンセプト」:{concept} ［１］：「楽曲中に出現したコードの種類と頻度」を基に「楽曲のおしゃれさ」「楽曲の明るさ」「楽曲の暗さ」「切なさ」「複雑さ」を100段階でスコアリングしてください。［2］：［１］で抽出した得点および「楽曲のコンセプト」をもとにこの楽曲のコンセプトアートとして適切な画像のイメージを描写してください。［3］：［2］で描写した適切な画像のイメージを300文字の英語で抽出してください。［4］[1]～[3]を順に実施し返答してください。ただし出力は[3]で生成した300文字の英語のみとしてください。",
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, inputs):
        return self.chain.run(inputs)

def convert_audio_to_midi_and_visualize(file_path, concept, openai_api_key):
    chain = CustomLLMChain(openai_api_key)
    # Generate MIDI data from audio
    model_output, midi_data, note_events = predict(file_path)

    # Save MIDI data temporarily as a MIDI file
    temp_midi_path = 'temp_output.mid'
    midi_data.write(temp_midi_path)
    
    # Read MIDI data using pypianoroll
    multitrack = pypianoroll.read(temp_midi_path, resolution=12)
    multitrack_numpy = multitrack.stack()
    
    # Extracting the number of time steps
    num_time_steps = multitrack_numpy.shape[1]
    
    # Tally chords
    chord_tally = collections.defaultdict(int)
    
    # Iterate through time steps and identify chords
    for time_step in range(num_time_steps):
        notes_present = set()
        for pitch in range(128):
            if multitrack_numpy[0, time_step, pitch] > 0:
                note_idx = pitch % 12
                notes_present.add(note_idx)
                
        # Identify chords in this timestep
        chords_in_timestep = identify_chord(notes_present)
        
        # Tally the identified chords
        for chord in chords_in_timestep:
            chord_tally[chord] += 1
    
    # Visualize the chord tally as a bar graph
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(chord_tally.keys(), chord_tally.values())
    ax.set_xlabel("Chord Type")
    ax.set_ylabel("Occurrences")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save image in bytes
    image_byte = BytesIO()
    plt.savefig(image_byte, format='png')
    image_byte.seek(0)

    # Convert byte image to PIL
    pil_image = Image.open(image_byte)
    
    # Run LLM chain with the chord tally and concept as inputs
    chord_tally_str = str(chord_tally)
    prediction = chain.run({"chord": chord_tally_str, "concept": concept})

    # Make request to OpenAI image generation API
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai_api_key
    }
    data = {
        "prompt": prediction.strip(),
        "n": 1,
        "size": "512x512"
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_data = response.json()

    # Debugging: print the response data structure
    print("Response Data:", response_data)

    # Process the response data to obtain the image URL
    try:
        if 'data' in response_data and len(response_data['data']) > 0:
            image_url = response_data['data'][0]['url']
            # Download the image and save it to a temporary file
            with urllib.request.urlopen(image_url) as response, tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(response.read())
                tmp_file_path = tmp_file.name  # this will be the path to the temporary file
        else:
            tmp_file_path = None  # Or a placeholder file path in case of failure
    except KeyError:
        print("Error: The key 'url' is not found in the response data.")
        tmp_file_path = None  # Or a placeholder file path in case of failure

    # Upload the image to Firebase Storage
    if tmp_file_path:
        public_url = upload_to_firebase_storage(tmp_file_path)
        
        # Optionally, remove the local temporary file
        # (uncomment the following line if you want to remove the local file)
        # os.remove(tmp_file_path)
    else:
        public_url = None

    return pil_image, temp_midi_path, chord_tally, prediction.strip(), public_url

# Define interface to accept uploaded files, text input for the concept, and the OpenAI API key
inputs = [
    gr.inputs.Audio(type="filepath", label="Upload a WAV file"),
    gr.inputs.Textbox(label="Enter the Concept of the Music"),
    gr.inputs.Textbox(label="Enter OpenAI API Key")

]

# Modifying the outputs to return the public URL instead of the file
outputs = [
    gr.outputs.Image(type="pil", label="Chord Tally Visualization"),
    gr.outputs.File(label="Converted MIDI File"),
    gr.outputs.JSON(label="Chord Tally Data"),
    gr.outputs.Textbox(label="Prediction"),
    gr.outputs.Textbox(label="Generated Image Public URL")  # Return the public URL instead of the file
]

# Launch the interface
gr.Interface(fn=convert_audio_to_midi_and_visualize, inputs=inputs, outputs=outputs).launch()
