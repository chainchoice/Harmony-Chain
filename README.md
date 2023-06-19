# TuneLink

## About
Under the name "TuneLink", this project utilizes advanced digital technologies to meticulously integrate multiple media contents, with the aim of promoting cultural harmony across the globe. The term “Tune” implies both music and harmony, while “Link” signifies connection and association, which embodies the essence of “TuneLink” as a project with a critical mission - as suggested by its name - to digitally analyze different music genres and cultural elements, and by linking and reconstructing them, to create musically diverse compositions.

### Efforts in the Hackathon
In this hackathon, we developed the core engine essential for realizing the "TuneLink" project. We take music files in WAV format as initial inputs, and convert them into performance data (MIDI), which is then transformed into statistical data with musical implications. This statistical data is fed into a Large Language Model (LLM), which extracts conceptual images associated with the music in text format. Based on this text, an image-generation AI is employed to visually represent the conceptual images of the music.

### Generation as NFT
In the final stage, the generated image is combined with the music, and this is produced as a Non-Fungible Token (NFT) and stored on the blockchain. Consequently, each piece of work is treated as a valuable digital asset with its uniqueness. In conclusion, “TuneLink” is a groundbreaking initiative that digitally analyzes and links music and culture to create new value. This project can be described as one that paves new possibilities for the future by connecting the melodies of music through digital chains and deepening global cultural connections.


# Music Concept Art Generator

This project is a Music Concept Art Generator, which analyzes an input audio file and generates concept art based on the chords present in the music and a user-defined concept. The application is built with Python and uses Gradio for the user interface, pypianoroll for MIDI processing, matplotlib for visualization, and OpenAI's GPT for natural language processing.

## Features

- Converts audio to MIDI.
- Identifies chords from MIDI data.
- Visualizes chords present in the music using a bar graph.
- Takes a user-defined concept and generates a description of an image that would suit the music and concept.
- Uses OpenAI's API to generate an image based on the description.

## Installation

Before running the script, you need to install the required Python packages:

```sh
pip install gradio pypianoroll matplotlib requests Pillow
```

## Usage

Run the script:

```sh
python main.py
```

This will launch a web-based user interface. Follow these steps to use the application:

1. Upload an audio file in WAV format.
2. Enter a concept for the music in the text field.
3. Enter your OpenAI API key.
4. Click submit and wait for the processing to complete.

The interface will display a visualization of the chords present in the music, provide a download link for the MIDI file, show the chord tally data, and display the description generated for the concept art. Additionally, it provides a download link for the generated concept art image.

## How it works

The application consists of several components:

1. **Audio to MIDI conversion**: Using a pitch prediction model, the application converts the input audio into MIDI data.

2. **Chord Identification**: It iterates through the MIDI data to identify the chords present in each time step. Chords are identified based on sets of notes and are tallied.

3. **Chord Visualization**: Using matplotlib, the application creates a bar graph showing the occurrences of different chords.

4. **Concept Description Generation**: With the help of the OpenAI API, the application generates a description of concept art that would suit the input music and concept. This description is based on the chords present and the user-defined concept.

5. **Image Generation**: The application sends a request to OpenAI's image generation API with the description and generates an image based on it.

## Note

This application uses the OpenAI API which is a paid service. Make sure you have an account and the necessary API key to access it.

Also, the code imports from a module called `basic_pitch.inference`. Please ensure you have this module or replace it with an alternative method for audio to MIDI conversion.

## Dependencies

- Python 3.x
- gradio
- pypianoroll
- matplotlib
- requests
- Pillow
- OpenAI API Key (required for image generation)








