# TuneLink

## About
TuneLink is an avant-garde project that synergizes advanced technologies to bridge music and visual arts. It dissects musical compositions, visualizes chord structures, and employs artificial intelligence to generate concept art that echoes the soul of the music. "Tune" embodies music and harmony, while "Link" encapsulates the connections this project aims to build among various art forms and cultures. TuneLink's mission is to analyze, interpret, and interlink musical genres and cultural elements, creating a fusion that speaks to the diversity and harmony of global culture.

### Technical Aspects
TuneLink's core engine ingests audio files in WAV format and converts them to MIDI data. This data is transformed into musical statistics that get processed by a Large Language Model (LLM) to conceive textual descriptions of images resonating with the music. Employing image generation AI, TuneLink brings these images to life.

The generated image can be fused with the original music and minted as a Non-Fungible Token (NFT), turning it into a unique digital asset on the blockchain. Through this, TuneLink stretches the boundaries of art by interlacing melodies with visual forms, forging deeper cultural connections.

## Features

- Converts audio to MIDI data.
- Identifies and visualizes chords through a bar graph.
- Accepts a user-specified concept as a creative input.
- Generates text-based descriptions of images related to the music and concept.
- Uses OpenAI's API to synthesize images from the text descriptions.
- Offers the option to mint the music and generated image as an NFT.（unfinished）

## Installation

Install the required Python packages:

```sh
pip install gradio pypianoroll matplotlib requests Pillow firebase_admin
```

Place your Firebase Admin SDK certificate file in the project directory and update the `credentials.Certificate` and `storageBucket` accordingly in the script.

## Usage

Execute the script:

```sh
python main.py
```

This command launches a web interface where you:

1. Upload an audio file (WAV format).
2. Enter a musical concept in the text field.
3. Provide your OpenAI API key.
4. Click submit and wait for the processing.

You will see a chord visualization, the MIDI file, chord data, a text-based description of the concept art, and a public URL to the generated concept art image.

## Under the Hood

1. **Audio to MIDI Conversion**: The script converts audio to MIDI using a pitch prediction model.

2. **Chord Identification & Visualization**: Chords are identified from MIDI data and visualized through a bar graph.

3. **Concept Generation**: The Large Language Model uses chords and a user-defined concept to generate text descriptions for concept art.

4. **Image Generation**: Using OpenAI's API, the application synthesizes images based on the text description.

5. **Storing to Firebase**: The generated image is uploaded to Firebase storage and made accessible via a public URL.

6. **NFT Minting**: (Optional) The music and image can be minted as a Non-Fungible Token, validating its uniqueness and value.（unfinished）

TuneLink, through its innovative blend of technology, invites you on a journey where music and art converge, building bridges across cultures and stimulating new avenues in the realm of artistic expression.
