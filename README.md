# TuneLink

## About
「TuneLink」という名の下、本プロジェクトは高度なデジタル技術を活用し、複数のメディアコンテンツを精密に連携させ、世界における文化的調和を促進することを目的としています。Tuneは音楽と調和の両方を意味し、Linkは連携と結びつきを示しており、これにより「TuneLink」はその名称が示す通り、異なる音楽ジャンルや文化の要素をデジタル的に解析し、これらを結びつけ、再構築することで多様性に富む音楽作品を生み出すという、重要な使命を帯びたプロジェクトです。

### ハッカソンでの取り組み
本ハッカソンにおいては、「TuneLink」プロジェクトの実現に必要な核となるエンジンの開発を行いました。WAV形式の音楽ファイルを初期入力として受け取り、これを演奏データ（MIDI）に変換した上で音楽的意味合いを持つ統計データへと変換します。この統計データをLarge Language Model (LLM) に供給し、楽曲に関連する概念的なイメージをテキスト形式で抽出します。このテキストを基に、画像生成AIを用いて楽曲の概念的イメージを視覚的に表現した画像を生成します。

### NFTとしての生成

最終段階では、生成された画像と音楽を結合し、これをNFT（非代替性トークン）として生成、ブロックチェーン上に格納します。これにより、それぞれの作品は独自性を持つ価値あるデジタル資産として扱われることになります。
結論として、「TuneLink」は音楽と文化をデジタル的に解析し、結びつけることで、新たな価値を創造する画期的な取り組みです。このプロジェクトは、音楽の旋律をデジタル的な連鎖で繋ぎ合わせ、世界的な文化的つながりを深めることにより、未来に向けて新しい可能性を切り開くものであると言えます。


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
python script_name.py
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








