"""
Help Guide Tab

Displays documentation and tips for using Voice Clone Studio.

Standalone testing:
    python -m modules.core_components.tools.help_page
"""

# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
from textwrap import dedent
from modules.core_components.tool_base import Tool, ToolConfig


# Help content functions
def show_voice_clone_help():
    """Return help content for Voice Clone tab."""
    return dedent("""
        ### 🎙️ Voice Clone

        **Clone any voice from a reference sample**

        &nbsp;

        #### How it works:
        1. **Select a voice sample** from your samples folder
        2. **Enter text** you want to generate in that voice
        3. **Choose engine**: Qwen3 (Small/Large), VibeVoice (Small/Large), or LuxTTS
        4. **Set language** (Auto-detect recommended)
        5. **Click Generate** - first time processing creates a cached prompt

        &nbsp;

        #### Model Options:
        - **Qwen3-Small (0.6B)**: Fast, good quality, lower VRAM
        - **Qwen3-Large (1.7B)**: Best quality, higher VRAM
        - **VibeVoice-Small (1.5B)**: Alternative engine, good for long-form
        - **VibeVoice-Large**: Highest quality, most VRAM
        - **LuxTTS (Default)**: 48 kHz output, fast voice cloning

        &nbsp;

        #### Performance Tips:
        - ⚡ **First generation** for a sample takes longer (creates prompt cache)
        - ⚡ **Subsequent generations** use cached prompt and are much faster
        - 🔄 Cache is **per model size** - switching sizes recreates cache
        - 💾 Qwen caches are saved as `.pt` files; LuxTTS caches are `.prompt` files

        &nbsp;

        #### Seed Control:
        - Use **same seed** for reproducible results
        - Set to **-1** for random generation each time
        """)


def show_conversation_help():
    """Return help content for Conversation tab."""
    return dedent("""
        ### 💬 Conversation Generator

        **Create multi-speaker dialogues with different voices**

        &nbsp;

        #### Two Engines Available:

        **1. Qwen3-TTS (CustomVoice)** - Fast, premium speakers
        - Uses 9 built-in premium voices
        - Supports style instructions in parentheses
        - Best for short to medium conversations

        **2. VibeVoice** - Long-form, custom voices
        - Use your own voice samples (1-4 speakers)
        - Can generate up to **90 minutes** of audio
        - Best for audiobooks, podcasts, long dialogues

        &nbsp;

        #### Script Format:
        ```
        [1]: Hello there!
        [2]: Nice to meet you.
        [1]: How are you today?
        ```
        Or use speaker names:
        ```
        Vivian: Hello there!
        Ryan: Nice to meet you.
        ```

        &nbsp;

        #### Style Instructions (Qwen3 only):
        Add emotions in parentheses:
        ```
        [1]: (excited) This is amazing!
        [2]: (confused) What do you mean?
        ```

        &nbsp;

        #### VibeVoice Setup:
        1. Select voice samples for Speaker 1-4
        2. Use `[1]:`, `[2]:` etc. in your script
        3. Adjust CFG scale (3.0 recommended)
        4. Generate - can take several minutes for long content

        &nbsp;

        #### Pause Duration:
        Controls silence between dialogue lines (Qwen3 only)
        - Default: 0.5s
        - Longer pauses: More natural conversations
        - Shorter pauses: Faster pacing
        """)


def show_voice_presets_help():
    """Return help content for Voice Presets tab."""
    return dedent("""
        ### 🎭 Voice Presets (CustomVoice)

        **Use premium pre-trained voices with style control**

        &nbsp;

        #### Available Speakers:
        - **Vivian**: Bright, edgy young female (Chinese)
        - **Serena**: Warm, gentle young female (Chinese)
        - **Uncle_Fu**: Seasoned male, low mellow timbre (Chinese)
        - **Dylan**: Youthful Beijing male, clear (Chinese/Beijing)
        - **Eric**: Lively Chengdu male, slightly husky (Chinese/Sichuan)
        - **Ryan**: Dynamic male, strong rhythm (English)
        - **Aiden**: Sunny American male, clear midrange (English)
        - **Ono_Anna**: Playful Japanese female, light (Japanese)
        - **Sohee**: Warm Korean female, rich emotion (Korean)

        &nbsp;

        #### Style Instructions (Optional):
        Add emotional or stylistic guidance:
        - "excited", "nervous", "calm", "happy", "sad"
        - "whispering", "shouting", "formal", "casual"
        - "fast pace", "slow pace", "dramatic"

        &nbsp;

        #### Model Size:
        - **Small (0.6B)**: Faster, lower VRAM
        - **Large (1.7B)**: Better quality, higher VRAM

        &nbsp;

        #### Best Practices:
        - Match speaker language with your text language
        - Keep style instructions simple and clear
        - Experiment with different speakers for variety
        """)


def show_voice_design_help():
    """Return help content for Voice Design tab."""
    return dedent("""
        ### 🎨 Voice Design

        **Create entirely new voices from natural language descriptions**

        &nbsp;

        #### How it works:
        1. **Enter reference text** - what the voice should say
        2. **Describe the voice** in natural language
        3. **Generate** - creates a unique voice matching your description
        4. **Save to samples** if you like it (to use for cloning)

        &nbsp;

        #### Voice Description Examples:
        - "A young male with a deep, calm voice"
        - "Elderly woman, warm and gentle tone"
        - "Energetic teenager, slightly high-pitched"
        - "Professional narrator, clear and authoritative"
        - "Friendly female, casual and approachable"

        &nbsp;

        #### Tips:
        - Be **specific** about age, gender, tone, emotion
        - Mention **voice characteristics**: deep, high, raspy, smooth
        - Add **personality traits**: friendly, serious, energetic, calm
        - Keep descriptions **concise** but descriptive

        &nbsp;

        #### Design → Clone Workflow:
        1. Create a designed voice you like
        2. Save it as a sample (gives it a name)
        3. Use it in **Voice Clone** tab for longer generations
        4. Share your custom voices with others

        &nbsp;

        #### Model:
        - Uses **Qwen3-TTS VoiceDesign** (1.7B only)
        - More creative but less predictable than cloning
        - May take several attempts to get desired result
        """)


def show_prep_audio_help():
    """Return help content for Prep Samples tab."""
    return dedent("""
        ### 🎬 Prep Samples

        **Prepare audio samples for voice cloning**

        &nbsp;

        #### Workflow:
        1. **Load audio or video** - drag & drop or browse
        2. **Auto-extract audio** from video if needed
        3. **Normalize** audio levels for consistency
        4. **Convert to mono** if stereo
        5. **Transcribe** using Whisper or VibeVoice ASR
        6. **Edit transcription** if needed
        7. **Save as sample** with a name

        &nbsp;

        #### Supported Formats:
        - **Audio**: WAV, MP3, FLAC, OGG, M4A, AAC
        - **Video**: MP4, AVI, MOV, MKV, WebM, FLV

        &nbsp;

        #### Transcription Engines:
        - **Whisper**: High accuracy, supports many languages
        - **VibeVoice ASR**: Multi-speaker transcription, good for conversations

        &nbsp;

        #### Audio Requirements:
        - **Clean audio**: Minimal background noise
        - **Clear speech**: Well-articulated, not mumbled
        - **Length**: 5-30 seconds ideal for reference samples
        - **Single speaker** (for Voice Clone tab)
        - **Multi-speaker** OK for dataset preparation

        &nbsp;

        #### Batch Transcription:
        - Process entire folders at once
        - Automatically detects existing transcriptions
        - Option to replace or skip existing .txt files
        - Great for preparing training datasets
        """)


def show_dataset_help():
    """Return help content for Prep Dataset tab."""
    return dedent("""
        ### 📚 Prep Dataset

        **Prepare datasets for voice model training**

        &nbsp;

        #### Getting Started:
        1. **Create subfolders** in `datasets/` to organize different training sets
        2. **Place audio files** in your chosen subfolder
        3. **Select the subfolder** in Prep Audio Sample, switch to Dataset and select your subfolder from the Dataset Folder dropdown

        &nbsp;

        #### Individual File Editing:
        - **Load audio**: Select from dropdown to edit individual files
        - **Trim audio**: Use waveform editor to cut unwanted sections
        - **Save trimmed**: Save changes if you edited the audio
        - **Transcribe**: Auto-transcribe or manually type the transcript
        - **Save transcript**: Save the text file with matching filename

        &nbsp;

        #### Batch Transcription:
        - **Configure settings**: Select transcription model and language
        - **Replace existing**: Check to re-transcribe all files (skip unchecks to keep existing)
        - **Batch Transcribe**: Process all audio files at once
        - **Great for**: Processing large datasets quickly

        &nbsp;

        #### Format Requirements:
        - **Audio**: 24kHz, 16-bit, mono WAV (auto-converted during training setup)
        - **Transcript**: Exact text spoken in the audio
        - **Recommendation**: Use the same reference audio for all samples

        &nbsp;

        #### Dataset Structure:
        ```
        datasets/
          my_voice/
            audio_001.wav
            audio_001.txt
            audio_002.wav
            audio_002.txt
            ...
        ```

        &nbsp;

        #### Best Practices:
        - **50-100 samples**: Ideal for training
        - **5-15 seconds each**: Good length per sample
        - **Diverse content**: Different sentences, emotions
        - **Consistent quality**: Same mic, environment
        - **Accurate transcriptions**: Critical for training success
        - **Clean audio**: Minimal background noise

        &nbsp;

        #### Next Steps:
        Once your dataset is ready, go to the **Train Model** tab to prepare your dataset and start training.

        &nbsp;

        #### Transcription Models:
        - **Whisper**: High accuracy, many languages, good for single speakers
        - **VibeVoice ASR**: Better for conversations, multi-speaker support
        """)


def show_train_help():
    """Return help content for Train Model tab."""
    return dedent("""
        ### 🏋️ Train Model

        **Train custom voice models on your own voice data**

        &nbsp;

        #### Training Workflow:
        1. **Select dataset folder**: Choose your prepared dataset (50-100 audio clips recommended)
        2. **Enter speaker name**: Give your trained model a unique name
        3. **Select reference audio**: Pick one sample (5-10 seconds, clear quality)
        4. **Configure parameters**: Adjust training settings (defaults work well)
        5. **Start training**: Click "Start Training" and wait for completion

        &nbsp;

        #### After Training:
        Your model will be saved in `trained_models/{speaker_name}/`

        **Use it with:**
        Voice Presets tab → Trained Models → Select Speaker Name

        &nbsp;

        #### Training Parameters:
        - **Model Size**: 1.7B (Large) for best quality, 0.6B (Small) for faster training
        - **Batch Size**: Number of samples per training step (reduce if out of memory)
        - **Learning Rate**: Controls training speed (default: 2e-5 works well)
        - **Epochs**: How many times to train on full dataset (3-5 recommended)

        &nbsp;

        #### Hardware Requirements:
        - **VRAM**: 8GB+ for Small (0.6B), 16GB+ for Large (1.7B)
        - **Storage**: Several GB for checkpoints and model files
        - **Time**: Hours depending on dataset size and hardware

        &nbsp;

        #### Dataset Requirements:
        - **50-100 samples**: Ideal training size
        - **Consistent quality**: Same recording setup throughout
        - **Accurate transcripts**: Critical for training success
        - **Reference audio**: One high-quality sample (5-10s) for voice characteristics

        &nbsp;

        #### Monitoring Training:
        - Watch training status in real-time
        - Model checkpoints saved per epoch
        - Later epochs usually perform better
        - Test checkpoints in Voice Presets tab

        &nbsp;

        #### Best Practices:
        - Start with default parameters
        - Use high-quality reference audio
        - Ensure dataset is well-prepared (Prep Audio Samples / dataset tab)
        - Don't interrupt training mid-epoch
        - Keep multiple checkpoints for comparison

        &nbsp;

        #### Troubleshooting:
        - **Out of memory**: Reduce batch size or use smaller model
        - **Poor quality**: Check dataset quality and transcription accuracy
        - **Slow training**: Normal for large datasets, be patient
        - **Model not improving**: May need more epochs or better data

        &nbsp;

        #### Use Cases:
        - Create personal voice assistant
        - Train on specific accent or dialect
        - Improve quality for specialized content
        - Create character voices for games/content
        - Clone voice with limited reference samples
        """)


def show_tips_help():
    """Return help content for Tips & Tricks."""
    return dedent("""
        ### 💡 Tips & Tricks

        **Get the best results from Voice Clone Studio**

        &nbsp;

        #### Performance Optimization:
        - **First-time generation** creates cached prompts - be patient
        - **Use Large models** for best quality (if you have VRAM)
        - **Close other GPU apps** to free VRAM
        - **Monitor VRAM usage** in Task Manager

        &nbsp;

        #### Quality Tips:
        - **Clean samples**: Less background noise = better clones
        - **Clear speech**: Well-articulated reference audio
        - **Match length**: Reference sample length affects style
        - **Consistent audio**: Same recording setup for datasets

        &nbsp;

        #### Workflow Tips:
        - **Save designs you like** to use again later
        - **Organize samples** with descriptive names
        - **Use batch transcription** for large datasets
        - **Experiment with seeds** for variations

        &nbsp;

        #### Model Selection Guide:
        - **Voice Clone (Qwen3)**: all-around voice cloning
        - **Voice Clone (VibeVoice)**: Best for long-form content
        - **Voice Presets**: Quick, high-quality results
        - **Voice Design**: Creative new voices
        - **Conversation (Qwen3)**: Fast multi-speaker dialogues
        - **Conversation (VibeVoice)**: Long-form, custom voices

        &nbsp;

        #### Troubleshooting:
        - **Out of VRAM**: Use Small models, close other apps
        - **Poor quality**: Use better reference samples
        - **Slow generation**: Normal for first time, fast after caching
        - **Model not loading**: Check CUDA installation

        &nbsp;

        #### Language Support:
        - **Auto-detect** works for most cases
        - **Specify language** for better accuracy
        - **Match speaker language** with text language
        - **Multi-lingual**: Use appropriate voice presets

        &nbsp;

        #### Advanced Features:
        - **Style instructions**: Add emotions to conversations
        - **CFG scale**: Higher = more guidance (VibeVoice)
        - **Seed control**: Reproducible generation
        - **Batch processing**: Automate repetitive tasks
        """)


class HelpGuideTool(Tool):
    """Help Guide tool implementation."""

    config = ToolConfig(
        name="Help Guide",
        module_name="tool_help_page",
        description="Documentation and usage tips",
        enabled=True,
        category="utility",
    )

    @classmethod
    def create_content(cls, shared_state):
        """Create Help Guide content without a container wrapper.

        Use this when rendering help outside of the main tab bar
        (e.g. in an Accordion at the bottom of the page).
        """
        components = {}
        format_help_html = shared_state.get("format_help_html")

        gr.Markdown("# Voice Clone Studio - Help & Guide")

        components["help_topic"] = gr.Radio(
            choices=[
                "Voice Clone",
                "Voice Presets",
                "Conversation",
                "Voice Design",
                "Prep Samples",
                "Prep Dataset",
                "Train Model",
                "Tips & Tricks",
            ],
            value="Voice Clone",
            show_label=False,
            interactive=True,
            container=False,
        )

        components["help_content"] = gr.HTML(
            value=format_help_html(show_voice_clone_help()),
            container=True,
            padding=True,
        )

        return components

    @classmethod
    def create_tool(cls, shared_state):
        """Create Help Guide tool UI."""
        components = {}

        # Extract needed items from shared_state
        format_help_html = shared_state.get("format_help_html")

        with gr.TabItem("Help Guide"):
            gr.Markdown("# Voice Clone Studio - Help & Guide")

            components["help_topic"] = gr.Radio(
                choices=[
                    "Voice Clone",
                    "Voice Presets",
                    "Conversation",
                    "Voice Design",
                    "Prep Samples",
                    "Prep Dataset",
                    "Train Model",
                    "Tips & Tricks",
                ],
                value="Voice Clone",
                show_label=False,
                interactive=True,
                container=False,
            )

            components["help_content"] = gr.HTML(
                value=format_help_html(show_voice_clone_help()),
                container=True,
                padding=True,
            )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Help Guide events."""

        # Extract needed items from shared_state
        format_help_html = shared_state.get("format_help_html")

        def show_help(topic):
            """Show help for selected topic."""
            help_map = {
                "Voice Clone": show_voice_clone_help,
                "Conversation": show_conversation_help,
                "Voice Presets": show_voice_presets_help,
                "Voice Design": show_voice_design_help,
                "Prep Samples": show_prep_audio_help,
                "Prep Dataset": show_dataset_help,
                "Train Model": show_train_help,
                "Tips & Tricks": show_tips_help,
            }
            return format_help_html(help_map[topic]())

        # Event handler for radio selection
        components["help_topic"].change(
            fn=show_help,
            inputs=components["help_topic"],
            outputs=components["help_content"],
        )


# Export for tab registry
get_tool_class = lambda: HelpGuideTool


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone

    run_tool_standalone(HelpGuideTool, port=7869, title="Help Guide - Standalone")
