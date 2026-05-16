#!/usr/bin/env python3
"""DramaBox — Gradio demo (warm server).

Loads the warm TTSServer once, then handles requests at ~2.5 s each. All
generated audio is invisibly watermarked with Resemble Perth before being
returned to the user.
"""
import logging
import os
import sys
import tempfile
import time

import gradio as gr
import spaces

# Local src import.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from inference_server import TTSServer  # noqa: E402
from model_downloader import get_all_paths  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info("Fetching DramaBox checkpoints from HuggingFace (cached after first run)...")
PATHS = get_all_paths()

# Module-level warm load (same pattern as IndexTTS-2-Demo on ZeroGPU). The
# `spaces` package patches torch so that .to("cuda") at import time pins the
# weights into ZeroGPU's shared memory; each @spaces.GPU call then maps them
# onto the actual GPU instantly. First user request is ~2.5 s instead of ~30 s.
logging.info("Loading DramaBox warm server (Gemma + DiT + VAE + Decoder)...")
tts = TTSServer(
    checkpoint=PATHS["transformer"],
    full_checkpoint=PATHS["audio_components"],
    gemma_root=PATHS["gemma_root"],
    device="cuda",
    dtype=os.environ.get("LTX_DTYPE", "bf16"),
    compile_model=False,                  # torch.compile breaks under ZeroGPU's brief GPU windows
    bnb_4bit=True,                        # unsloth Gemma is pre-quantized
)
logging.info("TTSServer ready.")


# ── Example prompts shipped with a matching voice reference ──────────────────
# Files live under assets/voices/ so users can click a row and generate
# without uploading anything.
_VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "voices")

EXAMPLES: list[tuple[str, str, str]] = [
    (
        "Villain monologue",
        os.path.join(_VOICES_DIR, "male_harvey_keitel.mp3"),
        'A shadowy villain speaks with cold menace, "You have entered my domain, mortal." '
        'He chuckles darkly, "Such arrogance will be your undoing." '
        'His voice rises with fury, "Kneel, or be destroyed where you stand!"',
    ),
    (
        "Talk-show host wheeze-laugh",
        os.path.join(_VOICES_DIR, "male_conan.mp3"),
        'A talk show host gasps with shock, "No! You did NOT just say that!" '
        'He bursts into uncontrollable laughter, "Hahaha! Oh my god, oh my god!" '
        'He wheezes, "I cannot, I literally cannot breathe right now!"',
    ),
    (
        "Tender goodnight whisper",
        os.path.join(_VOICES_DIR, "female_shadowheart.wav"),
        'A woman speaks tenderly, "It has been a long day, my love." '
        'She whispers, "Close your eyes. I am right here." '
        'She hums quietly, "Mmmm-mmm. Sleep now."',
    ),
    (
        "Old-school radio anchor",
        os.path.join(_VOICES_DIR, "male_old_movie.wav"),
        'A radio host clears his throat, "Excuse me, pardon that." '
        'He settles into a warm, professional tone, "Good evening everyone, '
        'and welcome back to the show. We have got a wonderful lineup tonight."',
    ),
    (
        "Catgirl uncontrollable giggling",
        os.path.join(_VOICES_DIR, "female_american.wav"),
        'A playful girl already mid-giggle, "Hehehe, oh my gosh you should see your face!" '
        'She gasps for air between giggles, "Oh my, hehe, oh my, I cannot stop!" '
        'She tries to compose herself, "Ahhhhh okay okay okay, I will stop, I promise."',
    ),
    (
        "Hero stammering courage",
        os.path.join(_VOICES_DIR, "male_arnie.mp3"),
        'A young warrior speaks with a trembling voice, "I... I do not know if I can do this." '
        'He takes a shaky breath, "But someone has to try." '
        'His voice steadies with growing fire, "No more running. I WILL fight!"',
    ),
    (
        "Exhausted dad, fraying patience",
        os.path.join(_VOICES_DIR, "male_petergriffin.wav"),
        'An exhausted father speaks with fraying patience, "Sweetie, daddy is asking very nicely." '
        'He sighs deeply, "Ohhhh my goodness." '
        'He puts on an overly cheerful voice, "Hey buddy! Look at the shiny thing!" '
        'Then he laughs helplessly, "Hahaha, I am losing my mind."',
    ),
    (
        "Smug-confident announcer",
        os.path.join(_VOICES_DIR, "male_samuel_j.mp3"),
        'A confident announcer speaks proudly, "And now, the moment you have all been waiting for." '
        'He chuckles knowingly, "Heheh, trust me, this one is going to blow you away."',
    ),
    # ── Long-form examples (~30 s each) ───────────────────────────────────────
    # These pair a richer multi-beat scene with gen_duration = 30 s in the
    # Examples row below so the model is asked for a full half-minute clip.
    (
        "30s • Villain soliloquy",
        os.path.join(_VOICES_DIR, "male_harvey_keitel.mp3"),
        'A shadowy villain stands at the edge of his throne room, gazing into the dark. '
        'He speaks with slow, measured menace, "So, the little hero has come to finish me, has he?" '
        'He chuckles low and humourless, "Hehe, oh how delightfully predictable you mortals are." '
        'His voice hardens into ice, "I have lived ten thousand years. I have seen empires rise and fall like the tide." '
        'He scoffs, "And you think you, with your borrowed sword and your trembling hands, will be the one to end me?" '
        'A long pause. He whispers, almost tenderly, "I will give you a single chance to turn around and walk away." '
        'Then his voice rises with crushing finality, "Choose, child. The door behind you, or the grave at your feet."',
    ),
    (
        "30s • Late-night radio monologue",
        os.path.join(_VOICES_DIR, "male_old_movie.wav"),
        'A radio host clears his throat softly into the microphone in the late hours of the night. '
        'He settles into a warm, smoky tone, "Good evening, dear listeners, and welcome back to the After Hours Hour." '
        'He sighs contentedly, "Mmm, what a night it has been. The rain is tapping at my window like an old friend." '
        'He chuckles softly, "Heheh, you know the kind of friend, the one that always shows up unannounced." '
        'His voice drops, intimate, "I want you to lean back, wherever you are. Pour yourself something warm." '
        'He pauses, breath audible, "Tonight we are going to talk about love, and loss, and the songs that hold us together." '
        'A smile in his voice, "And I have got the perfect record cued up to start us off, so stay right where you are."',
    ),
    (
        "30s • Stand-up wheeze-laugh",
        os.path.join(_VOICES_DIR, "male_conan.mp3"),
        'A talk show host walks out and the crowd is already roaring. He gasps in mock outrage, "No! No no no!" '
        'He bursts into uncontrollable laughter, "Hahahaha, oh my god, oh my god, you cannot do that to me already!" '
        'He wheezes, gasping for air, "I have not even, hahaha, I have not even said hello yet!" '
        'He tries to compose himself, "Okay, okay, just give me a second here, give me a second." '
        'He clears his throat dramatically, "Ahem. Good evening, ladies and gentlemen." '
        'Then he loses it again, "Hahaha! No, sorry, sorry, I just remembered what happened in the green room." '
        'He pants, "Oh man, oh man, this is going to be one of those nights, I can already tell."',
    ),
    (
        "30s • Bedtime story",
        os.path.join(_VOICES_DIR, "female_shadowheart.wav"),
        'A mother sits at the edge of her child\'s bed in the dim glow of a single lamp. '
        'She speaks softly, "Once upon a time, in a kingdom by the sea, there lived a small dragon named Pip." '
        'She lowers her voice playfully, "Now Pip was not like the other dragons. Pip was afraid of fire." '
        'She smiles warmly, "Mmm, can you imagine? A dragon who was afraid of his own breath?" '
        'A gentle pause, "But Pip had something the other dragons did not have. Pip had courage in his heart." '
        'She hums softly, "Mmmmm. And one cold winter night, when the village below ran out of warmth..." '
        'Her voice drops to a whisper, "Pip closed his eyes, took a deep, deep breath, and remembered who he was."',
    ),
    (
        "30s • Sports commentary",
        os.path.join(_VOICES_DIR, "male_samuel_j.mp3"),
        'A sports commentator leans into the microphone with the crowd roaring around him. '
        'He shouts with rising energy, "Oh, this is it! This is the moment we have been waiting for all season!" '
        'He pants between phrases, "She has the ball at midfield, she is dribbling past one, past two!" '
        'A sudden gasp, "Oh my, what a move! Did you see that footwork, ladies and gentlemen?" '
        'His voice climbs, "She is twenty yards out, fifteen yards out, she winds back, and she SHOOTS!" '
        'A massive pause, then, "GOAAAAAAL! What a strike! What an absolute thunderbolt of a goal!" '
        'He laughs in disbelief, "Hahaha! Unbelievable! Forty thousand fans on their feet, and so am I!"',
    ),
]


@spaces.GPU(duration=120)
def on_generate(prompt: str, audio_ref, cfg: float, stg: float, dur_mult: float,
                gen_dur: float, ref_dur: float, seed: int):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is empty.")
    t0 = time.time()
    ref_path = audio_ref if audio_ref and os.path.exists(str(audio_ref)) else None
    output = tempfile.mktemp(suffix=".wav", prefix="dramabox_", dir="output")
    tts.generate_to_file(
        prompt=prompt,
        output=output,
        voice_ref=ref_path,
        cfg_scale=cfg, stg_scale=stg,
        duration_multiplier=dur_mult, seed=int(seed),
        gen_duration=float(gen_dur),
        ref_duration=float(ref_dur),
    )
    elapsed = time.time() - t0
    logging.info(f"Generated in {elapsed:.2f}s -> {output}")
    return output


# ── UI ──────────────────────────────────────────────────────────────────────
_BANNER_CSS = """
.prompt-box textarea { font-size: 14px !important; line-height: 1.5 !important; }
.ltx-banner {
    background: linear-gradient(90deg, #1a1f3a 0%, #2a1f3a 100%);
    border-left: 4px solid #ff6b35;
    padding: 10px 16px;
    margin: 0 0 12px 0;
    border-radius: 6px;
    color: #e8e8f0;
    font-size: 13px;
    line-height: 1.5;
}
.ltx-banner a { color: #ff9a6c; font-weight: 600; text-decoration: none; }
.ltx-banner a:hover { text-decoration: underline; }
.ltx-banner strong { color: #ffffff; }
"""

with gr.Blocks(
    title="DramaBox — Expressive TTS",
    theme=gr.themes.Default(),
    css=_BANNER_CSS,
    analytics_enabled=False,
) as app:
    gr.Markdown("# 🎭 DramaBox — Expressive TTS with Voice Cloning")
    gr.HTML(
        '<div class="ltx-banner">'
        '🏗️&nbsp; Built on <a href="https://github.com/Lightricks/LTX-2">LTX-2</a> by '
        '<a href="https://huggingface.co/Lightricks">Lightricks</a>. '
        '<strong>DramaBox</strong> is <strong>Resemble AI\'s</strong> expressive TTS, '
        'trained on top of the LTX-2.3 audio branch under the LTX-2 Community License. '
        'Huge thanks to the Lightricks team for open-sourcing the base.'
        '</div>'
    )
    gr.Markdown(
        "Write a scene prompt, optionally upload a 10-second voice reference, "
        "and generate. Audio is automatically watermarked with "
        "[Resemble Perth](https://github.com/resemble-ai/Perth).\n\n"
        "**Tips:** put dialogue inside `\"double quotes\"`, scene directions outside. "
        "Phonetic sounds (`\"Hahaha\"`, `\"Mmmm\"`, `\"Ugh\"`) go inside quotes; named "
        "actions (`She sighs.`, `He clears his throat.`) go outside."
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_box = gr.Textbox(
                label="Scene prompt",
                placeholder=EXAMPLES[0][2],
                lines=6, elem_classes=["prompt-box"],
            )
            audio_ref = gr.Audio(
                label="Voice reference (optional, 10+ seconds)",
                type="filepath",
            )
            gen_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Accordion("Inference settings", open=False):
                cfg_slider = gr.Slider(1.0, 10.0, value=2.5, step=0.5, label="CFG scale")
                stg_slider = gr.Slider(0.0, 5.0, value=1.5, step=0.5, label="STG scale")
                dur_slider = gr.Slider(0.8, 2.0, value=1.1, step=0.05,
                                       label="Duration × (only used when target duration = 0)")
                gen_dur_slider = gr.Slider(0.0, 60.0, value=0.0, step=1.0,
                                           label="Target duration (s) — 0 = auto from prompt; "
                                                 "set higher (≥20s) for long-form music or scenes")
                ref_dur_slider = gr.Slider(3.0, 30.0, value=10.0, step=1.0,
                                           label="Reference duration (s) — how many seconds of the "
                                                 "uploaded voice reference the model conditions on")
                seed_input = gr.Number(value=42, label="Seed", precision=0)
            audio_out = gr.Audio(label="Generated audio", type="filepath")
            with gr.Accordion("Prompt writing guide", open=False):
                gr.Markdown(
                    "**Structure:** `<speaker description>, \"<dialogue>\" <action> \"<more dialogue>\"`\n\n"
                    "**Inside quotes** (model speaks them):\n"
                    "- Dialogue: `\"Hello, how are you?\"`\n"
                    "- Phonetic sounds: `\"Hahaha\"`, `\"Hehehe\"`, `\"Mmmmm\"`, `\"Ugh\"`, `\"Argh\"`\n\n"
                    "**Outside quotes** (stage directions):\n"
                    "- `She sighs deeply.`, `He gulps nervously.`, `A long pause.`\n"
                    "- `Her voice cracks.`, `He clears his throat.`\n\n"
                    "**Avoid inside quotes:** Ahem, Pfft, Sigh, Gasp, Cough — the model speaks them literally."
                )

    gen_btn.click(
        on_generate,
        inputs=[prompt_box, audio_ref, cfg_slider, stg_slider,
                dur_slider, gen_dur_slider, ref_dur_slider, seed_input],
        outputs=[audio_out],
    )

    # Click-to-generate example table. Each row preloads a paired voice
    # reference + prompt and runs the model immediately.
    gr.Examples(
        label="🎬 Click any row to generate a sample",
        examples=[
            # rows tagged "30s •" force a 30-second target duration; the rest
            # use the prompt-driven auto estimate (gen_dur = 0).
            [name, prompt, voice_path, 2.5, 1.5, 1.1,
             30.0 if name.startswith("30s") else 0.0, 10.0, 42]
            for name, voice_path, prompt in EXAMPLES
        ],
        example_labels=[name for name, _, _ in EXAMPLES],
        inputs=[gr.Textbox(visible=False, label="Scene"),
                prompt_box, audio_ref,
                cfg_slider, stg_slider, dur_slider, gen_dur_slider,
                ref_dur_slider, seed_input],
        outputs=[audio_out],
        fn=lambda _name, prompt, ref, cfg, stg, dur, gen_dur, ref_dur, seed: on_generate(
            prompt, ref, cfg, stg, dur, gen_dur, ref_dur, seed),
        cache_examples=False,
        run_on_click=True,
        examples_per_page=20,
    )


if __name__ == "__main__":
    # HF Spaces routes external traffic to container port 7860 by default.
    # Defaulting to 7861 caused the gateway to return 500 for every external request.
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app.queue(max_size=10).launch(
        server_name="0.0.0.0", server_port=port,
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        ssr_mode=False,                       # Gradio 5 SSR + ZeroGPU fork has known race conditions
        show_api=False,                       # don't auto-derive Python schemas (caused bool-iter / dict-cache crashes)
    )
