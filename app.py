"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import torch
import requests
import datetime
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
MODEL = None
def my_get(url, **kwargs):
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)
original_get = requests.get
requests.get = my_get
import gradio as gr
requests.get = original_get
def load_model(version):
    print("Loading model", version)
    return MusicGen.get_pretrained(version)
def generate(model, text, melody, duration, topk, topp, temperature, cfg_coef,base_duration, sliding_window_seconds):
    
    final_length_seconds = duration
    descriptions = text
    global MODEL
    topk = int(topk)
    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model)
    if duration > 30:
        MODEL.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=base_duration,
        )
    else:
        MODEL.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=duration,
        )
    iterations_required = int(final_length_seconds / sliding_window_seconds)
    
    print(f"Iterations required: {iterations_required}")
    sr = MODEL.sample_rate
    print(f"Sample rate: {sr}")
    msr=None
    wav = None # wav shape will be [1, 1, sr * seconds]
    melody_boolean = False
    if melody:
        print("test")
        msr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
        print(melody.shape)
        if melody.dim() == 2:
            melody = melody[None]
        melody = melody[..., :int(msr * MODEL.lm.cfg.dataset.segment_duration)]
        melody_boolean = True
    
    if(duration > 30):
        for i in range(iterations_required):
                print(f"Generating {i + 1}/{iterations_required}")
                if i == 0:
                    if melody_boolean:
                        wav = MODEL.generate_with_chroma(
                            descriptions=[text],
                            melody_wavs=melody,
                            melody_sample_rate=msr,
                            progress=False
                        )
                    else:
                        wav = MODEL.generate(descriptions=[text], progress=False)
                    # take only first sliding_window_seconds, sometimes the model generates fading-out music, which results in a continuation into silence 
                    wav = wav[:, :, :sr * sliding_window_seconds]
                else:
                    new_chunk=None
                    previous_chunk = wav[:, :, -sr * (base_duration - sliding_window_seconds):]
                    new_chunk = MODEL.generate_continuation(previous_chunk, descriptions=[text], prompt_sample_rate=sr,progress=False)
                    wav = torch.cat((wav, new_chunk[:, :, -sr * sliding_window_seconds:]), dim=2)
    else:
        if melody_boolean:
            wav = MODEL.generate_with_chroma(
                descriptions=[text],
                melody_wavs=melody,
                melody_sample_rate=msr,
                progress=False
            )
        else:
            wav = MODEL.generate(descriptions=[text], progress=False)
    print(f"Final length: {wav.shape[2] / sr}s")

    output = wav.detach().cpu().numpy()
    return MODEL.sample_rate, output


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(
        """
        # MusicGen

        This is a webui for MusicGen with 30+ second generation support.
        
        Models
        1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
        2. Small -- a 300M transformer decoder conditioned on text only.
        3. Medium -- a 1.5B transformer decoder conditioned on text only.
        4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.)

        When the optional melody conditioning wav is provided, the model will extract
        a broad melody and try to follow it in the generated samples. Only the first chunk of the song will
        be generated with melody conditioning, the others will just continue on the first chunk.

        Base duration of 30 seconds is recommended.
        
        Sliding window of 10/15/20 seconds is recommended.

        Gradio analytics are disabled.
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                text = gr.Text(label="Input Text", interactive=True)
                melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
            with gr.Row():
                submit = gr.Button("Submit")
            with gr.Row():
                model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
            with gr.Row():
                duration = gr.Slider(minimum=1, maximum=300, value=60, label="Duration", interactive=True)
                base_duration = gr.Slider(minimum=1, maximum=30, value=30, label="Base duration", interactive=True)
                sliding_window_seconds=gr.Slider(minimum=1, maximum=30, value=15, label="Sliding window", interactive=True)
            with gr.Row():
                topk = gr.Number(label="Top-k", value=250, interactive=True)
                topp = gr.Number(label="Top-p", value=0, interactive=True)
                temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
        with gr.Column():
            output = gr.Audio(label="Generated Music", type="numpy")
    submit.click(generate, inputs=[model, text, melody, duration, topk, topp, temperature, cfg_coef,base_duration, sliding_window_seconds], outputs=[output])
    gr.Examples(
        fn=generate,
        examples=[
            [
                "An 80s driving pop song with heavy drums and synth pads in the background",
                "./assets/bach.mp3",
                "melody"
            ],
            [
                "A cheerful country song with acoustic guitars",
                "./assets/bolero_ravel.mp3",
                "melody"
            ],
            [
                "90s rock song with electric guitar and heavy drums",
                None,
                "medium"
            ],
            [
                "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                "./assets/bach.mp3",
                "melody"
            ],
            [
                "lofi slow bpm electro chill with organic samples",
                None,
                "medium",
            ],
        ],
        inputs=[text, melody, model],
        outputs=[output]
    )





demo.launch()
