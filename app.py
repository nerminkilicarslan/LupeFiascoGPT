import gradio as gr
import os
import torch
from generate import load_model, generate_text

# ---------- Model ----------
MODEL_PATH = "checkpoints/model.pt"
model, stoi, itos = None, None, None

def ensure_model():
    global model, stoi, itos
    if model is None:
        if not os.path.exists(MODEL_PATH):
            return False
        model, stoi, itos = load_model(MODEL_PATH)
    return True

# ---------- Preset prompts ----------
PRESETS = [
    "I used to",
    "The city never sleeps",
    "Food and Liquor",
    "Never",
]

# ---------- Core function ----------
def generate(prompt, max_tokens):
    if not ensure_model():
        return (
            "Model not trained yet.\n\n"
            "Run the following steps first:\n"
            "  1. python collect_lyrics.py\n"
            "  2. python train.py\n\n"
            "Refresh this page after training completes."
        )
    if not prompt or not prompt.strip():
        prompt = "I"
    return generate_text(
        model, stoi, itos,
        prompt=prompt.strip(),
        max_tokens=int(max_tokens),
        temperature=0.8,
        top_k=50,
    )

# ---------- Custom CSS ----------
CSS = """
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0a0a !important;
    color: #e8e8e8 !important;
    font-family: 'Courier New', Courier, monospace !important;
}

/* ── Header banner ── */
#lupe-header {
    text-align: center;
    padding: 1.2rem 1rem 1rem;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 1.2rem;
    background: #0a0a0a;
}
#lupe-header .mic-icon {
    font-size: 2.6rem;
    display: block;
    margin-bottom: 0.4rem;
    filter: grayscale(1);
}
#lupe-header h1 {
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 0.18em;
    color: #ffffff;
    margin: 0 0 0.3rem;
}
#lupe-header .sub {
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #555;
    margin: 0 0 0.8rem;
}
#lupe-header .albums {
    font-size: 0.65rem;
    color: #333;
    letter-spacing: 0.12em;
}

/* ── Panels ── */
.gr-panel, .gr-box, .gr-form,
div[class*="block"], .gradio-container .prose {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 4px !important;
}

/* ── Labels ── */
label span, .gr-label {
    color: #888 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-family: 'Courier New', Courier, monospace !important;
}

/* ── Textboxes ── */
textarea, input[type="text"] {
    background: #0e0e0e !important;
    color: #e8e8e8 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 3px !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 0.88rem !important;
    caret-color: #fff;
}
textarea:focus, input[type="text"]:focus {
    border-color: #555 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ── Sliders ── */
input[type="range"] {
    accent-color: #ffffff;
}
.gr-slider-value, span[class*="value"] {
    color: #aaa !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 0.75rem !important;
}

/* ── Preset buttons ── */
.preset-btn button {
    background: #111 !important;
    color: #666 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 3px !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    font-family: 'Courier New', Courier, monospace !important;
    padding: 4px 10px !important;
    transition: all 0.15s ease;
    cursor: pointer;
}
.preset-btn button:hover {
    background: #1c1c1c !important;
    color: #ccc !important;
    border-color: #444 !important;
}

/* ── Generate button ── */
#gen-btn button {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 3px !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    font-size: 0.78rem !important;
    font-family: 'Courier New', Courier, monospace !important;
    padding: 12px 32px !important;
    transition: opacity 0.15s ease;
    width: 100%;
}
#gen-btn button:hover { opacity: 0.88; }

/* ── Output box ── */
#output-box textarea {
    background: #080808 !important;
    color: #d4d4d4 !important;
    border: 1px solid #1e1e1e !important;
    font-size: 0.9rem !important;
    line-height: 1.75 !important;
    letter-spacing: 0.01em !important;
}

/* ── Footer ── */
#lupe-footer {
    text-align: center;
    padding: 1.4rem 1rem;
    border-top: 1px solid #1a1a1a;
    margin-top: 2rem;
    font-size: 0.62rem;
    color: #2d2d2d;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
"""

# ---------- Build UI ----------
with gr.Blocks(css=CSS, title="Lupe Fiasco GPT") as demo:

    # Header
    gr.HTML("""
    <div id="lupe-header">
        <span class="mic-icon">🎤</span>
        <h1>LUPE FIASCO GPT</h1>
        <p class="sub">Character-level language model · trained on rap lyrics</p>
        <p class="albums">
            Food &amp; Liquor &nbsp;·&nbsp; The Cool &nbsp;·&nbsp; Lasers &nbsp;·&nbsp;
            Tetsuo &amp; Youth &nbsp;·&nbsp; Drill Music in Zion
        </p>
    </div>
    """)

    with gr.Row():
        # ── Left column: controls ──
        with gr.Column(scale=1, min_width=280):

            prompt_input = gr.Textbox(
                label="Starting line",
                placeholder="I used to...",
                lines=2,
            )

            gr.Markdown(
                "<p style='font-size:0.65rem;color:#444;letter-spacing:0.12em;"
                "text-transform:uppercase;margin:8px 0 4px;'>Quick starters</p>"
            )
            with gr.Row():
                for p in PRESETS[:2]:
                    btn = gr.Button(p, elem_classes="preset-btn")
                    btn.click(fn=lambda x=p: x, inputs=None, outputs=prompt_input)
            with gr.Row():
                for p in PRESETS[2:]:
                    btn = gr.Button(p, elem_classes="preset-btn")
                    btn.click(fn=lambda x=p: x, inputs=None, outputs=prompt_input)

            max_tokens = gr.Number(
                value=300,
                label="Length  (characters)",
                precision=0,
                minimum=80,
                maximum=1200,
            )

            gen_btn = gr.Button("GENERATE", elem_id="gen-btn")

        # ── Right column: output ──
        with gr.Column(scale=2):
            output_box = gr.Textbox(
                label="Generated lyrics",
                lines=22,
                interactive=False,
                elem_id="output-box",
            )

    # Wiring
    gen_btn.click(
        fn=generate,
        inputs=[prompt_input, max_tokens],
        outputs=output_box,
    )
    prompt_input.submit(
        fn=generate,
        inputs=[prompt_input, max_tokens],
        outputs=output_box,
    )

    # Footer
    gr.HTML("""
    <div id="lupe-footer">
        Lupe Fiasco GPT &nbsp;·&nbsp; Washika Yusuf Jaco Ngozi Fiasco
        &nbsp;·&nbsp; Chicago, Illinois
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, inbrowser=True)
