import os
from collections.abc import Iterator
from threading import Thread

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DESCRIPTION = """
# GWQ PREV
"""

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_id = "prithivMLmods/GWQ2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.sliding_window = 4096
model.eval()


@spaces.GPU(duration=120)
def generate(
    message: str,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    conversation = chat_history.copy()
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


demo = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=None,
    examples=[
        ["Write a Python function to reverses a string if it's length is a multiple of 4. def reverse_string(str1): if len(str1) % 4 == 0: return ''.join(reversed(str1)) return str1 print(reverse_string('abcd')) print(reverse_string('python')) "],
        ["Rectangle $ABCD$ is the base of pyramid $PABCD$. If $AB = 10$, $BC = 5$, $\overline{PA}\perp \text{plane } ABCD$, and $PA = 8$, then what is the volume of $PABCD$?"],
        ["Difference between List comprehension and Lambda in Python lst  =  [x ** 2  for x in range (1, 11)   if  x % 2 == 1] print(lst)"],
        ["What happens when the sun goes down?"],
    ],
    cache_examples=False,
    type="messages",
    description=DESCRIPTION,
    css_paths="style.css",
    fill_height=True,
)


if __name__ == "__main__":
    demo.queue(max_size=20).launch()
