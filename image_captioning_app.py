import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(inp_image: np.ndarray):
    raw_image = Image.fromarray(inp_image).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption using the BLIP model."
)

iface.launch()
