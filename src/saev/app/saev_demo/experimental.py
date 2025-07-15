import gradio as gr
import os
import json
from google import genai
from google.genai import types
from google.genai import errors
import time

DEFAULT_PROMPT = "The following was identified as a *blank* the bounding boxes shown are the deciding factor for how the animal was identified. Based on this, describe why the organism was identified the way it was."

def generate_content_str(api_key, prompt, pil_image, retries):
    client = genai.Client(api_key=api_key)
    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
    )
    while True:
        try:
            response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[prompt, pil_image],
            config = generate_content_config,
            )
            print("Result", response.text)
            return response.text
        
        except errors.ServerError as e:
            retries -= 1
            if retries == 0:
                raise e
            print(f"Retrying... {e}")
            time.sleep(5)

with gr.Blocks(title="Features Demo") as demo:
    gr.Markdown("# Image Analysis")
    

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload an image and enter a prompt to get predictions")
            api_key_input = gr.Textbox(
                label="Gemini API Key",
                placeholder="Enter your Gemini API key here...",
                type="password",
            )

            image_input = gr.Image(label="Upload an Image", type="pil"
            )
        
            gr.Markdown("The prompt below must request bounding boxes.")
            prompt_input = gr.TextArea(
                label="Enter your prompt",
                placeholder="Describe what you want to analyze...",
                value=DEFAULT_PROMPT,
            )
            submit_btn = gr.Button("Analyze")
            # gr.Markdown("## Results")
            output = gr.Markdown("## Gemini Description")
        #     gr.Markdown("## Cropped Images")
        #     image_gallery = gr.Gallery(label="Images", show_label=True)



        # with gr.Column():
            
        submit_btn.click(
            fn=generate_content_str,
            inputs=[api_key_input, prompt_input, image_input],
            outputs=[output]
        )

if __name__ == "__main__":
    demo.launch()