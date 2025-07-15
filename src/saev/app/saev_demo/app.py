import gradio as gr
import os
import json
from google import genai
from google.genai import types
from google.genai import errors
import time
from bioclip import TreeOfLifeClassifier, Rank

DEFAULT_PROMPT = """
Create bounding boxes for the 5 most important morphological traits that helped you successfully identify the organism shown. Be sure to respond using valid JSON.""".strip()

classifier = TreeOfLifeClassifier()

def crop_image(image, gemini_bounding_box):
    width, height = image.size
    y_min, x_min, y_max, x_max = gemini_bounding_box

    left = int(x_min / 1000 * width)
    upper = int(y_min / 1000 * height)
    right = int(x_max /1000 * width)
    lower = int(y_max / 1000 * height)

    return image.crop((left, upper, right, lower))

def make_crops(image, predictions_json_txt):
    """
    Process predictions to crop images based on bounding boxes.
    :param image: PIL Image object
    :param predictions: str of JSON List of prediction dictionaries containing bounding boxes
    :return: List of cropped images
    """
    cropped_images = []
    try:
        print('Output for testing', predictions_json_txt)
        predictions = json.loads(predictions_json_txt)
    except json.JSONDecodeError as e:
        print(str(e))
        return []  # Return empty list if JSON parsing fails

    for prediction in predictions:
        if "box_2d" in prediction:
            gemini_bounding_box = prediction["box_2d"]
            # Crop the image using the bounding box
            try:
                cropped_image = crop_image(image, gemini_bounding_box)
                cropped_images.append(cropped_image)
            except Exception as e:
                print(f"Error cropping image: {e}")

    return cropped_images

def generate_content_str(api_key, prompt, pil_image, retries = 2):
    client = genai.Client(api_key=api_key)
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
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
            crop_images = make_crops(
                image=pil_image, predictions_json_txt=response.text
            )
            crop_images_with_labels = []
            for img in crop_images:
                prediction = predict_species(img)
                label = f"{prediction['common_name']} - {prediction['species']} - {round(prediction['score'], 3)}"
                crop_images_with_labels.append((img, label))
            return response.text, crop_images_with_labels
        except errors.ServerError as e:
            retries -= 1
            if retries == 0:
                raise e
            print(f"Retrying... {e}")
            time.sleep(5)

def predict_species(img):
    predictions = classifier.predict([img], Rank.SPECIES, k=1)
    return predictions[0]


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


        with gr.Column():
            gr.Markdown("## Results")
            output = gr.JSON(label="Predictions")
            gr.Markdown("## Cropped Images")
            image_gallery = gr.Gallery(label="Images", show_label=True)

        submit_btn.click(
            fn=generate_content_str,
            inputs=[api_key_input, prompt_input, image_input],
            outputs=[output, image_gallery]
        )

if __name__ == "__main__":
    demo.launch()