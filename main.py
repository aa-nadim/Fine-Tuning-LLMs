import os
from PIL import Image
from gradio_client import Client


# Set Hugging Face token
os.environ['HF_TOKEN'] = ''

# Create the inference client
client = Client("black-forest-labs/FLUX.1-schnell")

# Set the prompt and parameters for image generation
result = client.predict(
    prompt="A serene lake surrounded by lush green mountains and a clear blue sky. The water is calm and still, reflecting the vibrant blue sky and the surrounding mountains. The scene is peaceful and tranquil, with a sense of harmony and balance. The lake is surrounded by a dense forest, with tall trees and wildflowers adding to the natural beauty of the scene. The sun is shining brightly, casting a warm glow over the entire landscape.",
    seed=0,
    randomize_seed=True,
    width=1024,
    height=1024,
    num_inference_steps=4
)

# Open and save the generated image
img = Image.open(result[0])
img.save("generated_image_004.png")

print("Image generated and saved as 'generated_image_004.png'")