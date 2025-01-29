import os
from PIL import Image
from gradio_client import Client


# Set Hugging Face token
os.environ['HF_TOKEN'] = ''

# Create the inference client
client = Client("black-forest-labs/FLUX.1-schnell")

# Set the prompt and parameters for image generation
result = client.predict(
    prompt="A stunning view of Lalbagh Fort in Dhaka, Bangladesh, captured during the late afternoon with soft golden light. The historical fort stands proudly with its intricate Mughal architecture, featuring ornate arches, towering minarets, and a large, serene water pond at the front. Lush green gardens surround the fort, with vibrant flowers in bloom. The fort's reddish stone walls reflect the warm sunlight, and a peaceful atmosphere fills the scene. In the background, the sky is a mix of soft oranges and pinks, signaling the end of the day. Visitors can be seen walking along the pathways, admiring the beauty of this historical site.",
    seed=0,
    randomize_seed=True,
    width=1024,
    height=1024,
    num_inference_steps=4
)

# Open and save the generated image
img = Image.open(result[0])
img.save("generated_image-011.png")

print("Image generated and saved as 'generated_image-011.png'")