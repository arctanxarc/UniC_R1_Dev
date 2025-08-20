import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='gpt_api/gemini_exp.json'

def evaluate(image_urls):
    prompt="""
Please act as an image comparison evaluator and complete the following tasks:
1.Compare the person/object in the first(generated) image with the person/object in the second(reference) image to determine if they are the same.
2.Provide a score from 1 to 100 based on the similarity and the fidelity of the person/object in the first(generated) image(1 means completely different, 100 means exactly the same; the score should reflect the matching degree between the two).
3.Output only the score without any additional text or explanation.
"""
    vertexai.init(project="mmu-gemini-caption-1-5pro", location="us-central1")
    model = GenerativeModel("gemini-2.5-pro")
    def generate(filenames):
        """
        Generates a description for an image file using the Gemini model.
        """
        contents=[]
        for filename in filenames:
            with open(filename, "rb") as f:
                image_content = f.read()
            image_file = Part.from_data(image_content, mime_type="image/png") 
            contents.append(image_file)
        
        contents.append(prompt)
        responses = model.generate_content(
            contents, generation_config=generation_config, safety_settings=safety_settings,
        )
        return responses.text

    generation_config = { "max_output_tokens": 2048, "temperature": 1e-5, "top_p": 1.0}
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    return generate(image_urls)

if __name__ == "__main__":
    # --- Configuration for Image Processing ---
    # Replace with the actual path to your image directory
    image_dir = "/home/daigaole/code/ex/dataset/unictokens_data/concept/test/adrien_brody/0.png"
    
    # Example list of image file paths.
    # You can manually list them or dynamically generate the list from a directory:
    # image_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # For demonstration, using a placeholder list. Replace with your actual image paths.
    image_list = [
        "/home/daigaole/code/ex/dataset/unictokens_data/concept/test/adrien_brody/0.png", 
        # Add more image paths here
    ]
    # ------------------------------------------

    for imagename in image_list:
        try:
            res = generate(imagename)
            print(f"--- Describing: {imagename} ---")
            print(res)
            print("\n" + "="*50 + "\n") # Separator for clarity
        except Exception as e:
            print(f"Error processing {imagename}: {e}")
