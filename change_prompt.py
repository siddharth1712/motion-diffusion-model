from pydantic import BaseModel
import os
import json
from pathlib import Path
import base64
from tqdm import tqdm
import shutil

# class Prompt(BaseModel):
#     full_prompt: str
#     left_prompt: str
#     right_prompt: str

def generate_prompt(client, image_path):
    text_input = open(image_path,'r').read()
    prompt = text_input +'\n' + f"""
    Given this text, convert it into a JSON object in the following format:
    {{
    "[MOTION1]" : <string>
    "[MOTION2]" : <string>
    "[MOTION3]" : <string>
    and so on as needed
    }}
    Describe each motion in a short and concise manner. You can combine or break different [MOTION] tags as required to keep each motion descriptions within 10 words. Also describe every motion as beginning with "The person is"
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that describes text."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }
        ],
        max_tokens=1000,
        temperature=0.0,
        response_format = {"type" : "json_object"}
    )
    
    content = response.choices[0].message.content
    return content

def load_prompt_from_json(gpt_response):
    data = json.loads(gpt_response)

    final_prompt = ""
    for i,key in enumerate(data):
        if i == 0:
            final_prompt = final_prompt + key + " " + data[key][:-1] + ";"
        elif not i==len(data) - 1:
            final_prompt = final_prompt + " " + key + " " + data[key][:-1] + ";"
        else:
            final_prompt = final_prompt + " " + key + " " + data[key]
    
    return final_prompt
    
def process_directory(client, input_dir, output_dir):
    """
    Process all text files in input directory and save to new directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get list of image files first
    text_files = [f for f in input_path.iterdir()]
    
    # Process each image in directory with progress bar
    for text_file in tqdm(text_files, desc="Processing text prompts"):
        try:
            full_image_path = input_path / text_file.name
            gpt_response = generate_prompt(client, str(full_image_path))
            # print(gpt_response)
            output_full_path = output_path / text_file.name
            
            final_prompt = load_prompt_from_json(gpt_response)
            f = open(output_full_path, "a")
            f.write(final_prompt)
            f.close()
            
        except Exception as e:
            print(f"Error processing {text_file.name}: {str(e)}")
            print("Continuing with next image...")
            continue

# Example usage:
if __name__ == "__main__":
    from openai import OpenAI
<<<<<<< HEAD
    #client = OpenAI(api_key="<dummy>")
=======
    client = OpenAI(api_key="<dummy>")
>>>>>>> 90686cf... Removed api key
    
    # input_directory = "concat_images"
    # output_json_path = "prompts_output_SS_Panels.json"
    
    input_directory="./dataset/HumanML3D/babel_texts"
    final_prompt_folder ="./dataset/HumanML3D/babel_texts_modified"
    
    if os.path.exists(final_prompt_folder):
        shutil.rmtree(final_prompt_folder)
    os.mkdir(final_prompt_folder)
        
    process_directory(client, input_directory, final_prompt_folder)