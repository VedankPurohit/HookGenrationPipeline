import google.generativeai as genai
import os
import json
import random
from dotenv import load_dotenv
from MoreFeatures.LLM import prompts

# Load environment variables from a .env file
load_dotenv()

template_map = {
    "best": prompts.BestClip,  # Default - full creative freedom
    "rapidfire": prompts.RapidFireQna,
    "general": prompts.GeneralClipGenerator,
    "emotional": prompts.EmotionalHighlight,
    "keytakeaway": prompts.KeyTakeaway,
    "controversial": prompts.ControversialMoment,
    "shorts": prompts.ShortsTemplate,
}

random_templates = [
    prompts.RapidFireQna,
    prompts.EmotionalHighlight,
    prompts.KeyTakeaway,
    prompts.ControversialMoment,
]

def generate_clips(
    input_filepath: str,
    output_filepath: str,
    template_name: str,
    custom_instructions: str = None,
    model_name: str = "gemini-3-flash-preview",
):
    """
    Generates video clips based on a specified template and optional custom instructions.

    Args:
        input_filepath (str): The path to the input text file (transcript).
        output_filepath (str): The path to save the output JSON file.
        template_name (str): The name of the template to use (e.g., "rapidfire", "general", "emotional").
        custom_instructions (str, optional): Custom instructions for the "general" template.
                                             If "general" is selected and this is None, a random
                                             template will be chosen. Defaults to None.
        model_name (str): The name of the Gemini model to use. Defaults to "gemini-2.5-flash".
    """
    selected_template = None

    if template_name == "general":
        if custom_instructions:
            selected_template = prompts.GeneralClipGenerator.format(
                custom_instructions=custom_instructions
            )
        else:
            print("No custom instructions provided for 'general' template. Selecting a random template.")
            selected_template = random.choice(random_templates)
    else:
        selected_template = template_map.get(template_name)
        if not selected_template:
            print(f"Error: Template '{template_name}' not found. Please choose from {list(template_map.keys())}.")
            return

    if selected_template:
        convert_text_to_json(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            sub_template=selected_template,
            model_name=model_name,
        )


def convert_text_to_json(
    input_filepath: str,
    output_filepath: str,
    sub_template: str,
    model_name: str = "gemini-3-flash-preview",
):
    """
    Converts the content of a text file to a JSON file using a specified Gemini model.

    This function now constructs the system prompt using a main template and a sub-template.

    Args:
        input_filepath (str): The path to the input text file.
        output_filepath (str): The path to save the output JSON file.
        sub_template (str): The specific template content to be injected into the main system prompt.
        num_clips (int): The desired number of clips to be generated.
        model_name (str): The name of the Gemini model to use. Defaults to "gemini-2.5-flash".
    """
    # Configure the Gemini API client
    try:
        # It's best practice to load the key and configure the client once.
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
    except (ValueError, KeyError) as e:
        print(f"API Key Error: {e}")
        print("Please set the GOOGLE_API_KEY environment variable. Get your key from https://aistudio.google.com/")
        return

    # Construct the system prompt using the mainSystem template and injected sub_template
    system_prompt = prompts.mainSystem.format(
        sub_template=sub_template
    )
    print(f"system prompt - : {system_prompt}")
    # Instantiate the model
    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)

    # 1. Read the content of the input text file
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            text_content = f.read()
    except FileNotFoundError:
        print(f"Error: The input file '{input_filepath}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the input file: {e}")
        return

    # 2. Set the generation configuration to ensure JSON output
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json"
    )
    print(f"Transcript : {text_content[:200]}")

    # 3. Call the Gemini API
    print(f"Calling {model_name} to generate JSON...")
    try:
        response = model.generate_content(
            contents=text_content,
            generation_config=generation_config,
        )
        # print(f"LLm response : {response}")
        # 4. Check for a valid JSON response and save it
        # The response text should be a valid JSON string
        if response.text:
            try:
                # Clean the response text - strip markdown code blocks if present
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                # The API returns the JSON as a string, so we need to parse it
                json_data = json.loads(response_text)

                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=4)
                print(f"Successfully converted '{input_filepath}' to '{output_filepath}'.")
            except json.JSONDecodeError as e:
                print(f"Error: Gemini did not return a valid JSON string. {e}")
                print("--- Model Output ---")
                print(response.text[:500])
                print("--------------------")
        else:
            print("Gemini returned an empty response.")
            
    except Exception as e:
        print(f"An error occurred during the API call: {e}")

# --- Example Usage ---

if __name__ == "__main__":

    # Use os.path.join for robust file paths
    input_file = os.path.join("inputs", "llm_summary.txt")
    output_file = "output.json"

    # Example usage with the new prompt structure
    # We'll use the RapidFireQna sub-template.
    convert_text_to_json(
        input_file,
        output_file,
        sub_template=prompts.RapidFireQna,
    )

    # Example using GeneralClipGenerator with custom instructions
    print("\n--- Demonstrating GeneralClipGenerator with custom instructions ---")
    custom_instructions = "Extract 3 clips that highlight the speaker's most insightful advice on career development."
    generate_clips(
        input_filepath=input_file,
        output_filepath="general_output.json",
        template_name="general",
        custom_instructions=custom_instructions,
    )

    # Example using GeneralClipGenerator with random template selection
    print("\n--- Demonstrating GeneralClipGenerator with random template selection ---")
    generate_clips(
        input_filepath=input_file,
        output_filepath="random_output.json",
        template_name="general",
    )

    # Example using EmotionalHighlight
    print("\n--- Demonstrating EmotionalHighlight ---")
    generate_clips(
        input_filepath=input_file,
        output_filepath="emotional_output.json",
        template_name="emotional",
    )

    # Example using KeyTakeaway
    print("\n--- Demonstrating KeyTakeaway ---")
    generate_clips(
        input_filepath=input_file,
        output_filepath="key_takeaway_output.json",
        template_name="keytakeaway",
    )

    # Example using ControversialMoment
    print("\n--- Demonstrating ControversialMoment ---")
    generate_clips(
        input_filepath=input_file,
        output_filepath="controversial_output.json",
        template_name="controversial",
    )

    

    # You can now check the output.json file to see the result.
    # Here's how you can read and print the generated JSON to verify.
    print("\n--- Verifying the output JSON file ---")
    try:
        with open(output_file, "r") as f:
            generated_json = json.load(f)
            print(json.dumps(generated_json, indent=4))
    except FileNotFoundError:
        print(f"Error: The output file '{output_file}' was not created.")
    except json.JSONDecodeError:
        print(f"Error: The output file '{output_file}' is not a valid JSON.")
    except Exception as e:
        print(f"An error occurred while verifying the output file: {e}")