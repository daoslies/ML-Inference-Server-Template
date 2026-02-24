import yaml
import ast

def load_prompts(yaml_path: str):
    """Load prompts from a YAML file"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def extract_names(text: str, llm, prompts_path: str):
    """Extract person names from text using LLM"""
    prompts = load_prompts(prompts_path)
    prompt = prompts['extract_names'].format(text=text)
    response = llm(
        prompt,
        max_tokens=256,
        temperature=0.3,
        stop=["</s>", "\n\n"],
    )

    llm.reset()  # Required for some models

    if isinstance(response, dict) and 'choices' in response:
        response_text = response['choices'][0]['text'].strip()
    elif isinstance(response, str):
        response_text = response.strip()
    else:
        response_text = str(response).strip()
    try:
        start = response_text.find('[')
        end = response_text.rfind(']')
        if start != -1 and end != -1 and end > start:
            list_str = response_text[start:end+1]
            try:
                names = ast.literal_eval(list_str)
            except Exception:
                names = list_str  # fallback: return as string
        else:
            names = response_text  # fallback: return as string
        return names
    except Exception:
        return []