import sys, os 
import time 
import json 
from pathlib import Path 
from google.genai import errors
import chromadb
import google.genai as genai
from sentence_transformers import SentenceTransformer
CMDDROOT = os.environ['CMDDROOT'] # Expected to be a global environment variable. If not set, navigate to root of repository, call "source ./.env"
sys.path.append(CMDDROOT)
from config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
)
print(GEMINI_API_KEY)
from index_documents import LocalEmbeddingFunction
from query_documents import ask, generate_answer, retrieve, load_collection   

client_genai = genai.Client(api_key=GEMINI_API_KEY)

prompt_input_path = f"{CMDDROOT}/loong.jsonl" 

def read_prompt_input_path():
    data = []
    try:
        with open(prompt_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data.append(json.loads(line))

    except FileNotFoundError:
        # Fixed variable name from file_path to prompt_input_path
        print(f"Error: The file {prompt_input_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from {prompt_input_path}. {e}")

    return data

def retrieve_filtered(
    collection: chromadb.Collection,
    query: str,
    n_results: int,
    target_files: list 
) -> dict:
    """Retrieve the most relevant chunks for a query."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"source": {"$in": target_files}}
    )
    return results


def main():
    collection = load_collection()
    prompts = read_prompt_input_path()
    
    output_path = "evaluated_loong.jsonl"
    processed_ids = set()

    # Read the output file if it exists and collect all finished IDs
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): # Skip empty lines
                    try:
                        processed_obj = json.loads(line)
                        if "id" in processed_obj:
                            processed_ids.add(processed_obj["id"])
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_ids)} already processed items. Resuming...")

    total_en_prompts = sum(1 for p in prompts if p.get("language") == "en")
    counter = 0

    print(f"\nAppending new results to {output_path}...")
    
    with open(output_path, 'a', encoding='utf-8') as f:
        for prompt_data in prompts:
            if prompt_data.get("language") != "en":
                continue
                
            # check if this ID is already done
            prompt_id = prompt_data.get("id")
            if prompt_id in processed_ids:
                counter += 1 
                continue
                
            target_files = prompt_data["doc"]
            question = prompt_data["question"]

            # Query ChromaDB to get the actual text
            retrieved_data = retrieve_filtered(
                collection=collection,
                query=question,
                n_results=10,
                target_files=target_files # Using the filter
            )

            # Combine retrieved chunks
            retrieved_text = ""
            for chunk in retrieved_data["documents"][0]:
                retrieved_text += chunk + "\n\n"

            # prompt template expects these data
            prompt_data["docs"] = retrieved_text

            # format the template
            template_str = prompt_data["prompt_template"]
            filled_prompt = template_str.format(**prompt_data)

            # Call the LLM
            # ---> ROBUST RETRY LOGIC <---
            while True:
                try:
                    # Call the LLM
                    response = client_genai.models.generate_content(
                        model=GENERATION_MODEL,
                        contents=filled_prompt,
                    )
                    prompt_data["answer"] = response.text
                    break # Success! Break out of the retry loop
                    
                # Catch BOTH ClientErrors (429 Rate Limits) and ServerErrors (503 Overloads)
                except (errors.ClientError, errors.ServerError) as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        print("\n[Rate Limit Hit] Sleeping for 35 seconds...")
                        time.sleep(35)
                    elif "503" in error_msg or "UNAVAILABLE" in error_msg:
                        print("\n[Server Overloaded] Google API is busy. Sleeping for 60 seconds...")
                        time.sleep(60)
                    else:
                        # If it's a completely different error, crash normally so you can debug
                        raise e    
            
            prompt_data["answer"] = response.text
            prompt_data.pop("docs", None) # dont write our retrieved chunks to the file 

            # Write the completed object to the file 
            json_string = json.dumps(prompt_data, ensure_ascii=False)
            f.write(json_string + '\n')
            f.flush() # Forces Python to write to the hard drive right now (crash protection)
            
            # Add to our tracked set so we know it's done
            processed_ids.add(prompt_id)

            counter += 1
            print(f"=== Completed {counter}/{total_en_prompts} = {(counter/total_en_prompts)*100:.2f}% ===")

if __name__ == "__main__":
    main()
