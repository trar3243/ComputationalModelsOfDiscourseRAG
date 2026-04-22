import sys, os
import gc
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import json
from pathlib import Path


CMDDROOT = os.environ['CMDDROOT'] # Expected to be a global environment variable. If not set, navigate to root of repository, call "source ./.env"
sys.path.append(CMDDROOT)

from ClassDefinition.Utils import Logger, ArgumentParser # type: ignore
from ClassDefinition.segmenter import Segmenter

required_arguments = []
optional_arguments = {
    "dataPath": f"{CMDDROOT}/Data/",
    "verbose": "False"
}
g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

USAGE = """
main.py
    Required:
        
    Optional:
        dataPath=<dataPath>
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE) # add usage note 
        raise
    g_ArgParse.set("device", "cuda" if torch.cuda.is_available() else "cpu")
    g_ArgParse.set("verbose", True if g_ArgParse.get('verbose')=="True" else False) # string to bool 
    g_ArgParse.printArguments()

def path_to_text(path):
    with open(path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content 


def LOONG(segmenter):
    papers = ["1803.08375.md","1802.08129.md", "1802.03426.md", "1709.03082.md", "1612.04662.md", "2405.21046.md", "2405.21040.md", "2405.21018.md", "2405.20974.md", "2405.20774.md"]
    papers = [f"{g_ArgParse.get('dataPath')}/LOONG/data/doc/paper/{paper}" for paper in papers]
    
    financials = ["2022-f10k2021_boxscorebrands.txt","2022-avni123121form10k.txt","2022-aqb-20211231x10k.txt","2021-form10-k.txt","2021-f10k2020_boxscorebrands.txt","2021-avni123120form10k.txt","2020-form10-k.txt","2020-f10k2019_boxscorebrands.txt","2020-avni123119form10k.txt","2019-avni123118form10k.txt"]
    financials = [f"{g_ArgParse.get('dataPath')}/LOONG/data/doc/financial/{paper}" for paper in financials]

    source_list = {"paper": papers, "financial": financials}
    
    method_list = ["nonlinear"]
    search_window_list = {
        "conservative_258": {"target_size_tokens": 258, "search_window": 128}, # target was 384 
        "liberal": {"target_size_tokens": 258, "search_window": 254}
    }
    full_sentence_inclusion_list = [True, False]
    weighted_list = [True, False]

    for domain, paths in source_list.items():
        for path in paths:
            text = path_to_text(path)
            source_file_name = Path(path).name 
            clusters = segmenter.get_clusters(text)
            
            if(g_ArgParse.get('verbose')):
                print(f"Found {len(clusters)} global clusters:")
                for i, cluster in enumerate(clusters): 
                    # Slice the original text using the start and end character indices
                    words = [text[start:end].replace('\n', ' ') for start, end in cluster]
                    
                    # Print the list of strings found in this cluster
                    print(f"Cluster {i}: {words}")
                print("--------------------------------------------------\n")
            
            for method in method_list:
                for window_name, window_params in search_window_list.items():
                    target_size = window_params["target_size_tokens"]
                    search_win = window_params["search_window"]
                    
                    for is_full_sentence in full_sentence_inclusion_list:
                        for is_weighted in weighted_list: 
                            
                            print(f"Processing: {source_file_name} | {method} | {window_name} | Sent:{is_full_sentence} | Weight:{is_weighted}")
                            
                            # run it 
                            chunks = segmenter.coreference_chunk_text(
                                text=text,
                                clusters=clusters,
                                method=method,
                                target_size_tokens=target_size,
                                search_window=search_win,
                                full_sentence_inclusion=is_full_sentence, 
                                weighted=is_weighted
                            )
                            
                            sent_folder = "full_sentence_included" if is_full_sentence else "full_sentence_excluded"
                            weight_folder = "weighted" if is_weighted else "unweighted"
                            
                            # Build chunks/<paper|financial>/<source_file_name>/<method>/<full_sentence_included|excluded>/<weighted|unweighted>/<conservative|liberal>
                            output_dir = Path("chunks") / domain / source_file_name / method / sent_folder / weight_folder / window_name
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            # json
                            output_data = {
                                "metadata": {
                                    "domain": domain,
                                    "source_file": source_file_name,
                                    "method": method,
                                    "sentence_inclusion": is_full_sentence,
                                    "weighted": is_weighted,
                                    "strategy": window_name,
                                    "target_size_tokens": target_size,
                                    "search_window": search_win,
                                    "total_chunks_generated": len(chunks)
                                },
                                "chunks": chunks
                            }
                            
                            output_file = output_dir / "chunks_output.json"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(output_data, f, indent=4)
            del clusters
            del text
            gc.collect()
            torch.cuda.empty_cache()

def OntoNotes(segmenter):
    print("Downloading/Loading OntoNotes (CoNLL-2012) validation split...")
    ontonotes = load_dataset("ramybaly/conll2012", split="validation", trust_remote_code=True)

    # group documents by genre 
    docs_by_genre = {}
    for doc in ontonotes:
        genre = doc['document_id'].split('/')[0]
        if genre not in docs_by_genre:
            docs_by_genre[genre] = []
        docs_by_genre[genre].append(doc)

    selected_samples = []
    genres = list(docs_by_genre.keys())
    idx = 0
    while len(selected_samples) < 10: # 10 of them 
        genre = genres[idx % len(genres)]
        if docs_by_genre[genre]:
            selected_samples.append(docs_by_genre[genre].pop(0))
        idx += 1

    print(f"Selected 10 documents perfectly spread across {len(genres)} genres.")

    # Setup Hyperparameters
    method_list = ["most_recent", "most_recent_low_accessible"]
    search_window_list = {
        "conservative_median_258": {"target_size_tokens": 258, "search_window": 128},
        "liberal": {"target_size_tokens": 258, "search_window": 254}
    }
    full_sentence_inclusion_list = [True, False]
    weighted_list = [True, False]

    for sample in selected_samples:
        doc_id = sample['document_id']
        domain = doc_id.split('/')[0] # Use the genre (e.g., 'nw') as the domain 
        
        # Flatten the list of lists of words into a single text string
        words = [word for sentence in sample['sentences'] for word in sentence['words']]
        text = " ".join(words)
        
        source_file_name = doc_id.replace('/', '_') + ".txt"

        clusters = segmenter.get_clusters(text)

        print(clusters)
        
        if(g_ArgParse.get('verbose')):
            print(f"Found {len(clusters)} global clusters for {source_file_name}:")
            for i, cluster in enumerate(clusters): 
                # Slice the original text using the start and end character indices
                cluster_words = [text[start:end].replace('\n', ' ') for start, end in cluster]
                print(f"Cluster {i}: {cluster_words}")
            print("--------------------------------------------------\n")

        for method in method_list:
            for window_name, window_params in search_window_list.items():
                target_size = window_params["target_size_tokens"]
                search_win = window_params["search_window"]
                
                for is_full_sentence in full_sentence_inclusion_list:
                    for is_weighted in weighted_list: 
                        
                        print(f"Processing: {source_file_name} | {method} | {window_name} | Sent:{is_full_sentence} | Weight:{is_weighted}")
                        
                        # run it 
                        chunks = segmenter.coreference_chunk_text(
                            text=text,
                            clusters=clusters,
                            method=method,
                            target_size_tokens=target_size,
                            search_window=search_win,
                            full_sentence_inclusion=is_full_sentence, 
                            weighted=is_weighted
                        )
                        
                        sent_folder = "full_sentence_included" if is_full_sentence else "full_sentence_excluded"
                        weight_folder = "weighted" if is_weighted else "unweighted"
                        
                        # Build output directory: chunks/ontonotes/<genre>/<filename>/...
                        output_dir = Path("chunks") / "ontonotes" / domain / source_file_name / method / sent_folder / weight_folder / window_name
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # json construction
                        output_data = {
                            "metadata": {
                                "dataset": "ontonotes_conll2012",
                                "document_id": doc_id, # unique to OntoNotes 
                                "domain": domain,                     
                                "source_file": source_file_name,
                                "method": method,
                                "sentence_inclusion": is_full_sentence,
                                "weighted": is_weighted,
                                "strategy": window_name,
                                "target_size_tokens": target_size,
                                "search_window": search_win,
                                "total_chunks_generated": len(chunks)
                            },
                            "chunks": chunks
                        }
                        
                        output_file = output_dir / "chunks_output.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, indent=4)
                            
        # Memory cleanup between heavy documents
        del clusters
        del text
        gc.collect()
        torch.cuda.empty_cache()
def LOONG_generate_all_chunks(segmenter):
    root_path = Path(CMDDROOT)

    base_dir = root_path / "documents/loong_docs"
    source_list = []
    source_list.extend(list(base_dir.glob("financial/*.txt")))
    source_list.extend(list(base_dir.glob("paper/*.md")))

    total_docs = len(source_list)
    if total_docs == 0:
        print("No documents found.")
        return

    method = "nonlinear"
    target_size_tokens = 258
    search_window = 254
    full_sentence_included = False
    weighted = False 


    folder_name = f"method={method}_target_size_tokens={target_size_tokens}_search_window={search_window}_full_sent={full_sentence_included}_weighted={weighted}"

    output_dir = root_path / "chunks" / folder_name

    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_count = 0

    # Use enumerate to track the current position (starting at 1 for percentage calculation)
    for i, path in enumerate(source_list, start=1):
        # Calculate percentage
        percent_complete = (i / total_docs) * 100
        
        # Define the output path first
        json_file_path = output_dir / f"{path.stem}.json"

        # Check if the output file already exists and skip if it does
        if json_file_path.exists():
            print(f"[{percent_complete:.2f}%] Skipping {path.name}: {json_file_path.name} already exists.")
            skipped_count += 1
            continue
            
        print(f"[{percent_complete:.2f}%] Processing {path.name}...")

        text = path_to_text(path)
        clusters = segmenter.get_clusters(text)

        chunks = segmenter.coreference_chunk_text(
            text=text,
            clusters=clusters,
            method=method,
            target_size_tokens=target_size_tokens,
            search_window=search_window,
            full_sentence_inclusion=full_sentence_included,
            weighted=weighted
        )

        output_data = {
            "source_file_name": path.name,
            "chunks": chunks
        }

        # Write to JSON
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        del clusters
        del text
        gc.collect()
        torch.cuda.empty_cache()

    processed_count = total_docs - skipped_count
    print(f"Successfully processed {processed_count} files (skipped {skipped_count}) into: {output_dir}")

def SQuAD_generate_all_chunks(segmenter):
    root_path = Path(CMDDROOT)
    
    # Target your specific parquet file
    parquet_file = root_path / "squad_testing/data/squad_wiki_full_articles.parquet"
    
    if not parquet_file.exists():
        print(f"Error: Could not find dataset at {parquet_file}")
        return

    print(f"Loading SQuAD parquet file from {parquet_file}...")
    # Leveraging the already-imported Hugging Face datasets library
    dataset = load_dataset("parquet", data_files={"train": str(parquet_file)}, split="train")
    total_docs = len(dataset)

    if total_docs == 0:
        print("No documents found in the parquet file.")
        return

    method = "nonlinear"
    target_size_tokens = 258
    search_window = 254
    full_sentence_included = False
    weighted = False

    folder_name = f"method={method}_target_size_tokens={target_size_tokens}_search_window={search_window}_full_sent={full_sentence_included}_weighted={weighted}"
    
    # Store in a dedicated 'squad' directory
    output_dir = root_path / "chunks" / "squad" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_count = 0

    for i, row in enumerate(dataset, start=1):
        percent_complete = (i / total_docs) * 100
        
        # Extract title and text. Fallback to generic names/fields if standard SQuAD keys are missing
        title = row.get('title', f"squad_doc_{i}")
        text = row.get('text', row.get('context', ''))
        
        if not text:
            print(f"[{percent_complete:.2f}%] Skipping {title}: No text/context found.")
            skipped_count += 1
            continue

        # Sanitize title to make it a safe filename (removes spaces, slashes, etc.)
        safe_title = "".join([c if c.isalnum() else "_" for c in title])
        json_file_path = output_dir / f"{safe_title}.json"

        # Check for existing file to allow resuming interrupted runs
        if json_file_path.exists():
            print(f"[{percent_complete:.2f}%] Skipping {title}: {json_file_path.name} already exists.")
            skipped_count += 1
            continue

        print(f"[{percent_complete:.2f}%] Processing {title}...")

        # Run your chunking strategy
        clusters = segmenter.get_clusters(text)
        chunks = segmenter.coreference_chunk_text(
            text=text,
            clusters=clusters,
            method=method,
            target_size_tokens=target_size_tokens,
            search_window=search_window,
            full_sentence_inclusion=full_sentence_included,
            weighted=weighted
        )

        # JSON construction
        output_data = {
            "source_file_name": safe_title,
            "original_title": title,
            "chunks": chunks
        }

        # Write out
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        # Memory cleanup
        del clusters
        del chunks
        del text
        gc.collect()
        torch.cuda.empty_cache()

    processed_count = total_docs - skipped_count
    print(f"Successfully processed {processed_count} files (skipped {skipped_count}) into: {output_dir}")

def main(inputArguments):
    initialize(inputArguments)
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    segmenter = Segmenter(g_ArgParse.get("device"), tokenizer)
    
    # LOONG(segmenter)
    # LOONG_generate_all_chunks(segmenter)
    SQuAD_generate_all_chunks(segmenter)

    

    
    




if __name__ == "__main__":
    try:
        main(sys.argv[1:]) # don't pass the script name to the argument parser 
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
