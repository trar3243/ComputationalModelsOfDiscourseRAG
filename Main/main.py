import sys, os
import torch
from transformers import AutoTokenizer
import json
from pathlib import Path


CMDDROOT = os.environ['CMDDROOT'] # Expected to be a global environment variable. If not set, navigate to root of repository, call "source ./.env"
sys.path.append(CMDDROOT)

from ClassDefinition.Utils import Logger, ArgumentParser # type: ignore
from ClassDefinition.segmenter import Segmenter

required_arguments = []
optional_arguments = {
    "dataPath": f"{CMDDROOT}/Data/"
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
    g_ArgParse.printArguments()

def path_to_text(path):
    with open(path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content 

def main(inputArguments):
    initialize(inputArguments)
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    segmenter = Segmenter(g_ArgParse.get("device"), tokenizer)
    
    papers = ["1803.08375.md","1802.08129.md", "1802.03426.md", "1709.03082.md", "1612.04662.md", "2405.21046.md", "2405.21040.md", "2405.21018.md", "2405.20974.md", "2405.20774.md"]
    papers = [f"{g_ArgParse.get('dataPath')}/LOONG/data/doc/paper/{paper}" for paper in papers]
    
    financials = ["2022-f10k2021_boxscorebrands.txt","2022-avni123121form10k.txt","2022-aqb-20211231x10k.txt","2021-form10-k.txt","2021-f10k2020_boxscorebrands.txt","2021-avni123120form10k.txt","2020-form10-k.txt","2020-f10k2019_boxscorebrands.txt","2020-avni123119form10k.txt","2019-avni123118form10k.txt"]
    financials = [f"{g_ArgParse.get('dataPath')}/LOONG/data/doc/financial/{paper}" for paper in financials]

    source_list = {"paper": papers, "financial": financials}
    
    method_list = ["most_recent", "most_recent_low_accessible"]
    search_window_list = {
        "conservative": {"target_size_tokens": 384, "search_window": 128}, 
        "liberal": {"target_size_tokens": 258, "search_window": 254}
    }
    full_sentence_inclusion_list = [True, False]
    weighted_list = [True, False]

    for domain, paths in source_list.items():
        for path in paths:
            text = path_to_text(path)
            source_file_name = Path(path).name 
            
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




if __name__ == "__main__":
    try:
        main(sys.argv[1:]) # don't pass the script name to the argument parser 
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
