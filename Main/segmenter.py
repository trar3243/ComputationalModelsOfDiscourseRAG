import json
from ClassDefinition.Utils import Logger# type: ignore
import torch 
import os 
from fastcoref import FCoref
from fastcoref.modeling import FCorefModel
FCorefModel.all_tied_weights_keys = property(lambda self: {}) # fix transformers 5.0 compat



def initialize_coref(device):
    print("Initializing FCoref...")
    model = FCoref(model_name_or_path='biu-nlp/f-coref', device=device)
    print("Completed Initialization of FCoref")
    return model 

def get_text(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)
    return data


def segment(data_path, device):
    model = initialize_coref(device)
    text_data = get_text(data_path)
    target_size = 1000  #Target characters per chunk
    search_window = 750 # How far from target_size we can look for a "clean" break. Can modulate this 

    for doc in text_data:
        full_text = (doc.get('abstract') or "") + "\n" + (doc.get('text') or "")
        if not full_text.strip(): continue

        # model predictions Clusters
        preds = model.predict(texts=[full_text])
        clusters = preds[0].get_clusters(as_strings=False)

        # This tells us exactly how many clusters are "open" at any char index
        deltas = [0] * (len(full_text) + 1)
        for cluster in clusters:
            # We treat the whole cluster footprint (min to max) as a single "block"
            c_start = min(span[0] for span in cluster)
            c_end = max(span[1] for span in cluster)
            deltas[c_start] += 1
            deltas[c_end] -= 1

        # Calculate overlap count per character
        overlap_counts = []
        current_overlap = 0
        for d in deltas[:-1]:
            current_overlap += d
            overlap_counts.append(current_overlap)

        # find best cut 
        chunks = []
        start_idx = 0
        
        while start_idx < len(full_text):
            end_search = start_idx + target_size
            if end_search >= len(full_text):
                chunks.append(full_text[start_idx:])
                break

            window_start = max(start_idx + 1, end_search - search_window)
            window_end = min(len(full_text) - 1, end_search + search_window)
            
            # Find the index with the lowest overlap in this window
            best_cut_idx = window_start
            min_overlap = float('inf')
            
            for i in range(window_start, window_end):
                if overlap_counts[i] < min_overlap:
                    min_overlap = overlap_counts[i]
                    best_cut_idx = i
                elif overlap_counts[i] == min_overlap:
                    # If ties, pick the one closer to target_size
                    if abs(i - end_search) < abs(best_cut_idx - end_search):
                        best_cut_idx = i

            chunks.append(full_text[start_idx:best_cut_idx])
            print(f"Created chunk: {start_idx} to {best_cut_idx} (Intersects {min_overlap} clusters)")
            start_idx = best_cut_idx
            
        file_name = f"doc_x_segmented.txt"
        # file_path = os.path.join(output_dir, file_name)
        
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("=== ORIGINAL TEXT ===\n")
            f.write(full_text)
            f.write("\n\n=== SEGMENTED TEXT (BROKEN BY LINEBREAKS) ===\n")
            f.write("\n=======\n".join(chunks))
        
        break # early break for now 

    return chunks