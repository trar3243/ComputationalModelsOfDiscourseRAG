import json
from ClassDefinition.Utils import Logger# type: ignore
import torch 
import os 
import re
from fastcoref import FCoref
from fastcoref.modeling import FCorefModel
FCorefModel.all_tied_weights_keys = property(lambda self: {}) # fix transformers 5.0 compat

# small_example = "Reggie is my cat. He is a good boy. He likes to play. Reggie is very loud. He meows all the time. He never ever stops. He is super noisy. He demands his food."

class Segmenter:
    def __init__(self, device, tokenizer):
        self.device = device
        self.tokenizer=tokenizer
        self.model = self.__initialize_coref__(self.device)


    def __initialize_coref__(self,device):
        print("Initializing FCoref...")
        model = FCoref(model_name_or_path='biu-nlp/f-coref', device=device)
        print("Completed Initialization of FCoref")
        return model 

    def __get_accessibility_score__(self,text): # ref: Croft notes on accessability 
        t = text.lower().strip()
        words = t.split()

        # Anaphoric Pronouns (Highest for English spans ignoring the null anaphora since our tokenizer isnt gonna pick it up 
        if t in {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their"}:
            return 8.0

        # First/Second Person 
        if t in {"i", "me", "my", "you", "your", "we", "us"}:
            return 7.0

        # Demonstrative Pronouns ("This")
        if t in {"this", "that", "these", "those"}:
            return 6.0

        # 4. Demonstrative Attributive ("This king")
        if words[0] in {"this", "that", "these", "those"} and len(words) > 1:
            return 4.0

        # 5. Definite NP ("The monster")
        if words[0] == "the":
            return 3.0

        # 6. Proper Nouns / Names (Lowest accessibility)
        return 1.0
    def get_text(self,data_path):
        with open(data_path, 'r') as file:
            data = json.load(file)
        return data
    def __char_to_token__(self,char_idx, offsets):
        for i, (start, end) in enumerate(offsets):
            if start <= char_idx < end:
                return i
            # If the char is whitespace/unmapped before this token
            if char_idx < start:
                return max(0, i - 1)
        return len(offsets) - 1
    
    ## IDEA 1: Treat each cluster as a monolith. This means that there is one edge per cluster, which goes from the first entity mention to the last entity mention. The weights are the maximum weight of the edges across the cluster 
    def segment_monolithic_cluster(self,data_path, target_size_tokens=384, search_window=128, test=True):
        model = self.model
        text_data = self.get_text(data_path)
        tokenizer = self.tokenizer
        device = self.device 
        # target_size_tokens: Target tokens per chunk (mean of 256, 512)
        # search_window : How far from target_size we can look for a "clean" break. Minimum = 256 = 384 - 128. Max = 512 = 384 + 128 

        
        k = 0 # track number of docs in text data 
        for doc in text_data:
            full_text = (doc.get('abstract') or "") + "\n" + (doc.get('text') or "")
            if not full_text.strip(): continue
            # full_text = small_example #TEST 
            # add_special_tokens=False prevents [CLS]/[SEP] from messing with overlaps
            encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoded['input_ids']
            offsets = encoded['offset_mapping']
            num_tokens = len(input_ids)

            sent_char_spans = [m.span() for m in re.finditer(r'[^.!?\n]+[.!?\n]*', full_text)] # fancy regex for sentence boundaries 
            
            token_to_sent = {}
            for s_start, s_end in sent_char_spans:
                t_start = self.__char_to_token__(s_start, offsets)
                t_end = self.__char_to_token__(s_end - 1, offsets) 
                
                for i in range(t_start, t_end + 1):
                    token_to_sent[i] = (t_start, t_end)

            # predict from the F-Coref model 
            preds = model.predict(texts=[full_text]) 
            print(preds)
            clusters = preds[0].get_clusters(as_strings=False) # returns CHARACTER, not token, indices 

            # track clusters using weighted token indices 
            deltas = [0.0] * (num_tokens + 1) 
            
            for cluster in clusters:
                # Determine the weight of this cluster based on the most accessible mention it contains
                # If a cluster has a "he", the entire edge is weighted higher
                weight = max(self.__get_accessibility_score__(full_text[s:e]) for s, e in cluster)

                c_start = min(span[0] for span in cluster) 
                c_end = max(span[1] for span in cluster) 

                # Map the character span to a token span
                t_start = self.__char_to_token__(c_start, offsets)
                t_end = self.__char_to_token__(c_end, offsets)

                deltas[t_start] += weight 
                if t_end + 1 <= num_tokens: 
                    deltas[t_end + 1] -= weight

            # then apply the boundaries 
            overlap_counts = []
            current_overlap = 0.0
            for d in deltas[:-1]:
                current_overlap += d
                overlap_counts.append(current_overlap)

            chunks_text = []
            start_idx = 0
            
            while start_idx < num_tokens:
                end_search = start_idx + target_size_tokens
                if end_search >= num_tokens:
                    # Reconstruct text from the remaining tokens
                    expanded_start = token_to_sent.get(start_idx, (start_idx, start_idx))[0]
                    chunk_ids = input_ids[expanded_start:]
                    chunks_text.append(tokenizer.decode(chunk_ids))
                    break

                window_start = max(start_idx + 1, end_search - search_window)
                window_end = min(num_tokens - 1, end_search + search_window)
                
                # Find the token index with the lowest overlap in this window
                best_cut_idx = window_start
                min_overlap = float('inf')
                
                for i in range(window_start, window_end):
                    if overlap_counts[i] < min_overlap:
                        min_overlap = overlap_counts[i]
                        best_cut_idx = i
                    elif overlap_counts[i] == min_overlap:
                        # If ties, pick the one closer to target_size_tokens
                        if abs(i - end_search) < abs(best_cut_idx - end_search):
                            best_cut_idx = i

                # Slice the tokens and decode back to text
                expanded_start = token_to_sent.get(start_idx, (start_idx, start_idx))[0]
                last_token = best_cut_idx - 1
                expanded_end = token_to_sent.get(last_token, (last_token, last_token))[1] + 1
                
                chunk_ids = input_ids[expanded_start:expanded_end]# from start to best cut 

                chunks_text.append(tokenizer.decode(chunk_ids))
                
                print(f"Created chunk: {start_idx} to {best_cut_idx} (Weighted Overlap: {min_overlap})")
                start_idx = best_cut_idx
                
            file_name = f"doc_{k}_segmented_monolithic.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("=== ORIGINAL TEXT ===\n")
                f.write(full_text)
                f.write("\n\n=== SEGMENTED TEXT (BROKEN BY LINEBREAKS) ===\n")
                f.write("\n=======\n".join(chunks_text))
            
            if(test):
                break # early break for testing
            k = k + 1 

        return chunks_text

    ## IDEA 2: each edge is an edge from one entity mention to the most recent entity mention. The weight of the edge is based on the accessability of the second mention in each edge pair 
    def segment_most_recent(self,data_path, target_size_tokens=384, search_window=128, test=True):
        model = self.model
        text_data = self.get_text(data_path)
        device = self.device 
        tokenizer= self.tokenizer

        k = 0 
        for doc in text_data:
            full_text = (doc.get('abstract') or "") + "\n" + (doc.get('text') or "")
            if not full_text.strip(): continue
            # full_text = small_example # test 
            
            encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoded['input_ids']
            offsets = encoded['offset_mapping']
            num_tokens = len(input_ids)

            sent_char_spans = [m.span() for m in re.finditer(r'[^.!?\n]+[.!?\n]*', full_text)] 
            
            token_to_sent = {}
            for s_start, s_end in sent_char_spans:
                t_start = self.__char_to_token__(s_start, offsets)
                t_end = self.__char_to_token__(s_end - 1, offsets) 
                for i in range(t_start, t_end + 1):
                    token_to_sent[i] = (t_start, t_end)

            # Predict from the F-Coref model 
            preds = model.predict(texts=[full_text]) 
            print(preds) 
            clusters = preds[0].get_clusters(as_strings=False) 

            # Track tokens 
            deltas = [0.0] * (num_tokens + 1) 
            
            for cluster in clusters:
                # Sort mentions chronologically to create sequential edges
                sorted_mentions = sorted(cluster, key=lambda x: x[0]) # sort by start 

                for i in range(len(sorted_mentions)): # for each mention 
                    m_curr = sorted_mentions[i]
                    m_curr_text = full_text[m_curr[0]:m_curr[1]]
                    m_curr_score = self.__get_accessibility_score__(m_curr_text)
                    
                    # Weight the mention itself
                    t_start = self.__char_to_token__(m_curr[0], offsets)
                    t_end = self.__char_to_token__(m_curr[1] - 1, offsets)
                    
                    deltas[t_start] += m_curr_score
                    if t_end + 1 <= num_tokens:
                        deltas[t_end + 1] -= m_curr_score

                    #Weight the EDGE (gap between this and prior mention)
                    # Weighted according to the accessibility of the second mention (m_curr)
                    if i > 0:
                        m_prev = sorted_mentions[i-1]
                        prev_t_end = self.__char_to_token__(m_prev[1] - 1, offsets)
                        
                        gap_start = prev_t_end + 1
                        gap_end = t_start # Up to the start of current mention
                        
                        if gap_end > gap_start:
                            deltas[gap_start] += m_curr_score
                            if gap_end <= num_tokens:
                                deltas[gap_end] -= m_curr_score

            # Then apply the boundaries 
            overlap_counts = []
            current_overlap = 0.0
            for d in deltas[:-1]:
                current_overlap += d
                overlap_counts.append(current_overlap)

            chunks_text = []
            start_idx = 0
            
            while start_idx < num_tokens:
                end_search = start_idx + target_size_tokens
                if end_search >= num_tokens:
                    expanded_start = token_to_sent.get(start_idx, (start_idx, start_idx))[0]
                    chunk_ids = input_ids[expanded_start:]
                    chunks_text.append(tokenizer.decode(chunk_ids))
                    break

                window_start = max(start_idx + 1, end_search - search_window)
                window_end = min(num_tokens - 1, end_search + search_window)
                
                best_cut_idx = window_start
                min_overlap = float('inf')
                
                for i in range(window_start, window_end):
                    if overlap_counts[i] < min_overlap:
                        min_overlap = overlap_counts[i]
                        best_cut_idx = i
                    elif overlap_counts[i] == min_overlap:
                        if abs(i - end_search) < abs(best_cut_idx - end_search):
                            best_cut_idx = i

                expanded_start = token_to_sent.get(start_idx, (start_idx, start_idx))[0]
                last_token = best_cut_idx - 1
                expanded_end = token_to_sent.get(last_token, (last_token, last_token))[1] + 1
                
                chunk_ids = input_ids[expanded_start:expanded_end]
                chunks_text.append(tokenizer.decode(chunk_ids))
                

                print(f"Created chunk: {start_idx} to {best_cut_idx} (Weighted Overlap {min_overlap})")
                start_idx = best_cut_idx
                
            file_name = f"doc_{k}_segmented_most_recent.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("=== ORIGINAL TEXT ===\n")
                f.write(full_text)
                f.write("\n\n=== SEGMENTED TEXT (BROKEN BY LINEBREAKS) ===\n")
                f.write("\n=======\n".join(chunks_text))
            
            if(test):
                break # early break for testing
            k = k + 1 

        return chunks_text

    ## IDEA 3: as before, weight each edge according to its second pair. However, the antecedent for each edge is the last antecedent which had an antecedent score geq 3 (includes proper nouns, names, definite noun phrase )
    def segment_low_accessible_antecedent(self,data_path, target_size_tokens=384, search_window=128, test=True):
        model = self.model
        device = self.device 
        tokenizer=self.tokenizer
        text_data = self.get_text(data_path)

        k = 0 
        for doc in text_data:
            full_text = (doc.get('abstract') or "") + "\n" + (doc.get('text') or "")
            if not full_text.strip(): continue
            # full_text = small_example
            
            #tokenize text to get input_ids and character offsets mapping
            encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoded['input_ids']
            offsets = encoded['offset_mapping']
            num_tokens = len(input_ids)

            sent_char_spans = [m.span() for m in re.finditer(r'[^.!?\n]+[.!?\n]*', full_text)] 
            
            token_to_sent = {}
            for s_start, s_end in sent_char_spans:
                t_start = self.__char_to_token__(s_start, offsets)
                t_end = self.__char_to_token__(s_end - 1, offsets) 
                for i in range(t_start, t_end + 1):
                    token_to_sent[i] = (t_start, t_end)

            # Predict from the F-Coref model 
            preds = model.predict(texts=[full_text]) 
            print(preds) 
            clusters = preds[0].get_clusters(as_strings=False) 

            # Track clusters using weighted token indices (floats)
            deltas = [0.0] * (num_tokens + 1) 
            
            for cluster in clusters:
                # Sort mentions chronologically to create sequential edges
                sorted_mentions = sorted(cluster, key=lambda x: x[0]) 
                
                # Keep track of the most recent qualifying antecedent 
                last_strong_antecedent = None

                for i in range(len(sorted_mentions)):
                    m_curr = sorted_mentions[i]
                    m_curr_text = full_text[m_curr[0]:m_curr[1]]
                    m_curr_score = self.__get_accessibility_score__(m_curr_text)
                    
                    # Weight
                    t_start = self.__char_to_token__(m_curr[0], offsets)
                    t_end = self.__char_to_token__(m_curr[1] - 1, offsets)
                    
                    deltas[t_start] += m_curr_score
                    if t_end + 1 <= num_tokens:
                        deltas[t_end + 1] -= m_curr_score

                    # Weight the EDGE (gap between this mention and the last strong antecedent)
                    if last_strong_antecedent is not None:
                        prev_t_end = self.__char_to_token__(last_strong_antecedent[1] - 1, offsets)
                        
                        gap_start = prev_t_end + 1
                        gap_end = t_start # Up to the start of current mention
                        
                        if gap_end > gap_start:
                            deltas[gap_start] += m_curr_score
                            if gap_end <= num_tokens:
                                deltas[gap_end] -= m_curr_score


                    if m_curr_score <= 3.0: # special part 
                        last_strong_antecedent = m_curr

            # Then apply the boundaries 
            overlap_counts = []
            current_overlap = 0.0
            for d in deltas[:-1]:
                current_overlap += d
                overlap_counts.append(current_overlap)

            chunks_text = []
            start_idx = 0
            
            while start_idx < num_tokens:
                end_search = start_idx + target_size_tokens
                if end_search >= num_tokens:
                    expanded_start = token_to_sent.get(start_idx, (start_idx, start_idx))[0]
                    chunk_ids = input_ids[expanded_start:]
                    chunks_text.append(tokenizer.decode(chunk_ids))
                    break

                window_start = max(start_idx + 1, end_search - search_window)
                window_end = min(num_tokens - 1, end_search + search_window)
                
                best_cut_idx = window_start
                min_overlap = float('inf')
                
                for i in range(window_start, window_end):
                    if overlap_counts[i] < min_overlap:
                        min_overlap = overlap_counts[i]
                        best_cut_idx = i
                    elif overlap_counts[i] == min_overlap:
                        if abs(i - end_search) < abs(best_cut_idx - end_search):
                            best_cut_idx = i

                expanded_start = token_to_sent.get(start_idx, (start_idx, start_idx))[0]
                last_token = best_cut_idx - 1
                expanded_end = token_to_sent.get(last_token, (last_token, last_token))[1] + 1
                
                chunk_ids = input_ids[expanded_start:expanded_end]
                chunks_text.append(tokenizer.decode(chunk_ids))
                
                print(f"Created chunk: {start_idx} to {best_cut_idx} (Weighted Overlap {min_overlap})")
                start_idx = best_cut_idx
                
            file_name = f"doc_{k}_segmented_low_accessible_antecedent.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("=== ORIGINAL TEXT ===\n")
                f.write(full_text)
                f.write("\n\n=== SEGMENTED TEXT (BROKEN BY LINEBREAKS) ===\n")
                f.write("\n=======\n".join(chunks_text))
            if(test):
                break # early break for testing
            k = k + 1 

        return chunks_text








