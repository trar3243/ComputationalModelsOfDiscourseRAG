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
        self.method = None # set during run 
        self.weighted = None # set during run 


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
            # data = json.load(file)
            head = file.readlines() # [next(file) for _ in range(100)]
        return " ".join(head) # "Reggie is my cat. He is a good boy. He likes to play. Reggie is very loud. He meows all the time. He never ever stops. He is super noisy. He demands his food."
    def __char_to_token__(self,char_idx, offsets):
        for i, (start, end) in enumerate(offsets):
            if start <= char_idx < end:
                return i
            # If the char is whitespace/unmapped before this token
            if char_idx < start:
                return max(0, i - 1)
        return len(offsets) - 1

    """def __monolithic_cluster_graph_addition__(self,cluster, text, graph, offsets):
        # weights is list of token weights 
        # cluster is list of tuples 

        # use the maximum weight in the entire cluster for this SINGLE edge 
        weight = max(self.__get_accessibility_score__(text[s:e]) for s, e in cluster)

        c_start = min(span[0] for span in cluster) # starting character index of the entire cluster 
        c_end = max(span[1] for span in cluster)  # ending character index of the entire cluster 
        
        t_start = self.__char_to_token__(c_start, offsets) # starting token index of the entire cluster 
        t_end = self.__char_to_token__(c_end, offsets) # ending token index of the entire cluster 

        i = t_start 
        while(i <= t_end):
            graph[i] = graph[i] + weight 
            i = i + 1 
        return graph  """

    # basic idea: draw an edge from each mention to the most recent prior mention. Edge weighted according to the accessability of the postcedent, if self.weighted==True 
    def __most_recent_cluster_graph_addition__(self,cluster,text,graph,offsets):
        prior_t_start = None 
        # prior_t_end = None 
        for i in range(len(cluster)):
            if (i==0): 
                prior_t_start = self.__char_to_token__(cluster[i][0], offsets)
                # prior_t_end = self.__char_to_token__(cluster[i][1], offsets)
                continue # not drawing an edge for first mention.. |E| = |V| - 1 
            t_start = self.__char_to_token__(cluster[i][0], offsets)
            t_end = self.__char_to_token__(cluster[i][1], offsets)
            weight = self.__get_accessibility_score__(text[cluster[i][0]:cluster[i][1]]) if self.weighted else 1.0 
            while(prior_t_start <= t_end):
                graph[prior_t_start] = graph[prior_t_start] + weight 
                prior_t_start = prior_t_start + 1 
        return graph 

    # basic idea: draw an edge from each mention to the prior LOW ACCESSIBLE mention. Edge weighted according to the accessability of the postcedent, if self.weighted==True 
    def __most_recent_low_accessible_cluster_graph_addition__(self,cluster,text,graph,offsets):
        # we can create sub-clusters, then treat those as the clusters for the second mode 
        sub_clusters = list() 
        sub_cluster = list()
        for i in range(len(cluster)):
            mention_score = self.__get_accessibility_score__(text[cluster[i][0]:cluster[i][1]])
            if(mention_score <= 3.0 and len(sub_cluster)): # if it is a proper noun or something like "the king"
                sub_clusters.append(sub_cluster)
                sub_cluster = list()
            sub_cluster.append(cluster[i])
        sub_clusters.append(sub_cluster)
        for sub_cluster in sub_clusters:
            graph = self.__most_recent_cluster_graph_addition__(sub_cluster,text,graph,offsets)
        return graph 



    # returns an array of chunks 
    def coreference_chunk_text(self, text, method: str, target_size_tokens: int, search_window: int, full_sentence_inclusion: bool, weighted:bool):
        ### Initialization ### 
        if(method not in ["most_recent", "most_recent_low_accessible"]):
            raise Exception(f'Supplied method {method} not in ["most_recent", "most_recent_low_accessible"]')
        self.method = method # update the class instance for each run for method. 
        self.weighted = weighted # update the class instance for each run for weighted. 

        if not text.strip(): return list() 
        encoded = self.tokenizer(text,return_offsets_mapping=True, add_special_tokens=False) 

        # list of integer tuples. Each tuple refers to the start and end character index for each token 
        offsets = encoded['offset_mapping'] 

        # number of tokens 
        num_tokens = len(offsets) # number of tokens 

        # preds is a single-element list of CorefResults. Includes text (original text) and clusters. Each cluster here is a list of words within that cluster
        preds = self.model.predict(texts=[text]) 

        # clusters with as_string=False is a list of list of tuples. Each higher-level list is a list of clusters. Each cluster contains a list of tuples. Each tuple is the start and end character index of each mention.
        clusters = preds[0].get_clusters(as_strings=False)

        # for post-processing later 
        sent_char_spans = [m.span() for m in re.finditer(r'[^.!?\n]+[.!?\n]*', text)] if full_sentence_inclusion else list() 
        # token_to_sent is a dictionary of key=token index (range 0 to len tokens - 1), value = start and end token indices of sentence to which it belongs 
        token_to_sent = {}
        for s_start, s_end in sent_char_spans:
            t_start = self.__char_to_token__(s_start, offsets)
            t_end = self.__char_to_token__(s_end - 1, offsets) 
            
            for i in range(t_start, t_end + 1):
                token_to_sent[i] = (t_start, t_end)

        ### graph construction ###
        # graph spans the tokens, encodes cost of cutting at each token 
        graph = [0.0] * (num_tokens) 
        for cluster in clusters:
            if(method == "most_recent"):
                graph = self.__most_recent_cluster_graph_addition__(cluster,text,graph,offsets)
            elif(method == "most_recent_low_accessible"):
                graph = self.__most_recent_low_accessible_cluster_graph_addition__(cluster, text,graph,offsets)


        ### Chunk construction ### 
        chunks = list()
        chunk_start = 0 
        while chunk_start < num_tokens:
            window_start = chunk_start + target_size_tokens - search_window
            window_end = chunk_start + target_size_tokens + search_window
            
            # take rest of text 
            if window_start >= num_tokens:
                chunk_end = num_tokens - 1
            else:
                # Cap the window_end so we don't search past the end of the document
                actual_window_end = min(window_end, num_tokens)

                window_costs = graph[window_start:actual_window_end]
                # if all identical...
                if len(set(window_costs)) <= 1:
                    # Default to the exact target size. 
                    chunk_end = min(chunk_start + target_size_tokens - 1, num_tokens - 1)
                else:
                    # Initialize with infinity for both weight and distance
                    min_cut_tuple = (float('inf'), float('inf'))
                    best_cut = window_start 
                    # The absolute ideal cut based on target size alone 
                    ideal_cut = chunk_start + target_size_tokens - 1
                    
                    for i in range(window_start, actual_window_end):
                        distance_to_ideal = abs(i - ideal_cut)
                        # Create a tuple: (Primary Sorting, Secondary Sorting)
                        current_cut = (graph[i], distance_to_ideal)
                        # If there is a tie, it will pick the one with the smaller distance_to_ideal.
                        if current_cut < min_cut_tuple:
                            min_cut_tuple = current_cut
                            best_cut = i
                    
                    chunk_end = best_cut
            if full_sentence_inclusion and chunk_end in token_to_sent:
                # token_to_sent mapping looks like: {token_idx: (sentence_start_token, sentence_end_token)}
                # We grab the end token of the sentence
                chunk_end = token_to_sent[chunk_end][1]
                
                # Ensure we don't accidentally push past the total number of tokens
                chunk_end = min(chunk_end, num_tokens - 1)
                
            chunks.append(text[offsets[chunk_start][0]:offsets[chunk_end][1]])
            
            # Advance chunk_start to the token IMMEDIATELY FOLLOWING the cut
            # to prevent duplicating a token across two chunks.
            chunk_start = chunk_end + 1

        return chunks 

    
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
            full_text = text_data# doc # (doc.get('abstract') or "") + "\n" + (doc.get('text') or "")
            if not full_text.strip(): continue
            k = k + 1 
            if k==1: continue
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
                
                print(f"Created chunk: {start_idx} to {best_cut_idx} (Weighted Overlap: {min_overlap}, size = {best_cut_idx-start_idx})")
                start_idx = best_cut_idx
                
            file_name = f"segments/doc_{k}_segmented_monolithic.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("=== ORIGINAL TEXT ===\n")
                f.write(full_text)
                f.write("\n\n=== SEGMENTED TEXT (BROKEN BY LINEBREAKS) ===\n")
                f.write("\n=======\n".join(chunks_text))
            
            if(test):
                break # early break for testing

        return chunks_text

    ## IDEA 2: each edge is an edge from one entity mention to the most recent entity mention. The weight of the edge is based on the accessability of the second mention in each edge pair 
    def segment_most_recent(self,data_path, target_size_tokens=384, search_window=128, test=True):
        model = self.model
        text_data = self.get_text(data_path)
        device = self.device 
        tokenizer= self.tokenizer

        k = 0 
        for doc in text_data:
            full_text = text_data# doc # (doc.get('abstract') or "") + "\n" + (doc.get('text') or "")
            if not full_text.strip(): continue
            k = k + 1
            if k == 1: continue
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
                

                print(f"Created chunk: {start_idx} to {best_cut_idx} (Weighted Overlap {min_overlap} size = {best_cut_idx-start_idx})")
                start_idx = best_cut_idx
                
            file_name = f"segments/doc_{k}_segmented_most_recent.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("=== ORIGINAL TEXT ===\n")
                f.write(full_text)
                f.write("\n\n=== SEGMENTED TEXT (BROKEN BY LINEBREAKS) ===\n")
                f.write("\n=======\n".join(chunks_text))
            
            if(test):
                break # early break for testing

        return chunks_text

    ## IDEA 3: as before, weight each edge according to its second pair. However, the antecedent for each edge is the last antecedent which had an antecedent score geq 3 (includes proper nouns, names, definite noun phrase )
    def segment_low_accessible_antecedent(self,data_path, target_size_tokens=384, search_window=128, test=True):
        model = self.model
        device = self.device 
        tokenizer=self.tokenizer
        text_data = self.get_text(data_path)

        k = 0 
        for doc in text_data:
            full_text = text_data # (doc.get('abstract') or "") + "\n" + (doc.get('text') or "")
            if not full_text.strip(): continue
            k = k + 1 
            if k==1: continue
            # full_text = small_example
            
            #tokenize text to get input_ids and character offsets mapping
            encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoded['input_ids']
            offsets = encoded['offset_mapping']
            num_tokens = len(input_ids)

            sent_char_spans = [m.span() for m in re.finditer(r'[^.!?\n]+[.!?\n]*', full_text)] 
            sent_char_spans = [m.span() for m in re.finditer(r'[ghuadshfdsadsafdsafdsadwsfe]', full_text)] 
            
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
                
                print(f"Created chunk: {start_idx} to {best_cut_idx} (Weighted Overlap {min_overlap} size = {best_cut_idx-start_idx})")
                start_idx = best_cut_idx
                
            file_name = f"segments/doc_{k}_segmented_low_accessible_antecedent.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("=== ORIGINAL TEXT ===\n")
                f.write(full_text)
                f.write("\n\n=== SEGMENTED TEXT (BROKEN BY LINEBREAKS) ===\n")
                f.write("\n=======\n".join(chunks_text))
            if(test):
                break # early break for testing

        return chunks_text








