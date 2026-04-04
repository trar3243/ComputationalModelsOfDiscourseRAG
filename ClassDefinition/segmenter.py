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
    def get_clusters(self, text, max_chunk_chars=13000, overlap_chars=4000):
        # had to do special stuff to not crash GPU memory 
        # basically has sliding window over chunks of the text, then resolves the coreference chain. Means that a chain that spans more than 4000 characters which has not mentions within that window
        # is lost. However, this is likely not an issue with the purpose of this class. 
        if not text.strip(): 
            return []
            
        all_shifted_clusters = []
        i = 0
        text_len = len(text)
        
        # drop gradients, make GPU happier 
        with torch.inference_mode():
            while i < text_len:
                end = min(i + max_chunk_chars, text_len)
                
                if end < text_len:
                    safe_cut = text.rfind('\n', i, end)
                    if safe_cut == -1 or safe_cut < i + (max_chunk_chars // 2):
                        safe_cut = text.rfind(' ', i, end)
                    
                    if safe_cut != -1:
                        end = safe_cut + 1 
                
                text_chunk = text[i:end]
                
                if text_chunk.strip():
                    preds = self.model.predict(texts=[text_chunk], max_tokens_in_batch=1024)
                    chunk_clusters = preds[0].get_clusters(as_strings=False)
                    
                    for cluster in chunk_clusters:
                        shifted_cluster = []
                        for start, end_idx in cluster:
                            #see what the model found
                            mention_text = text_chunk[start:end_idx]
                            
                            # Only keep it if it contains actual alphanumeric characters/punctuation
                            if mention_text.strip():
                                shifted_cluster.append((start + i, end_idx + i))
                        
                        # If filtering out the whitespace left us with less than 2 mentions,
                        # it is no longer a valid coreference link
                        if len(shifted_cluster) > 1:
                            all_shifted_clusters.append(shifted_cluster)
                
                if end >= text_len:
                    break
                    
                next_start_tentative = end - overlap_chars
                
                safe_start = text.find('\n', next_start_tentative, end)
                if safe_start == -1:
                    safe_start = text.find(' ', next_start_tentative, end)
                
                if safe_start != -1 and safe_start > i:
                    i = safe_start + 1
                else:
                    i = next_start_tentative

        # Merging
        span_graph = {}
        for cluster in all_shifted_clusters:
            for span in cluster:
                if span not in span_graph:
                    span_graph[span] = set()
                for other_span in cluster:
                    if span != other_span:
                        span_graph[span].add(other_span)

        visited = set()
        final_global_clusters = []
        
        for span in span_graph:
            if span not in visited:
                component = []
                stack = [span]
                
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        component.append(node)
                        stack.extend(list(span_graph[node] - visited))
                
                if len(component) > 1:
                    final_global_clusters.append(component)
                    
        return final_global_clusters
    
    def __char_to_token__(self,char_idx, offsets):
        for i, (start, end) in enumerate(offsets):
            if start <= char_idx < end:
                return i
            # If the char is whitespace/unmapped before this token
            if char_idx < start:
                return max(0, i - 1)
        return len(offsets) - 1

    # basic idea: draw an edge from each mention to the most recent prior mention. Edge weighted according to the accessability of the postcedent, if self.weighted==True 
    def __most_recent_cluster_graph_addition__(self,cluster,text,graph,offsets,tie_to_first=False):
        prior_t_start = None 
        first_t_start = None 
        # prior_t_end = None 
        for i in range(len(cluster)):
            if (i==0): 
                prior_t_start = self.__char_to_token__(cluster[i][0], offsets)
                first_t_start = self.__char_to_token__(cluster[i][0], offsets)
                # prior_t_end = self.__char_to_token__(cluster[i][1], offsets)
                continue # not drawing an edge for first mention.. |E| = |V| - 1 
            t_start = self.__char_to_token__(cluster[i][0], offsets)
            t_end = self.__char_to_token__(cluster[i][1], offsets)
            weight = self.__get_accessibility_score__(text[cluster[i][0]:cluster[i][1]]) if self.weighted else 1.0 
            if(not tie_to_first):
                while(prior_t_start <= t_end):
                    graph[prior_t_start] = graph[prior_t_start] + weight 
                    prior_t_start = prior_t_start + 1 
            else:
                k = first_t_start
                while(k <= t_end):
                    graph[k] = graph[k] + weight 
                    k = k + 1 
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
            graph = self.__most_recent_cluster_graph_addition__(sub_cluster,text,graph,offsets,True) # force tie to first 
        return graph 



    # returns an array of chunks 
    def coreference_chunk_text(self, text:str, clusters, method: str, target_size_tokens: int, search_window: int, full_sentence_inclusion: bool, weighted:bool):
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
                chunk_end = token_to_sent[chunk_end][1]
                
                chunk_end = min(chunk_end, num_tokens - 1)
                
            chunks.append(text[offsets[chunk_start][0]:offsets[chunk_end][1]])
            
            chunk_start = chunk_end + 1

        return chunks 

    








