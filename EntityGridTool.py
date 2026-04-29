"""
IMPORTS
"""
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

"""
DEFINE ENTITY GRID CLASS 
"""
class ChunkedEntityGridViewer:
    def __init__(self, top_k=10):
        #LOAD ENGLISH SPACY
        self.nlp = spacy.load("en_core_web_sm")
        #USER DESIGNATED TOP-K
        self.top_k = top_k

    #FOR CONSISTENCY, GENERATES GRIDS FROM SAME BASE TEXT INPUT
    def rebuild_text(self, chunks):
        return " ".join(chunks["chunks"])

    #FUNCTION - GET CHUNK SPANS
    def build_chunk_spans(self, chunks):
        spans = []
        #START AT BEGINNING OF DOCUMENT
        cursor = 0
        #ITERATE THROUGH INPUT CHUNKS
        for chunk in chunks["chunks"]:
            #SPAN IS BETWEEN "CURSOR" START AND LENGTH O CHUNK
            doc = self.nlp(chunk)
            start = cursor
            end = cursor + len(doc)
            #APPEND THIS TO SPAN LIST
            spans.append((start, end))
            #MOVE CURSOR FORWARD
            cursor = end
        #RETURNS ALL SPANS FOR GIVEN METHOD
        return spans

    #FUNCTION - CHUNK ASSIGNMENT FOR SENTENCES
    def sentence_chunk_overlap(self, sentences, spans):
        result = []
        #ITERATE THROUGH SENTENCES
        for sent in sentences:
            #TAKE BEGIN AND END OF SENTENCE
            s, e = sent.start, sent.end
            #INSTANTIATE EMPTY SET OF OVERLAPS FOR INDIVIDUAL SENTENCE
            overlaps = set()
            #ITERATE THROUGH SPANS, DESIGNATE MATCHES
            for i, (cs, ce) in enumerate(spans):
                if not (e <= cs or s >= ce):
                    overlaps.add(i)
            #APPEND TO OVERLAP SETS FOR EACH SENTENCE 
            result.append(overlaps)
        return result

    #FUNCTION - GET TOP ENTITITES WITH SPACY
    def get_top_entities(self, sentences):
        freq = Counter()
        #ITERATE THROUGH SENTENCES, DESIGNATE A WORD AN ENTITY IF NOUN OR PROPER NOUN
        for sent in sentences:
            for t in sent:
                if t.pos_ in ("NOUN", "PROPN"):
                    #IF ENTITY, TAKE LEMMA AND ADD COUNTER
                    freq[t.lemma_.lower()] += 1
        #RETURN TOP K ENTITIES BY COUNT
        return [e for e, _ in freq.most_common(self.top_k)]

    #FUNCTION - DESIGNATE ENTITY ROLES FOR GIRD    
    def role(self, token):
        #IF SUBJECT POSITION, S
        if token.dep_ in ("nsubj", "nsubjpass"):
            return "S"
        #IF OBJECT POSITION, O
        elif token.dep_ in ("dobj", "pobj", "iobj"):
            return "O"
        #ELSE X
        return "X"

    #FUNCTION - MAKE ENTITY MAPS OF SENTENCES
    def sentence_maps(self, sentences, top_entities):
        maps = []
        #ITERATE THROUGH SENTENCES
        for sent in sentences:
            #INTANTIATE ENTITITES DICTIONARY FOR SENTENCE
            ents = {}
            #IF WORD IS ENTITY, PUT LEMMA IN DICTIONARY
            for t in sent:
                if t.pos_ in ("NOUN", "PROPN"):
                    ent = t.lemma_.lower()
                    #IF IN TOP K USER REQUESTED ENTITIES, ALSO GET ITS ROLE FOR DISPLAY
                    if ent in top_entities:
                        ents[ent] = self.role(t)
            #APPEND DICTIONARY FOR THIS SENTENCE TO LIST
            maps.append(ents)
        return maps

    #FUNCTION - BUILD GRID WITH SENTENCE-ENTITY MAPS AND TOP-K ENTITIES
    def build_grid(self, sent_maps, top_entities):
        return [[m.get(ent, "-") for ent in top_entities] for m in sent_maps]

    #FUNCTION - DRAW PANEL FOR GIVEN METHOD
    def draw_panel(self, ax, df, sentence_chunks, title, colors):
        #SET SENTENCES TO ROWS, TOP-K ENTITIES TO COLUMNS
        sentences = df.index
        top_entities = df.columns
        x_positions = np.linspace(0, len(top_entities) - 1, len(top_entities))

        #SET AXIS BOUNDS
        ax.set_xlim(-0.4, len(top_entities))
        ax.set_ylim(0, len(sentences))
        ax.set_aspect(0.25)

        #X-AXIS LABELLING, FORMATTING
        ax.set_xticks(x_positions)
        ax.set_xticklabels(top_entities, rotation=25, ha="left", fontsize=8)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        #Y-AXIS LABELLING, FORMATTING
        ax.set_yticks(range(len(sentences)))
        ax.set_yticklabels(sentences, fontsize=4)
        ax.tick_params(axis='both', length=0)

        #APPLY SENTENCE COLORING WITH ALTERNATION PER CHINK
        for i, chunk_set in enumerate(sentence_chunks):
            #IF ONE CHUNK ASSIGNED, APPLY SINGLE COLOR TO ROW, LOW ALPHA
            if len(chunk_set) == 1:
                c = list(chunk_set)[0]
                color = colors[0] if c % 2 == 0 else colors[1]
                ax.axhspan(i - 0.5, i + 0.5, color=color, alpha=0.22)
            #IF MORE THAN ONE CHUNK ASSIGNED, APPLY BLENDED COLOR 
            elif len(chunk_set) >= 2:
                for j, c in enumerate(list(chunk_set)[:2]):
                    ax.axhspan(i - 0.5, i + 0.5, color=colors[j % len(colors)], alpha=0.20)

        #TEXT FORMATTING
        for i in range(len(sentences)):
            for j, ent in enumerate(top_entities):
                ax.text(x_positions[j], i, df.iloc[i, j], ha="center", va="center", fontsize=7, color="black")
        #PLACE ENTITY NAMES ON TOP
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, pad=6)

    #MAIN DISPLAY FUNCTION
    #ASSUMES INPUTS FROM 4 CHUNKING METHODS OF SAME ARTICLE
    def show(self, chunks_1, chunks_2, chunks_3, chunks_4):

        #SENTENCE LAYOUT DEFINED BY FIRST TEXT INPUT
        text = self.rebuild_text(chunks_1)
        doc = self.nlp(text)
        sentences = list(doc.sents)
        #GET TOP ENTITIES
        top_entities = self.get_top_entities(sentences)
        #GET SENTENCE-ENTITY MAPS, BUILD THE GRID
        sent_maps = self.sentence_maps(sentences, top_entities)
        grid = self.build_grid(sent_maps, top_entities)

        df = pd.DataFrame(grid,columns=top_entities,index=[f"S{i+1}" for i in range(len(sentences))])

        #BUILD SPANS FOR ALL 4 METHODS
        spans_1 = self.build_chunk_spans(chunks_1)
        spans_2 = self.build_chunk_spans(chunks_2)
        spans_3 = self.build_chunk_spans(chunks_3)
        spans_4 = self.build_chunk_spans(chunks_4)

        #MAP OVERLAYS FOR ALL METHODS ON THE SAME SENTENCES
        chunks_1_map = self.sentence_chunk_overlap(sentences, spans_1)
        chunks_2_map = self.sentence_chunk_overlap(sentences, spans_2)
        chunks_3_map = self.sentence_chunk_overlap(sentences, spans_3)
        chunks_4_map = self.sentence_chunk_overlap(sentences, spans_4)

        #PLOT ALL CHARTS, DRAW PANELS TO DISPLAY AT ONCE
        per_chart_width = 12
        fig, axes = plt.subplots(1, 4,figsize=(per_chart_width * 4, max(3, len(sentences) * 0.18)))

        self.draw_panel(axes[0], df, chunks_1_map,
            "CoRegRAG Method A",
            ["lightgreen", "lightblue"]
        )

        self.draw_panel(axes[1], df, chunks_2_map,
            "CoRegRAG Method B",
            ["lightgreen", "lightblue"]
        )

        self.draw_panel( axes[2], df, chunks_3_map,
            "CoRegRAG Method C",
            ["lightgreen", "lightblue"]
        )

        self.draw_panel(axes[3], df, chunks_4_map,
            "Naive Baseline Chunking",
            #DIFFERENTIATE COLORS FOR BASELINE
            ["plum", "khaki"]
        )

        plt.subplots_adjust(top=0.92)
        plt.tight_layout()
        plt.show()

        return df, chunks_1_map, chunks_2_map, chunks_3_map, chunks_4_map

#INSTANTIATE GRID VIEW
#USER DESIGNATED TOP-K ENTITIES DISPLAYED
viewer = ChunkedEntityGridViewer(top_k=10)

#MANUAL INPUT CHUNKS OUTPUT FROM CHUNKING PHASE 1
#EACH IS A DICTIONARY
chunks_1 = {"chunks": []}

chunks_2 = { "chunks": []}

chunks_3 = { "chunks": []}

chunks_4 = { "chunks": []}

#DISPLAY ALL 4 GRIDS
df, c1, c2, c3, c4 = viewer.show(
    chunks_1,
    chunks_2,
    chunks_3,
    chunks_4
)
