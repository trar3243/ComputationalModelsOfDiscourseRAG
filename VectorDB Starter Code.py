###INSTALL IF NEEDED
#pip install chromadb openai

#IMPORT
import chromadb

# 1 - CONNECT TO DATABASE 
client = chromadb.CloudClient(
  api_key='INSERT API KEY HERE',
  tenant='d518a61e-e4ea-403e-830c-a8e2833e1b16',
  database='CoRefRAG'
)

# 2 - CONNECT TO COLLECTION
# We can have multiple text collections under the database
collection = client.get_collection("CoRefRAG_Collection")

# 3 - Add chunked text entries to the Data
# This uses the free default all-MiniLM-L6-v2 embedding function
collection.add( documents=[
        "The NCAA Division I men's basketball tournament, branded as March Madness",
        "Played mostly during March, the tournament was first conducted in 1939 and currently consists of 68 teams."
    ],
    #ADD A CORRESPONDING ID TO EACH ENTRY
    #To Do - I'll make a function to do this en masse without manual entry
    ids=["1", "2"]
)

# 4 - Basic query for Database Collection
# Returns n chunks with highest embedding similarity to query text
results = collection.query(
    #Input any query text here
    query_texts=["How many teams?"],
    #Number of results returned
    n_results=1
)
print(results["documents"])
