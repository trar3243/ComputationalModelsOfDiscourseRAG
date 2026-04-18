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

# 3 - Load chunked text entries in from a CSV file.
#Define CSV loader, linebreaking disabled. 
#Assumes all text chunked, all entries in a single CSV column, no column header.
def load_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as textfile:
        reader = csv.reader(textfile)
        for row in reader:
            yield row[0]
            
#Define batch loader to not ingest the entire CSV in one go.
def batch_loader(generator, batch_size=200):
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
            
#Add loaded batch to Chroma Database
#We can build this outfurther to append metadata
data_gen = load_csv("INSERT_FILEPATH_HERE.csv")
for i, batch in enumerate(batch_loader(data_gen, batch_size=200)):
    try:
        collection.add(
            documents=batch,
            # Uses 128-bit random generator to make unique global ID per chunk entry
            ids=[str(uuid.uuid4()) for _ in batch])
    #Prints failure point if there is an error.
    except Exception as e:
        print(f"Error inserting batch {i}: {e}")

# 4 - Basic query for Database Collection
# Returns n chunks with highest embedding similarity to query text
results = collection.query(
    #Input any query text here
    query_texts=["How many teams?"],
    #Number of results returned
    n_results=1
)
print(results["documents"])
