import os
import json
import re
import nltk
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

# Load the BERT tokenizer for Turkish
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def token_len(text):
    """Calculate the number of tokens using the Turkish BERT tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=True))

def read_documents(directory):
    """Read all text documents from a directory."""
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
                documents.append((file, f.read()))
    return documents

def extract_sections(text):
    """Extract sections based on SECTION_START and SECTION_END markers."""
    sections = []
    section_pattern = re.findall(r"#SECTION_START#.*?#SECTION_TOPIC: (.*?)#(.*?)#SECTION_END#", text, re.DOTALL)
    
    for topic, content in section_pattern:
        sections.append({"topic": topic.strip(), "content": content.strip()})
    
    return sections

def semantic_chunking_with_langchain(text, max_tokens=480, min_chunk_length=100):
    """
    Perform semantic chunking using LangChain's text splitters with improved context preservation.
    
    Args:
        text (str): The text to chunk
        max_tokens (int): Maximum tokens per chunk (default: 480)
        min_chunk_length (int): Minimum characters per chunk to avoid tiny fragments
        
    Returns:
        list: List of text chunks that respect semantic boundaries and token limits
    """
    # First use RecursiveCharacterTextSplitter with better parameters
    semantic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 5,  # Character approximation
        chunk_overlap=100,  
        separators=["\n\n", "\n", ".", "! ", "? ", ":"],
        keep_separator=True
    )
    
    # Get initial semantic chunks
    semantic_chunks = semantic_splitter.split_text(text)
    
    # Then use CharacterTextSplitter with custom token counter and better parameters
    token_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=max_tokens,
        chunk_overlap=50,  
        length_function=token_len  # Use the Turkish BERT tokenizer
    )
    
    final_chunks = []
    for chunk in semantic_chunks:
        # Skip chunks that are too small on their own
        if len(chunk) < min_chunk_length and len(semantic_chunks) > 1:
            continue
            
        # Further split if any chunk is still too large
        token_chunks = token_splitter.split_text(chunk)
        
        # Filter out tiny chunks
        valid_chunks = [c for c in token_chunks if len(c) >= min_chunk_length]
        
        # Merge tiny adjacent chunks if necessary
        if len(valid_chunks) < len(token_chunks):
            merged_chunks = []
            temp_chunk = ""
            for c in token_chunks:
                if len(c) < min_chunk_length:
                    temp_chunk += c
                else:
                    if temp_chunk:
                        if len(temp_chunk) >= min_chunk_length:
                            merged_chunks.append(temp_chunk)
                        elif merged_chunks:
                            merged_chunks[-1] += temp_chunk
                        temp_chunk = ""
                    merged_chunks.append(c)
            
            if temp_chunk and len(temp_chunk) >= min_chunk_length:
                merged_chunks.append(temp_chunk)
            elif temp_chunk and merged_chunks:
                merged_chunks[-1] += temp_chunk
                
            final_chunks.extend(merged_chunks)
        else:
            final_chunks.extend(valid_chunks)
    
    # Post-processing: merge any remaining tiny chunks
    if final_chunks:
        i = 0
        while i < len(final_chunks):
            if len(final_chunks[i]) < min_chunk_length:
                # If we can merge with the next chunk
                if i+1 < len(final_chunks) and token_len(final_chunks[i] + final_chunks[i+1]) <= max_tokens:
                    final_chunks[i+1] = final_chunks[i] + final_chunks[i+1]
                    final_chunks.pop(i)
                # If we can merge with the previous chunk
                elif i > 0 and token_len(final_chunks[i-1] + final_chunks[i]) <= max_tokens:
                    final_chunks[i-1] = final_chunks[i-1] + final_chunks[i]
                    final_chunks.pop(i)
                else:
                    i += 1
            else:
                i += 1
    
    return final_chunks

def process_documents(directory):
    """Read and parse all documents in the given directory using semantic chunking."""
    processed_data = []
    documents = read_documents(directory)
    
    for filename, text in documents:
        sections = extract_sections(text)
        for section in sections:
            chunks = semantic_chunking_with_langchain(section["content"])
            for i, chunk in enumerate(chunks):
                processed_data.append({
                    "filename": filename,
                    "topic": section["topic"],
                    "chunk_id": i + 1,
                    "content": chunk,
                    "token_count": token_len(chunk)  # Include token count for verification
                })
    
    return processed_data

def save_json(data, output_file):
    """Save the processed data as a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def verify_chunks(data, max_tokens=512):
    """Verify that no chunk exceeds the maximum token limit."""
    exceeded = []
    for item in data:
        if item["token_count"] > max_tokens:
            exceeded.append((item["filename"], item["chunk_id"], item["token_count"]))
    
    if exceeded:
        print(f"WARNING: {len(exceeded)} chunks exceed the token limit of {max_tokens}:")
        for filename, chunk_id, count in exceeded[:5]:  # Show first 5 examples
            print(f"  - {filename}, chunk {chunk_id}: {count} tokens")
        if len(exceeded) > 5:
            print(f"  ... and {len(exceeded) - 5} more")
    else:
        print(f"All chunks are within the token limit of {max_tokens}")

if __name__ == "__main__":
    input_directory = "MEB - Etiketlenmiş"
    output_file = "processed_corpus.json"
    
    print(f"Processing documents from {input_directory}...")
    structured_data = process_documents(input_directory)
    
    # Verify token counts
    verify_chunks(structured_data)
    
    # Save the processed data
    save_json(structured_data, output_file)
    print(f"Processed corpus saved to {output_file}")
    
    # Summary statistics
    total_chunks = len(structured_data)
    avg_tokens = sum(item["token_count"] for item in structured_data) / total_chunks if total_chunks > 0 else 0
    
    print(f"Total chunks: {total_chunks}")
    print(f"Average tokens per chunk: {avg_tokens:.2f}")