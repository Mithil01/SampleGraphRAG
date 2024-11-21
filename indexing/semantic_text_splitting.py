from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import TokenTextSplitter, SentenceSplitter
from llama_index.core.schema import MetadataMode
from typing import List, Dict
import pandas as pd

class LlamaIndexTextSplitter:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 20):
        # Initialize text splitters
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize node parsers
        self.token_parser = SimpleNodeParser.from_defaults(
            text_splitter=self.token_splitter
        )
        
        self.sentence_parser = SimpleNodeParser.from_defaults(
            text_splitter=self.sentence_splitter
        )

    def split_by_tokens(self, text: str) -> List[str]:
        """
        Split text into chunks based on tokens
        """
        doc = Document(text=text)
        nodes = self.token_parser.get_nodes_from_documents([doc])
        
        # Extract text from nodes
        chunks = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
        return chunks

    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentence boundaries
        """
        doc = Document(text=text)
        nodes = self.sentence_parser.get_nodes_from_documents([doc])
        
        # Extract text from nodes
        chunks = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
        return chunks

    def split_by_topics(self, text: str) -> List[Dict]:
        """
        Split text into topic-based chunks
        """
        # First split into base chunks
        doc = Document(text=text)
        base_nodes = self.sentence_parser.get_nodes_from_documents([doc])
        
        # Group chunks by topic using metadata
        topics = {}
        current_topic = "General"
        
        for node in base_nodes:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            
            # Try to identify topic from content
            topic_indicators = [
                "AUTO INSURANCE",
                "LIFE INSURANCE",
                "PROPERTY INSURANCE",
                "LIABILITY INSURANCE",
                "WORKERS COMPENSATION",
                "HEALTH INSURANCE"
            ]
            
            for indicator in topic_indicators:
                if indicator.lower() in text.lower():
                    current_topic = indicator
            
            if current_topic not in topics:
                topics[current_topic] = []
            
            topics[current_topic].append(text)
        
        # Format results
        topic_chunks = [
            {
                "topic": topic,
                "content": " ".join(chunks),
                "chunk_count": len(chunks)
            }
            for topic, chunks in topics.items()
        ]
        
        return topic_chunks

    def analyze_chunks(self, chunks: List[str]) -> pd.DataFrame:
        """
        Analyze the characteristics of text chunks
        """
        analysis = []
        for i, chunk in enumerate(chunks):
            analysis.append({
                "chunk_id": i,
                "length": len(chunk),
                "sentences": len(chunk.split('.')),
                "words": len(chunk.split()),
                "preview": chunk[:100] + "..."
            })
        
        return pd.DataFrame(analysis)

# Usage example
def process_insurance_text():
    # Example insurance text
    insurance_text = """
    AUTO INSURANCE
    Auto insurance protects against financial loss in the event of an accident. 
    It is a contract between the policyholder and the insurance company.
    
    LIFE INSURANCE
    Life insurance provides financial protection to beneficiaries upon the death of the insured.
    There are two main types: term and whole life insurance.
    
    PROPERTY INSURANCE
    Property insurance covers damage to property from various perils.
    This includes coverage for homes, businesses, and other structures.
    """
    
    # Initialize splitter
    splitter = LlamaIndexTextSplitter(chunk_size=512, chunk_overlap=50)
    
    # Split by different methods
    token_chunks = splitter.split_by_tokens(insurance_text)
    sentence_chunks = splitter.split_by_sentences(insurance_text)
    topic_chunks = splitter.split_by_topics(insurance_text)
    
    # Analyze results
    print("\nToken-based chunks:")
    print(f"Generated {len(token_chunks)} chunks")
    token_analysis = splitter.analyze_chunks(token_chunks)
    print(token_analysis)
    
    print("\nSentence-based chunks:")
    print(f"Generated {len(sentence_chunks)} chunks")
    sentence_analysis = splitter.analyze_chunks(sentence_chunks)
    print(sentence_analysis)
    
    print("\nTopic-based chunks:")
    for topic in topic_chunks:
        print(f"\nTopic: {topic['topic']}")
        print(f"Content preview: {topic['content'][:200]}...")
        print(f"Number of original chunks: {topic['chunk_count']}")

    return {
        'token_chunks': token_chunks,
        'sentence_chunks': sentence_chunks,
        'topic_chunks': topic_chunks,
        'token_analysis': token_analysis,
        'sentence_analysis': sentence_analysis
    }

if __name__ == "__main__":
    results = process_insurance_text()
    
    # Example of accessing specific chunks
    print("\nSample token chunk:")
    print(results['token_chunks'][0])
    
    print("\nSample topic content:")
    print(results['topic_chunks'][0]['content'])
    
    # Export analysis to CSV if needed
    results['token_analysis'].to_csv('token_chunk_analysis.csv', index=False)
    results['sentence_analysis'].to_csv('sentence_chunk_analysis.csv', index=False)