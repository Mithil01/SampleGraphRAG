import streamlit as st
from generation import Generator
from indexing.graph_communities import CommunitySummarizer
from indexing.data_index import DataIndexer
import plotly.express as px
import pandas as pd

class InsuranceRAGApp:
    def __init__(self):
        st.set_page_config(
            page_title="Insurance Knowledge Assistant",
            page_icon="🏥",
            layout="wide"
        )
        
        # Initialize RAG components
        self.summarizer = CommunitySummarizer()
        self.indexer = DataIndexer()
        self.summarizer.load()
        self.generator = Generator(self.indexer, self.summarizer)

    def render_sidebar(self):
        with st.sidebar:
            st.title("🏥 Insurance Assistant")
            st.markdown("---")
            st.markdown("""
            This app helps you understand insurance concepts and requirements 
            using advanced AI and knowledge graph technology.
            """)
            
            # Example questions
            st.markdown("### Example Questions")
            example_questions = [
                "What are the different types of auto insurance coverage?",
                "How does workers compensation insurance work?",
                "What is the difference between term and whole life insurance?",
                "Insurance policy requirements by state?",
                "What is the process for handling insurance claims?"
            ]
            
            for q in example_questions:
                if st.button(q):
                    st.session_state.query = q
                    
            st.markdown("---")
            st.markdown("### About")
            st.markdown("""
            Built with:
            - LlamaIndex
            - Neo4j
            - OpenAI
            - Streamlit
            """)

    def render_main(self):
        st.title("Insurance Knowledge Assistant")
        
        # Query input
        query = st.text_input(
            "Ask a question about insurance:",
            key="query",
            placeholder="e.g., What are the different types of auto insurance coverage?"
        )
        
        if query:
            with st.spinner("Generating response..."):
                # Get response
                response = self.generator.generate(query)
                
                # Get entities and summaries for visualization
                entities = self.generator.get_entities(query)
                summaries = self.generator.get_community_summaries(query)
                
                # Display results in columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Answer")
                    st.write(response)
                    
                    st.markdown("### Related Concepts")
                    # Create network graph of related entities
                    if entities:
                        self.plot_entity_network(entities)
                
                with col2:
                    st.markdown("### Knowledge Sources")
                    for i, summary in enumerate(summaries, 1):
                        with st.expander(f"Source {i}"):
                            st.write(summary)
    
    def plot_entity_network(self, entities):
        """Create a network visualization of related entities"""
        # Create nodes dataframe
        nodes = pd.DataFrame([
            {"id": e.name, "type": e.type} for e in entities
        ])
        
        # Create edges between entities that are related
        edges = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                edges.append({
                    "source": e1.name,
                    "target": e2.name,
                    "weight": 1
                })
        edges = pd.DataFrame(edges)
        
        # Create network graph
        if not edges.empty:
            fig = px.scatter_3d(
                nodes, 
                x=[0]*len(nodes), 
                y=[0]*len(nodes), 
                z=[0]*len(nodes),
                text="id",
                color="type",
                title="Related Concepts Network"
            )
            st.plotly_chart(fig)

    def run(self):
        """Run the Streamlit app"""
        self.render_sidebar()
        self.render_main()

        # Add custom CSS
        st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
        }
        .stMarkdown {
            text-align: justify;
        }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = InsuranceRAGApp()
    app.run()