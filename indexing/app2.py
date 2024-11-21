import streamlit as st
from generation import Generator
from indexing.graph_communities import CommunitySummarizer
from indexing.data_index import DataIndexer
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import random

# Custom color palette
COLORS = {
    'primary': '#1f77b4',  # Blue
    'secondary': '#2ca02c',  # Green
    'accent': '#ff7f0e',  # Orange
    'background': '#f0f2f6',
    'text': '#2c3e50',
    'node': '#3498db',
    'edge': '#95a5a6'
}

class InsuranceRAGApp:
    def __init__(self):
        st.set_page_config(
            page_title="Insurance Knowledge Assistant",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize RAG components
        self.summarizer = CommunitySummarizer()
        self.indexer = DataIndexer()
        self.summarizer.load()
        self.generator = Generator(self.indexer, self.summarizer)

    def plot_entity_network(self, entities):
        """Create a visually appealing network visualization"""
        # Create nodes dataframe
        nodes = pd.DataFrame([
            {
                "id": e.name,
                "label": e.metadata.get('label', 'Unknown') if hasattr(e, 'metadata') else 'Unknown',
                "size": random.randint(20, 40)  # Random size for visual variety
            } 
            for e in entities
        ])
        
        # Create edges
        edges = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                edges.append({
                    "source": e1.name,
                    "target": e2.name,
                    "weight": random.uniform(1, 3)  # Random weight for line thickness
                })
        edges = pd.DataFrame(edges)
        
        # Create network
        G = nx.from_pandas_edgelist(edges, 'source', 'target')
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        for edge in edges.itertuples():
            x0, y0 = pos[edge.source]
            x1, y1 = pos[edge.target]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    line=dict(width=edge.weight, color=COLORS['edge']),
                    mode='lines',
                    hoverinfo='none'
                )
            )
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_sizes.append(nodes[nodes['id'] == node]['size'].iloc[0])
            
        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=COLORS['node'],
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                text=node_text,
                textposition='top center',
                hoverinfo='text',
                textfont=dict(color=COLORS['text'])
            )
        )
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            margin=dict(t=40, b=40, l=40, r=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title=dict(
                text="Knowledge Graph Visualization",
                x=0.5,
                y=0.95,
                font=dict(size=20, color=COLORS['text'])
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_sidebar(self):
        with st.sidebar:
            st.title("🏥 Insurance Assistant")
            st.markdown(
                f"""
                <div style='background-color: {COLORS['background']}; padding: 15px; border-radius: 5px;'>
                This AI-powered assistant helps you understand insurance concepts and requirements.
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown("### 📝 Example Questions")
            example_questions = [
                "What are the different types of auto insurance coverage?",
                "How does workers compensation insurance work?",
                "What is the difference between term and whole life insurance?",
                "Insurance policy requirements by state?",
                "What is the process for handling insurance claims?"
            ]
            
            for q in example_questions:
                if st.button(
                    q,
                    key=q,
                    help=f"Click to ask: {q}"
                ):
                    st.session_state.query = q
            
            st.markdown("---")
            st.markdown(
                """
                ### 🛠️ Built with
                - LlamaIndex 🦙
                - Neo4j 📊
                - OpenAI 🤖
                - Streamlit 🌟
                """
            )

    def render_main(self):
        st.title("Insurance Knowledge Assistant 🏥")
        
        # Query input with custom styling
        query = st.text_input(
            "Ask your insurance question:",
            key="query",
            placeholder="e.g., What are the different types of auto insurance coverage?",
            help="Type your question here and press Enter"
        )
        
        if query:
            with st.spinner("🤔 Thinking..."):
                response = self.generator.generate(query)
                entities = self.generator.get_entities(query)
                summaries = self.generator.get_community_summaries(query)
                
                # Display results in styled tabs
                tabs = st.tabs(["💡 Answer", "🔗 Related Concepts", "📚 Sources"])
                
                with tabs[0]:
                    st.markdown(
                        f"""
                        <div style='background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid {COLORS['primary']};'>
                        {response}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with tabs[1]:
                    if entities:
                        # Display network visualization
                        self.plot_entity_network(entities)
                        
                        # Display entities list
                        st.markdown("#### Related Terms")
                        entity_names = [e.name for e in entities]
                        cols = st.columns(3)
                        for i, name in enumerate(entity_names):
                            with cols[i % 3]:
                                st.markdown(
                                    f"""
                                    <div style='background-color: {COLORS['background']}; 
                                             padding: 10px; 
                                             border-radius: 5px; 
                                             margin: 5px 0;
                                             text-align: center;'>
                                    {name}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                
                with tabs[2]:
                    for i, summary in enumerate(summaries, 1):
                        with st.expander(f"📄 Source {i}"):
                            st.markdown(
                                f"""
                                <div style='background-color: white; 
                                         padding: 15px; 
                                         border-radius: 5px;
                                         border: 1px solid {COLORS['secondary']};'>
                                {summary}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

    def run(self):
        """Run the Streamlit app"""
        self.render_sidebar()
        self.render_main()

        # Add custom CSS
        st.markdown(
            f"""
            <style>
            .stApp {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .stButton>button {{
                width: 100%;
                border-radius: 5px;
                border: 1px solid {COLORS['primary']};
                background-color: white;
                color: {COLORS['primary']};
                transition: all 0.3s;
            }}
            .stButton>button:hover {{
                background-color: {COLORS['primary']};
                color: white;
            }}
            .stMarkdown {{
                color: {COLORS['text']};
            }}
            .stTab {{
                background-color: white;
                border-radius: 5px;
            }}
            .stTextInput>div>div>input {{
                border-radius: 5px;
            }}
            h1, h2, h3 {{
                color: {COLORS['text']};
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    app = InsuranceRAGApp()
    app.run()