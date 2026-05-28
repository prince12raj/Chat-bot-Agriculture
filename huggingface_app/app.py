import pandas as pd
import numpy as np
import gradio as gr

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ====================================
# Load Dataset
# ====================================

qa_df = pd.read_csv(
    "agriculture_chatbot_dataset.csv"
)

# ====================================
# Load AI Model
# ====================================

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# ====================================
# Create Embeddings
# ====================================

questions = qa_df['question'].tolist()

question_embeddings = model.encode(
    questions,
    show_progress_bar=True
)

# ====================================
# Agriculture Chatbot Function
# ====================================

def agriculture_chatbot(user_query):

    user_query = str(user_query)

    query_embedding = model.encode(
        [user_query]
    )

    similarities = cosine_similarity(
        query_embedding,
        question_embeddings
    )

    best_match_index = np.argmax(
        similarities
    )

    response = qa_df.iloc[
        best_match_index
    ]['answer']

    return response

# ====================================
# Chat Interface
# ====================================

demo = gr.Interface(
    fn=agriculture_chatbot,

    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask agriculture related questions..."
    ),

    outputs="text",

    title="🌾 Agriculture AI Chatbot",

    description="""
    AI chatbot for:
    
    • Crop recommendation
    • Fertilizer recommendation
    • Market prices
    • Weather information
    • Smart farming
    • Agriculture guidance
    """
)

# ====================================
# Launch App
# ====================================

demo.launch(server_name="0.0.0.0")