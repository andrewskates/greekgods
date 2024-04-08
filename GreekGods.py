import streamlit as st
import pandas as pd
import numpy as np

from openai import OpenAI

from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


# Initialise session with OpenAI
client = OpenAI(
    api_key=st.secrets["openAI_api_key"],)

OPENAI_API_MODEL_ID = "text-embedding-ada-002"


# Functions

# Get embedding of piece of text
def get_embedding(s):
    res = client.embeddings.create(input=s, model=OPENAI_API_MODEL_ID)
    return res.data[0].embedding

# Compares two embeddings and returns a value which indicates closeness of semantic field - which indicates how similar the respective text is
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)

# Iterates through the dataframe and calculates the embeddings for each row
def get_embeddings_from_dataframe(dataframe, Descriptions):
    embeddings = []
    
    for text in dataframe[Descriptions]:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    
    return embeddings

# Gets the embedding for a concept and then compares that concept with each of the embeddings for all of the Gods in the dataframe
def compare_characteristic(concept,embeddings,df):
    concept_embedding = get_embedding(concept)
    similarities = [cosine_similarity(concept_embedding, sublist) for sublist in embeddings]
    if len(similarities) == len(df):
        df[concept] = similarities
    return df

# Plots a scatter plot of the Greek Gods and their relative similarities to two different concepts
def scatplot(df, x_column, y_column, text_column, plot_title):
    fig = px.scatter(df, x=x_column, y=y_column, text=text_column)
    fig.update_traces(textposition='top center')
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        autosize=False,
        width=700,
        height=700
	)
    fig.update_xaxes(showgrid=False, showticklabels=False, ticks="", title_font=dict(size=18, color='red', family='Arial, sans-serif'))
    fig.update_yaxes(showgrid=False, showticklabels=False, ticks="", title_font=dict(size=18, color='red', family='Arial, sans-serif'))
    return fig

# Main function which is called when the user presses Go
def qualities(a,b,n,emd,df):
    compare_characteristic(a,emd,df)
    compare_characteristic(b,emd,df)
    fig = scatplot(df,a,b,n, '')
    return fig

# End of functions

# Load in dataframe of descriptions of Gods
df = pd.read_csv('GreekGodsTemplate.csv')

# Load in embeddings of descriptions of Greek Gods
emd = np.load('GG_embeddings.npy')

# Getting the embeddings for the dataframe
# emd = get_embeddings_from_dataframe(df, "Descriptions")

# Creates the boxes for the Streamlit app
st.title('Greek Gods')

# Initialize session state for the entry boxes if they don't already exist
if 'entry1' not in st.session_state:
    st.session_state['entry1'] = ''
if 'entry2' not in st.session_state:
    st.session_state['entry2'] = ''

# xconcept = st.text_input('Enter Concept X')
# yconcept = st.text_input('Enter Concept Y')

# Text input boxes
st.text_input("Enter Concept X", key='entry1')
st.text_input("Enter Concept Y", key='entry2')

if st.button('Go'):
	fig=qualities(st.session_state['entry1'], st.session_state['entry2'], 'Greek Gods', emd, df)
	st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
