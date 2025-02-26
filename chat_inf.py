import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
import re
import gdown
import io

#Functions

def download_and_load_pickle(file_id, output_path):
    """Download and load a pickle file from Google Drive."""
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
    return pd.read_pickle(output_path)


def clean_column_names(df):
    """Remove non-printable characters and strip spaces from column names."""
    df.columns = [re.sub(r'[^\x20-\x7E]', '', col).strip() for col in df.columns]
    return df


def generate_embedding(text, api_key):
    """Generate embedding using Gemini."""
    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="semantic_similarity",
    )
    return np.array(response['embedding']).reshape(1, -1)


def generate_summary(prompt, api_key):
    """Generate summary from Gemini given a prompt."""
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.1,
        "top_k": 5,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )

    response = model.generate_content(prompt)
    return response.text


def get_top_matches(df, query_embedding, top_n):
    """Get top N matches based on cosine similarity."""
    embeddings_matrix = np.vstack(df['embeddings'].values)
    similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
    df['similarity'] = similarities
    return df.sort_values(by='similarity', ascending=False).head(top_n)


def convert_df_to_csv(df):
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')


def convert_text_to_txt(text):
    """Convert text to TXT bytes for download."""
    return io.BytesIO(text.encode('utf-8'))



# Streamlit App


st.set_page_config(page_title="NIHR Project Query Tool", layout="wide")
st.title("🔍 NIHR Project Query Tool")

st.markdown("""
This tool allows you to query NIHR-supported projects and receive a summary of research relevant to your query.  
- Select **Infrastructure**, **Programmes**, or **Both**.  
- Enter your **Google API Key** (can be generated at "https://aistudio.google.com/apikey")  
- Enter your **query**, e.g., "novel drug delivery device"  
- Specify the **focus** of the summary (optional).  
- Specify the **number of closest matches** to consider.  
- Press **Start** to generate the summary.  
- **Export the results** as CSV or TXT files.  
""")

# User inputs
api_key = st.text_input("🔑 Google API Key:", type="password")
query = st.text_input("📝 Query:")
focus_on = st.text_input("🎯 Focus (optional):")
run_option = st.selectbox("📂 Select dataset to query:", ["Infrastructure", "Programmes", "Both"])
top_n = st.number_input("🔢 Number of closest matching results to consider:", min_value=1, max_value=1000, value=150, step=10)
start_query = st.button("🚀 Start")

focus_text = f"\nFocus particularly on: {focus_on}." if focus_on else ""

# Initialize variables for downloads
infra_summary, prog_summary, final_summary = "", "", ""
top_infra, top_prog = pd.DataFrame(), pd.DataFrame()

# ---------------------------- #
# 🧠 Query Logic
# ---------------------------- #

if start_query and api_key and query:
    with st.spinner("✨ Generating summaries..."):
        query_embedding = generate_embedding(query, api_key)
        combined_summary = ""

        if run_option in ["Infrastructure", "Both"]:
            infra_file_id = "1T8ZJaRFLrCaPCyE2P_QOUe5bCFhenT-W"
            df_infra = download_and_load_pickle(infra_file_id, "inf_emb.pkl")
            df_infra = clean_column_names(df_infra)

            df_infra['text_for_prompt'] = (
                "Project Identifier: " + df_infra['ID'] + "\n" +
                "Financial Year: " + df_infra['Financial Year'] + "\n" +
                "Project Title: " + df_infra['Study Title'] + "\n" +
                "Research Summary: " + df_infra['Project Summary'] + "\n" +
                "Centre: " + df_infra['Centre'] + "\n" +
                "Centre Theme: " + df_infra['Research Theme'] + "\n" +
                "Researcher: " + df_infra['PI Full Name']
            )

            top_infra = get_top_matches(df_infra, query_embedding, top_n=top_n)
            infra_deets = top_infra["text_for_prompt"].astype(str).str.cat(sep="\n____\n\n")

            infra_prompt = f"""
            You work for the Department of Health and Social Care in the United Kingdom in a communications role.
            Below are projects supported by the NIHR Infrastructure.
            Provide an overview of the work provided that is relevant to the query: "{query}".{focus_text}
            Don't include work that is not relevant to the queyr.
            Be precise, use a neutral scientific tone, and reference Project IDs.
            Where possible, highlight the benefit of the projects.

    

            Projects:
            {infra_deets}
            """

            infra_summary = generate_summary(infra_prompt, api_key)
            st.subheader("📊 Infrastructure Summary")
            st.write(infra_summary)
            combined_summary += f"**Infrastructure Projects Summary:**\n{infra_summary}\n\n"

            # Export Infrastructure results
            st.download_button(
                label="📥 Download Infrastructure Results as CSV",
                data=convert_df_to_csv(top_infra.drop(columns=['embeddings', 'text_for_prompt'], errors='ignore')),
                file_name=f"infra_results_{query.replace(' ', '_')}.csv",
                mime="text/csv",
            )

        if run_option in ["Programmes", "Both"]:
            prog_file_id = "1dmT7Hvwv0jOTvTUSgDCDUCLUsjw2ADoR"
            df_prog = download_and_load_pickle(prog_file_id, "prog_emb.pkl")
            df_prog = clean_column_names(df_prog)

            df_prog['text_for_prompt'] = (
                "Award ID: " + df_prog['Project_ID'] + "\n" +
                "Start-End Dates: " + df_prog['Start_date'] + " until " + df_prog['End_Date'] + "\n" +
                "Project Title: " + df_prog['Project_Title'] + "\n" +
                "Abstract: " + df_prog['Scientific_Abstract'] + "\n" +
                "Contracted Organisation: " + df_prog['Contracted_Organisation'] + "\n" +
                "NIHR Programme: " + df_prog["Programme"] + "\n" +
                "Researcher: " + df_prog["Award_Holder_Name"]
            )

            top_prog = get_top_matches(df_prog, query_embedding, top_n=top_n)
            prog_deets = top_prog["text_for_prompt"].astype(str).str.cat(sep="\n____\n\n")

            prog_prompt = f"""
            You work for the Department of Health and Social Care in a communications role.
            Below are programme awards supported by NIHR.
            Provide an overview of the work provided that is relevant to the query: "{query}".{focus_text}
            Don't include work that is not relevant to the queyr.
            Be precise, use a neutral scientific tone, and reference Project IDs.
            Where possible, highlight the benefit of the projects.

            Programme Awards:
            {prog_deets}
            """

            prog_summary = generate_summary(prog_prompt, api_key)
            st.subheader("📊 Programme Awards Summary")
            st.write(prog_summary)
            combined_summary += f"**Programme Awards Summary:**\n{prog_summary}\n\n"

            # Export Programme results
            st.download_button(
                label="📥 Download Programme Results as CSV",
                data=convert_df_to_csv(top_prog.drop(columns=['embeddings', 'text_for_prompt'], errors='ignore')),
                file_name=f"programme_results_{query.replace(' ', '_')}.csv",
                mime="text/csv",
            )

        if run_option == "Both":
            combined_prompt = f"""
            You work for the Department of Health and Social Care in a communications role.
            You have been provided two separate summaries of NIHR-supported research:
            {combined_summary}

            Please provide a combined, cohesive overview highlighting how NIHR infrastructure and programme awards support research related to the query: "{query}".{focus_text}
            Link projects where appropriate, mention researchers and centres, and reference IDs.
            """

            final_summary = generate_summary(combined_prompt, api_key)
            st.subheader("📝 Combined Summary")
            st.write(final_summary)

            # Export combined summary as TXT
            st.download_button(
                label="📥 Download Combined Summary as TXT",
                data=convert_text_to_txt(final_summary),
                file_name=f"combined_summary_{query.replace(' ', '_')}.txt",
                mime="text/plain",
            )

elif start_query and (not api_key or not query):
    st.warning("⚠️ Please enter both your Google API Key and query.")
