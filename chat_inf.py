import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
import re
import gdown
import io
from time import sleep

# -- Session State Initialization --
if "infra_summary" not in st.session_state:
    st.session_state["infra_summary"] = ""
if "prog_summary" not in st.session_state:
    st.session_state["prog_summary"] = ""
if "final_summary" not in st.session_state:
    st.session_state["final_summary"] = ""
if "top_infra" not in st.session_state:
    st.session_state["top_infra"] = pd.DataFrame()
if "top_prog" not in st.session_state:
    st.session_state["top_prog"] = pd.DataFrame()

#Functions

def download_and_load_pickle(file_id, output_path):
    """Download and load a pickle file from Google Drive."""
    url = f"https://drive.google.com/uc?id={file_id}"
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
        "temperature": 0.5,
        "top_k": 20,
        "max_output_tokens": 4000,
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
- Enter your **Google API Key** (can be generated at "https://aistudio.google.com/apikey").
- Enter your **query**, e.g., "novel drug delivery device".  
- Specify the **focus** of the summary (optional), e.g., "the organisations undertaking the work", "do not talk about the researchers", or "group projects by their intended aim".  
- Specify the **number of closest matches** to consider, use more for broader topics.  
- Press **Start** to generate the summary.  
- **Export the results** as CSV or TXT files.  

**Please note:** 
- **Language models can make errors, only use these results as an overview and validate prior to sharing further.**
- Infrastructure supported projects only include those supported in **2022/23** and **2023/24**.
- Programme awards are only those available on **NIHR Open Data**.
- **Output csv files will simply show the results the summary has been generated using**, sorted by descending semantic similarity to the query, and may either show many **false positives** (selected number of results too high) or **miss relevant research** (selected number of results too high).


""")

# User inputs
api_key = st.text_input("🔑 Google API Key:", type="password")
query = st.text_input("📝 Query:")
focus_on = st.text_input("🎯 Focus of summary (optional, does not alter the query):")
run_option = st.selectbox("📂 Select dataset to query:", ["Infrastructure", "Programmes", "Both"])
top_n = st.number_input("🔢 Number of closest matching results to show the language model:", min_value=1, max_value=1000, value=150, step=10)
start_query = st.button("🚀 Start")

focus_text = f"\nFocus particularly on: {focus_on}." if focus_on else ""

# Query
if start_query and api_key and query:
    with st.spinner("✨ Generating summaries..."):
        query_embedding = generate_embedding(query, api_key)
        combined_summary = ""

        # Infrastructure
        if run_option in ["Infrastructure", "Both"]:
            infra_file_id = "1T8ZJaRFLrCaPCyE2P_QOUe5bCFhenT-W"
            df_infra = download_and_load_pickle(infra_file_id, "inf_emb.pkl")
            df_infra = clean_column_names(df_infra)
            df_infra = df_infra.fillna("")

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
**CONTEXT**
- Below, you have been provided a set of **projects** supported by National Institute for Health and Care (NIHR) infrastructure.
- You have been provided the project titles, research summaries, with the centre and year the project took place.
- There may be duplicated projects where they are supported across multiple years or multiple centres.


**TASK**
- Using only the projects listed below, provide an easy to read summary, grouping by common themes, of the research directly relevant to the query: "{query}"

**INSTRUCTIONS**
- Where possible, discuss work from all schemes - so BRCs, MICs, ARCs, PSTRCs and so on, these are determined by the 'centre name' which will have location and scheme in the name.
- Unless specified in the focus, limit response to 500 words. 
- If no projects are relevant to the query, say so.
- Try to advertise NIHR positively, linking between the sources to show how NIHR supports innovation across the translational pathway. 
- Link together centres and researchers when it is the same project where appropriate.
- Ensure that you write in a neutral scientific tone, being as accurate as possible, while still aiming to capture the audience's attention. 
- Only talk about the evidence you are presented with in the prompt, but make the links between projects whenever possible.
- Make sure the text is easy to take the main points away from. Where useful, use bold text, bullet points, and section headings. 
- Try and talk about aims and potential benefits of the research as much as possible.
- Talk around the projects and make links, I do not want just a list.
- Make sure you don't repeat yourself.
- If there are a lot of relevant projects, just talk about the most interesting and the common themes.

**Specific focus request from the user:**
{focus_text}


**Projects:**
{infra_deets}
            """

            infra_summary = generate_summary(infra_prompt, api_key)
            # Store in session_state
            st.session_state["infra_summary"] = infra_summary
            st.session_state["top_infra"] = top_infra

            combined_summary += f"**Infrastructure Projects Summary:**\n{infra_summary}\n\n"

        # Programmes
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
            top_prog = top_prog.drop_duplicates(subset=["Project_ID"])
            prog_deets = top_prog["text_for_prompt"].astype(str).str.cat(sep="\n____\n\n")

            prog_prompt = f"""
**CONTEXT**
- Below, you have been provided a set of programme awards funded by National Institute for Health and Care Research.
- You have been provided the project titles, research summaries, the researcher, contracted organisation, and the programme that funded the research.


**TASK**
- Using only the awards below, provide an easy to read summary, grouping by common themes, of the research directly relevant to the query: "{query}"

**INSTRUCTIONS**
- Unless specified in the focus, limit response to 500 words. 
- If no projects are relevant to the query, say so.
- Try to advertise NIHR positively, linking between the research to show how NIHR supports innovation across the translational pathway. 
- Talk about the programmes, researchers and organisations where appropriate. Make links between organisations and researchers where possible.
- Try and discuss work across different programmes where applicable.
- Ensure that you write in a neutral scientific tone, being as accurate as possible, while still aiming to capture the audience's attention. 
- Only talk about the evidence you are presented with in the prompt, but make the links between projects whenever possible.
- Make sure the text is easy to take the main points away from. Where useful, use bold text, bullet points, and section headings. 
- Try and talk about aims and potential benefits of the research as much as possible.
- Talk about UK based research first unless specified in the specific focus below.
- Talk around the projects and make links, I do not want just a list.
- Make sure you don't repeat yourself.
- If there are a lot of relevant projects, just talk about the most interesting and the common themes.


**Specific focus request from the user:**
{focus_text}


**Programme Awards**:
{prog_deets}
            """

            prog_summary = generate_summary(prog_prompt, api_key)
            # Store in session_state
            st.session_state["prog_summary"] = prog_summary
            st.session_state["top_prog"] = top_prog

            combined_summary += f"**Programme Awards Summary:**\n{prog_summary}\n\n"

        # Combined (only if "Both" selected)
        if run_option == "Both":
            combined_prompt = f"""
**INSTRUCTIONS**
- Below, you have been given summaries of NIHR programme awards and NIHR Infrastructure supported projects.
- I want you to provide an overview of the work relevant to the query: "{query}"
- Try to advertise NIHR positively, linking between the sources to show how NIHR supports innovation across the translational pathway. 
- Unless specified in the focus request, limit response to 500 words.
- Talk about research focused in the UK first, unless specified in the specific focus.
- Be as clear as possible, bullet points where sensible.
- Ensure that you write in a neutral scientific tone, being as precise as possible. Only talk about the evidence you are presented in the prompt, but make the links whenever possible.
- Don't just return a list.
- MAKE SURE YOU TALK ABOUT BOTH INFRASTRUCTURE SUPPORTED PROJCECTS AND PROGRAMME AWARDS, LINKING THE COMMON THEMES OF THE SUMMARIES UP. 
- This summary will be displayed with the combined summary, so no need to repeat everything, just make the key points. 

Specific focus request:
{focus_text}

Combined Summary:
{combined_summary}


            """

            final_summary = generate_summary(combined_prompt, api_key)
            st.session_state["final_summary"] = final_summary

elif start_query and (not api_key or not query):
    st.warning("⚠️ Please enter both your Google API Key and query.")

# --- DISPLAY RESULTS ---

# Show Infrastructure Summary if it exists
if st.session_state["infra_summary"]:
    st.subheader("📊 Infrastructure Summary")
    st.write(st.session_state["infra_summary"])

# Show Programme Summary if it exists
if st.session_state["prog_summary"]:
    st.subheader("📊 Programme Awards Summary")
    st.write(st.session_state["prog_summary"])

# Show Combined Summary if it exists
if st.session_state["final_summary"]:
    st.subheader("📝 Combined Summary")
    st.write(st.session_state["final_summary"])


# --- DOWNLOAD BUTTONS --- 
if not st.session_state["top_prog"].empty:
    st.download_button(
        label="📥 Download Programme Results as CSV",
        data=convert_df_to_csv(
            st.session_state["top_prog"].drop(
                columns=['embeddings', 'text_for_prompt'], 
                errors='ignore'
            )
        ),
        file_name=f"programme_results_{query.replace(' ', '_')}.csv",
        mime="text/csv",
    )

if not st.session_state["top_infra"].empty:
    st.download_button(
        label="📥 Download Infrastructure Results as CSV",
        data=convert_df_to_csv(
            st.session_state["top_infra"].drop(
                columns=['embeddings', 'text_for_prompt'], 
                errors='ignore'
            )
        ),
        file_name=f"infra_results_{query.replace(' ', '_')}.csv",
        mime="text/csv",
    )

if st.session_state["infra_summary"]:
    st.download_button(
        label="📥 Download Infrastructure Summary as TXT",
        data=convert_text_to_txt(st.session_state["infra_summary"]),
        file_name=f"infra_summary_{query.replace(' ', '_')}.txt",
        mime="text/plain",
    )

if st.session_state["prog_summary"]:
    st.download_button(
        label="📥 Download Programme Summary as TXT",
        data=convert_text_to_txt(st.session_state["prog_summary"]),
        file_name=f"programme_summary_{query.replace(' ', '_')}.txt",
        mime="text/plain",
    )

if st.session_state["final_summary"]:
    st.download_button(
        label="📥 Download Combined Summary as TXT",
        data=convert_text_to_txt(st.session_state["final_summary"]),
        file_name=f"combined_summary_{query.replace(' ', '_')}.txt",
        mime="text/plain",
    )
