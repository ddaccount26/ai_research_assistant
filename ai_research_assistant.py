import streamlit as st
import arxiv
import os
import shutil
import re
import pandas as pd
from tqdm import tqdm
from datetime import datetime

#RAG-specific imports
import os
from dotenv import load_dotenv
from openai import OpenAI
import fitz
import tiktoken
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


tokenizer = None
embedder = None

# Load environment variables
def get_secret(key: str, default=None):
    # Works on Community Cloud (st.secrets) and locally (secrets.toml).
    # Optional fallback to environment variables for flexibility.
    return st.secrets.get(key, os.getenv(key, default))

# Initialize LLM client
api_key = get_secret("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY. Add it to Streamlit Secrets (Cloud) or .streamlit/secrets.toml (local).")
    st.stop()

llm_client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

def init_rag_models():
    global tokenizer, embedder
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================================
# RAG PIPELINE FUNCTIONS (from rag_pipeline.py)
# ============================================================================

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    pages = []
    for page in doc:
        text = page.get_text()
        text = re.sub(r"\s+", " ", text)
        pages.append(text)
    return "\n".join(pages)


def chunk_text(text, max_tokens=500, overlap=100):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap

    return chunks


def build_chunks(pdf_dir, metadata_csv):
    meta = pd.read_csv(metadata_csv)
    rows = []

    for _, paper in meta.iterrows():
        pdf_path = os.path.join(pdf_dir, paper["title"].replace(" ", "_")[:150] + ".pdf")
        if not os.path.exists(pdf_path):
            continue

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        for chunk in chunks:
            rows.append({
                "text": chunk,
                "title": paper["title"],
                "year": paper["published"][:4],
                "paper_id": paper["arxiv_id"]
            })

    return pd.DataFrame(rows)


def embed_documents(texts, batch_size=32):
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


def embed_query(query):
    return embedder.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def retrieve(query, index, chunks_df, k=5):
    q_emb = embed_query(query)
    scores, indices = index.search(q_emb, k)
    return chunks_df.iloc[indices[0]]


def build_messages(query, retrieved_chunks):
    context = ""
    for _, row in retrieved_chunks.iterrows():
        context += f"""
SOURCE
Paper: {row['title']}
Year: {row['year']}
Content:
{row['text']}
"""

    return [
        {
            "role": "system",
            "content": (
                "You are a research assistant.\n"
                "Use ONLY the provided sources.\n"
                "Do NOT use prior knowledge.\n"
                "If the answer is missing, say: "
                "'Not found in the retrieved papers.'\n"
                "Cite sources."
            )
        },
        {
            "role": "user",
            "content": f"""
SOURCES:
{context}

QUESTION:
{query}

FORMAT:
Answer:

Sources:
- Paper Title
"""
        }
    ]


def generate_answer(messages):
    response = llm_client.chat.completions.create(
        model="gemini-2.5-flash-lite",
        messages=messages,
        temperature=0,
        max_tokens=400
    )
    return response.choices[0].message.content


def initialize_rag_system(pdf_dir="data/pdfs", metadata_csv="data/metadata.csv"):
    """Initialize or load the RAG system (chunks, embeddings, and FAISS index)."""
    init_rag_models()
    chunks_path = "data/chunks.pkl"
    index_path = "data/faiss.index"
    
    # Check if we have saved chunks and index
    if os.path.exists(chunks_path) and os.path.exists(index_path):
        try:
            chunks_df = pd.read_pickle(chunks_path)
            index = faiss.read_index(index_path)
            return chunks_df, index, True
        except Exception as e:
            st.warning(f"Error loading saved RAG data: {e}. Rebuilding...")
    
    # Check if we have PDFs and metadata to build from
    if not os.path.exists(metadata_csv):
        return None, None, False
    
    # Build the RAG system
    try:
        with st.spinner("Building RAG system (this may take a few minutes)..."):
            # Build chunks
            chunks_df = build_chunks(pdf_dir, metadata_csv)
            
            if len(chunks_df) == 0:
                st.error("No chunks created. Please ensure PDFs are downloaded.")
                return None, None, False
            
            # Embed all chunks
            embeddings = embed_documents(chunks_df["text"].tolist())
            
            # Build FAISS index
            index = build_faiss_index(embeddings)
            
            # Save for future use
            os.makedirs("data", exist_ok=True)
            faiss.write_index(index, index_path)
            chunks_df.to_pickle(chunks_path)
            
            st.success("RAG system initialized successfully!")
            return chunks_df, index, True
    except Exception as e:
        st.error(f"Error building RAG system: {e}")
        return None, None, False


def query_rag_system(query, index, chunks_df):
    """Query the RAG system and get an answer."""
    try:
        retrieved = retrieve(query, index, chunks_df)
        messages = build_messages(query, retrieved)
        answer = generate_answer(messages)
        return answer
    except Exception as e:
        return f"Error querying RAG system: {e}"

# ============================================================================
# CONFIGURATION: Categories and Keywords
# ============================================================================
ALLOWED_CATEGORIES = {
    "cs.CL",    # Computation & Language (LLMs)
    "cs.LG",    # Machine Learning
    "cs.AI",    # Artificial Intelligence
    "cs.CV",    # Computer Vision (multimodal work)
    "stat.ML"   # Statistical Machine Learning
}

KEYWORD_CATEGORIES = {
    "Language Models & NLP": [
        "large language model",
        "language model",
        "llm",
        "transformer",
        "bert",
        "gpt",
        "attention mechanism",
        "sequence-to-sequence",
        "natural language processing",
        "nlp"
    ],
    "Specialized Architectures": [
        "state space",
        "state-space",
        "ssm",
        "mamba",
        "selective state space",
        "world model",
        "v-jepa",
        "jepa",
        "vision transformer",
        "vit"
    ],
    "Learning Methods": [
        "reinforcement learning",
        "self-supervised learning",
        "self-supervised",
        "unsupervised learning",
        "supervised learning",
        "transfer learning",
        "curriculum learning",
        "meta-learning"
    ],
    "Multimodal & Foundation Models": [
        "multimodal",
        "foundation model",
        "pretraining",
        "prompt learning",
        "in-context learning",
        "few-shot learning",
        "zero-shot learning",
        "vision-language"
    ],
    "Deep Learning & Optimization": [
        "deep learning",
        "neural network",
        "convolutional neural network",
        "recurrent neural network",
        "gradient descent",
        "optimization",
        "backpropagation",
        "normalization"
    ],
    "Probabilistic & Bayesian": [
        "bayesian",
        "probabilistic model",
        "graphical model",
        "variational inference",
        "monte carlo",
        "generative model",
        "gaussian process",
        "expectation maximization"
    ]
}

# Map human-friendly category names to arXiv category codes
CATEGORY_TO_ARXIV = {
    "Language Models & NLP": ["cs.CL", "cs.LG", "cs.AI"],
    "Specialized Architectures": ["cs.LG", "cs.CV", "cs.AI"],
    "Learning Methods": ["cs.LG", "stat.ML", "cs.AI"],
    "Multimodal & Foundation Models": ["cs.CV", "cs.AI", "cs.CL", "cs.LG"],
    "Deep Learning & Optimization": ["cs.LG", "cs.AI", "cs.CV", "stat.ML"],
    "Probabilistic & Bayesian": ["stat.ML", "cs.LG", "cs.AI"],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def has_allowed_category(categories):
    """Check if any paper category is in the whitelist."""
    return any(cat in ALLOWED_CATEGORIES for cat in categories)

def contains_keyword(text, keywords):
    """Check if any keyword appears in text (case-insensitive)."""
    if not text:
        return False
    text = text.lower()
    return any(keyword in text for keyword in keywords)

def score_paper(paper, custom_keywords, category_keywords):
    """
    Score a paper based on keyword matches with weighted emphasis on custom keywords.
    Custom keywords: 50% weight
    Category keywords: 50% weight
    Returns a score between 0 and 1.
    """
    combined_text = f"{paper.title} {paper.summary}".lower()
    
    custom_matches = 0
    for kw in custom_keywords:
        if kw.lower() in combined_text:
            custom_matches += 1
    
    category_matches = 0
    for kw in category_keywords:
        if kw.lower() in combined_text:
            category_matches += 1
    
    # Calculate weighted score
    # If no keywords of a type exist, that component is 0
    custom_score = (custom_matches / len(custom_keywords)) if custom_keywords else 0
    category_score = (category_matches / len(category_keywords)) if category_keywords else 0
    
    # Custom keywords get 50% weight, category keywords get 50% weight
    total_score = (custom_score * 0.5) + (category_score * 0.5)
    
    return total_score

def is_recent(published_date, start_year, end_year):
    """Check if paper is published between start_year and end_year."""
    return start_year <= published_date.year <= end_year

def safe_filename(text):
    """Return a filesystem-safe filename fragment."""
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "_")[:150]

def download_pdfs(papers, output_dir="data/pdfs"):
    """Download PDFs for each paper."""
    # Remove existing PDF directory to ensure a clean overwrite on each run
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass
    
    os.makedirs(output_dir, exist_ok=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, paper in enumerate(papers):
        title_slug = safe_filename(paper.title)
        pdf_path = os.path.join(output_dir, f"{title_slug}.pdf")
        
        try:
            # Always (re)download to ensure latest content and overwrite
            paper.download_pdf(dirpath=output_dir, filename=f"{title_slug}.pdf")
            progress_bar.progress((idx + 1) / max(1, len(papers)))
            status_text.text(f"Downloaded: {idx + 1}/{len(papers)} PDFs")
        except Exception:
            status_text.text(f"Warning: Failed to download {paper.title[:50]}...")
    
    progress_bar.empty()
    status_text.empty()

def save_metadata(papers, path="data/metadata.csv"):
    """Save paper metadata to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for paper in papers:
        arxiv_id = paper.entry_id.split("/")[-1]
        authors = ", ".join(a.name for a in paper.authors)
        published = paper.published.strftime("%Y-%m-%d")
        rows.append({
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "authors": authors,
            "published": published,
            "categories": ", ".join(paper.categories),
            "summary": paper.summary,
            "pdf_url": paper.pdf_url
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

def fetch_relevant_papers(selected_keywords, custom_keywords, max_results, start_year, end_year, selected_category_names=None):
    """Fetch and filter papers from arXiv with weighted keyword scoring.
    Server-side query is constructed from user-selected categories.
    """
    # Build set of arXiv codes from selected human-readable categories
    codes = set()
    if selected_category_names:
        for name in selected_category_names:
            codes.update(CATEGORY_TO_ARXIV.get(name, []))
    
    # Fallback to allowed categories if none selected / mapping empty
    if not codes:
        codes = set(ALLOWED_CATEGORIES)
    
    # ---- NEW: build category part of query ----
    cat_query = " OR ".join(f"cat:{c}" for c in sorted(codes))
    
    # ---- NEW: build keyword part of query from custom keywords ----
    keyword_parts = []
    for kw in custom_keywords:
        kw = kw.strip()
        if kw:
            # search in title/abstract/etc.
            keyword_parts.append(f'all:"{kw}"')
    
    if keyword_parts:
        # (categories) AND (kw1 OR kw2 OR ...)
        keyword_query = " OR ".join(keyword_parts)
        query = f"({cat_query}) AND ({keyword_query})"
    else:
        # no custom keywords → fall back to category-only search
        query = cat_query
    
    # ---- run arXiv search ----
    search = arxiv.Search(
        query=query,
        max_results=max_results * 5,  # Over-fetch to ensure we get enough after scoring
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    
    # Materialize results so we can report how many were fetched before filtering
    entries = list(search.results())
    try:
        st.info(f"Fetched {len(entries)} papers from arXiv (before filtering)")
    except Exception:
        pass
    
    papers_with_scores = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, paper in enumerate(entries):
        if not has_allowed_category(paper.categories):
            continue
        
        if not is_recent(paper.published, start_year, end_year):
            continue
        
        # Score the paper based on custom and category keywords
        score = score_paper(paper, custom_keywords, selected_keywords)
        
        # Keep papers with at least some keyword match or custom keyword presence
        if score > 0 or (custom_keywords and any(kw.lower() in f"{paper.title} {paper.summary}".lower() for kw in custom_keywords)):
            papers_with_scores.append((paper, score))
        
        # Update progress
        try:
            progress_bar.progress(min((idx + 1) / max(1, len(entries)), 1.0))
        except Exception:
            pass
        status_text.text(f"Filtering papers... Found {len(papers_with_scores)} candidates")
    
    # Sort by score (descending) and take top max_results
    papers_with_scores.sort(key=lambda x: x[1], reverse=True)
    filtered_papers = papers_with_scores[:max_results]
    
    progress_bar.empty()
    status_text.empty()
    
    return filtered_papers

# ============================================================================
# STREAMLIT APP LAYOUT
# ============================================================================
st.set_page_config(page_title="arXiv RAG Dataset Builder", layout="wide")
st.title("arXiv RAG Dataset Builder")
st.divider()

# Create single row with 3 columns for all inputs
col1, col2, col3 = st.columns([1.2, 1.2, 1.2])

with col1:
    st.markdown("**Select Categories**")
    selected_categories = st.multiselect(
        "Research categories:",
        options=list(KEYWORD_CATEGORIES.keys()),
        default=[],
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**Custom Keywords**")
    custom_keywords_text = st.text_input(
        "Additional phrases:",
        placeholder="e.g., graph neural network, knowledge distillation",
        label_visibility="collapsed"
    )

with col3:
    st.markdown("**Year Range**")
    col3a, col3b = st.columns(2)
    with col3a:
        start_year = st.number_input(
            "From:",
            min_value=2015,
            max_value=2026,
            value=2019,
            label_visibility="collapsed"
        )
    with col3b:
        end_year = st.number_input(
            "To:",
            min_value=2015,
            max_value=2026,
            value=2026,
            label_visibility="collapsed"
        )

st.divider()

# Configuration in second row - 3 columns with empty middle
col_papers, col_empty, col_button = st.columns([1, 1.5, 0.8])

with col_papers:
    max_papers = st.slider(
        "Number of papers:",
        min_value=5,
        max_value=100,
        value=30,
        step=5
    )

with col_empty:
    pass  # Empty space

with col_button:
    build_button = st.button(
        "Build Dataset",
        type="primary",
        use_container_width=True,
        key="build_btn"
    )

st.divider()

# Set default output directory
output_dir = "data"

# Parse custom keywords
custom_keywords = []
if custom_keywords_text.strip():
    for phrase in custom_keywords_text.replace("\n", ",").split(","):
        cleaned = phrase.strip().lower()
        if cleaned:
            custom_keywords.append(cleaned)

# Build selected keywords set from selected categories (separate from custom)
selected_keywords_set = set()
# Add keywords from all selected categories
for category in selected_categories:
    if category in KEYWORD_CATEGORIES:
        for keyword in KEYWORD_CATEGORIES[category]:
            selected_keywords_set.add(keyword)

selected_keywords_list = sorted(list(selected_keywords_set))

# ============================================================================
# EXECUTION LOGIC
# ============================================================================
if build_button:
    if not selected_keywords_list and not custom_keywords:
        st.error("Please select at least one research category or add custom keywords.")
    else:
        st.markdown("### Building Dataset...")
        
        # Step 1: Fetch papers with weighted scoring
        st.markdown("**Step 1: Fetching papers from arXiv...**")
        try:
            papers = fetch_relevant_papers(
                selected_keywords=selected_keywords_list,
                custom_keywords=custom_keywords,
                max_results=max_papers,
                start_year=int(start_year),
                end_year=int(end_year),
                selected_category_names=selected_categories
            )
            
            if papers:
                # Extract papers and scores
                papers_list = [paper for paper, score in papers]
                scores_list = [score for paper, score in papers]
                st.success(f"Found {len(papers_list)} relevant papers")
                
                # Display top 3 matches
                if papers:
                    st.markdown("### Top 3 Matching Papers")
                    for rank, (paper, score) in enumerate(papers[:3], 1):
                        with st.container():
                            st.markdown(f"**Title:** {paper.title}")
                            st.markdown(f"**Authors:** {', '.join([a.name for a in paper.authors[:3]])}")
                            st.markdown(f"**Published:** {paper.published.strftime('%Y-%m-%d')}")
                            st.markdown(f"**Summary:** {paper.summary[:500]}...")
                            st.divider()
                
                # Step 2: Download PDFs
                st.markdown("**Step 2: Downloading PDFs...**")
                pdf_output_dir = os.path.join(output_dir, "pdfs")
                download_pdfs(papers_list, output_dir=pdf_output_dir)
                st.success(f"PDFs saved to {os.path.abspath(pdf_output_dir)}")
                
                # Step 3: Save metadata
                st.markdown("**Step 3: Saving metadata...**")
                metadata_path = os.path.join(output_dir, "metadata.csv")
                save_metadata(papers_list, path=metadata_path)
                st.success(f"Metadata saved to {os.path.abspath(metadata_path)}")
                
                # Step 4: Initialize RAG system
                st.markdown("**Step 4: Building RAG system...**")
                init_rag_models()

                chunks_df, index, success = initialize_rag_system(
                    pdf_dir=pdf_output_dir,
                    metadata_csv=metadata_path
                )
                
                if success:
                    # Store in session state
                    st.session_state['rag_initialized'] = True
                    st.session_state['chunks_df'] = chunks_df
                    st.session_state['index'] = index
                    st.success("RAG system ready for querying!")
                
                st.markdown("### Build Complete")
                st.info("Ready for RAG querying: Chat with your dataset below!")
                
            else:
                st.warning("No papers found matching your criteria. Try adjusting keywords or year range.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.divider()

## Chat with Dataset (moved to bottom)
st.markdown("### Chat with Dataset")

# Initialize RAG system if not already done
if 'rag_initialized' not in st.session_state:
    # Try to load existing RAG data
    chunks_df, index, success = initialize_rag_system()
    if success:
        st.session_state['rag_initialized'] = True
        st.session_state['chunks_df'] = chunks_df
        st.session_state['index'] = index
    else:
        st.session_state['rag_initialized'] = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history in a container
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Larger chat input area + send button
col_text, col_send = st.columns([10, 1])

with col_text:
    chat_input = st.text_area("Your question", height=160, placeholder="Ask a question about the papers...")

with col_send:
    send = st.button("Send")

if send and chat_input and chat_input.strip():
    user_msg = chat_input.strip()
    
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)
    
    # Check if RAG system is initialized
    if st.session_state.get('rag_initialized', False):
        init_rag_models()
        with st.spinner("Searching papers and generating answer..."):
            # Query RAG system
            assistant_reply = query_rag_system(
                user_msg,
                st.session_state['index'],
                st.session_state['chunks_df']
            )
    else:
        assistant_reply = "RAG system not initialized. Please build your dataset first by selecting categories and clicking 'Build Dataset'."
    
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.write(assistant_reply)
    
    # Force rerun to clear input
    st.rerun()

st.divider()
st.markdown("Built for research students | arXiv RAG Pipeline")
