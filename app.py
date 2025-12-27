"""
================================================================================
STREAMLIT CHATBOT INTERFACE FOR INDECIMAL RAG ASSISTANT
================================================================================

This module provides the web-based user interface for the RAG (Retrieval-Augmented
Generation) chatbot. It uses Streamlit framework to create an interactive chat
experience where users can ask questions about Indecimal's construction services.

Key Features:
- Clean, minimal design with Inter font
- Sidebar with system statistics and example questions
- Scrollable chat container with fixed input
- Retrieved context display with relevance scores
- Logo integration and favicon

Dependencies:
- streamlit: Web UI framework
- rag_pipeline: Custom RAG implementation module
- time: For measuring query response time
- re: For regex-based markdown cleaning
- base64: For encoding logo image
- pathlib: For file path handling
"""

# ================================================================================
# IMPORTS
# ================================================================================
import streamlit as st          # Streamlit framework for web UI
from rag_pipeline import RAGPipeline  # Custom RAG pipeline implementation
import time                      # For timing query execution
import re                        # For regex-based text processing
import base64                    # For encoding images to base64
from pathlib import Path         # For cross-platform file path handling


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def get_logo_base64():
    """
    Load the company logo and encode it as base64 for embedding in HTML.
    
    This function reads the logo image file from the same directory as this script
    and converts it to a base64 string that can be embedded directly in HTML img tags.
    This avoids needing to serve the image separately.
    
    Returns:
        str: Base64 encoded string of the logo image, or None if file doesn't exist
    """
    # Construct path to logo file relative to this script's location
    logo_path = Path(__file__).parent / "indecimallogo.jpg"
    
    # Check if logo file exists before attempting to read
    if logo_path.exists():
        # Open file in binary mode and encode to base64
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def clean_markdown_text(text, for_html=False):
    """
    Convert markdown-formatted text to clean, readable plain text.
    
    This function removes markdown syntax (headers, bold, italic, bullets) to
    display retrieved document chunks in a cleaner format in the UI.
    
    Args:
        text (str): Raw text that may contain markdown formatting
        for_html (bool): If True, convert newlines to <br> tags for HTML display
    
    Returns:
        str: Cleaned text with markdown syntax removed
    
    Example:
        Input:  "## Header\n**Bold text** with *italic*"
        Output: "Header\nBold text with italic"
    """
    # Remove markdown headers (# through ######)
    # Pattern: Start of line, 1-6 hash symbols, whitespace
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Convert bold **text** to just text
    # Pattern: **anything** → anything
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    
    # Convert italic *text* to just text
    # Pattern: *anything* → anything
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Convert markdown bullet points to Unicode bullets
    # Pattern: Start of line, dash, space → bullet point
    text = re.sub(r'^-\s+', '• ', text, flags=re.MULTILINE)
    
    # Clean up excessive whitespace (3+ newlines → 2 newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Optionally convert newlines to HTML breaks for web display
    if for_html:
        text = text.replace('\n', '<br>')
    
    return text


# ================================================================================
# STREAMLIT PAGE CONFIGURATION
# ================================================================================

# Load logo for use in header (cache at module level)
logo_b64 = get_logo_base64()

# Configure Streamlit page settings
# This must be called before any other Streamlit commands
st.set_page_config(
    page_title="Indecimal AI Assistant",           # Browser tab title
    page_icon="D:/Indecimel-RAG/indecimallogo.jpg", # Favicon
    layout="wide",                                  # Use full page width
    initial_sidebar_state="expanded"                # Show sidebar by default
)


# ================================================================================
# CUSTOM CSS STYLING
# ================================================================================

# Inject custom CSS for styling the application
# Uses CSS custom properties (variables) for theme compatibility
st.markdown("""
<style>
    /* ======================== TYPOGRAPHY ======================== */
    /* Import Google Fonts - Inter for modern, clean look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    /* Apply Inter font to all elements */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* ======================== LAYOUT ======================== */
    /* Main content container - constrain width and add padding */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1100px;
    }
    
    /* ======================== HEADER COMPONENT ======================== */
    /* Application header with logo and title */
    .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.5rem 2rem;
        background: var(--background-color);  /* Theme-aware background */
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 2rem;
    }
    
    /* Logo image size */
    .app-header img {
        height: 48px;
        width: auto;
    }
    
    /* Header title styling */
    .app-header h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Header subtitle styling */
    .app-header p {
        margin: 0.25rem 0 0 0;
        opacity: 0.7;
        font-size: 0.9rem;
    }
    
    /* ======================== SIDEBAR STATISTICS ======================== */
    /* Container for stat cards in sidebar */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Individual stat card */
    .stat-item {
        flex: 1;
        background: rgba(128,128,128,0.1);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Large number in stat card */
    .stat-value {
        font-size: 1.75rem;
        font-weight: 600;
    }
    
    /* Label below stat number */
    .stat-label {
        font-size: 0.75rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    /* ======================== CONTEXT CARDS ======================== */
    /* Container for retrieved document chunks */
    .context-card {
        background: rgba(128,128,128,0.05);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    
    /* Header row of context card (source + score) */
    .context-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(128,128,128,0.15);
    }
    
    /* Source file badge */
    .source-label {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(128,128,128,0.15);
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Section name label */
    .section-label {
        opacity: 0.7;
        font-size: 0.85rem;
        margin-left: 0.5rem;
    }
    
    /* Relevance score badge (green) */
    .match-score {
        background: #4caf50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Text content of context card */
    .context-text {
        font-size: 0.9rem;
        line-height: 1.7;
        opacity: 0.9;
    }
    
    /* ======================== SECTION HEADERS ======================== */
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.75rem 0;
    }
    
    /* ======================== HIDE STREAMLIT DEFAULTS ======================== */
    /* Hide the Streamlit hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ================================================================================
# SVG ICON DEFINITIONS
# ================================================================================

# Dictionary of inline SVG icons used throughout the UI
# Using inline SVG allows icons to inherit text color and scale properly
ICONS = {
    # Search/magnifying glass icon
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path></svg>',
    
    # File/document icon
    "file": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>',
    
    # Database/storage icon
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>',
    
    # CPU/processor icon
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>',
    
    # Right arrow/chevron icon
    "arrow": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>',
    
    # Lightning bolt/zap icon
    "zap": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>',
}


# ================================================================================
# CACHED RESOURCE LOADERS
# ================================================================================

@st.cache_resource  # Cache the pipeline to avoid reloading on every interaction
def load_rag_pipeline():
    """
    Load and cache the RAG pipeline instance.
    
    This function is decorated with @st.cache_resource to ensure the pipeline
    (including embeddings model and FAISS index) is only loaded once and reused
    across all user sessions. This significantly improves performance.
    
    Returns:
        RAGPipeline: Initialized RAG pipeline ready for queries
    """
    return RAGPipeline()


# ================================================================================
# UI COMPONENT RENDERERS
# ================================================================================

def render_header():
    """
    Render the application header with logo and title.
    
    Creates an HTML header component that displays:
    - Company logo (if available)
    - Application name
    - Tagline/description
    
    The header uses custom CSS classes defined above for styling.
    """
    # Build logo HTML only if logo was successfully loaded
    logo_html = ""
    if logo_b64:
        # Embed logo as base64 data URI (no external file needed)
        logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" alt="Indecimal">'
    
    # Render header using custom HTML with CSS classes
    st.markdown(f"""
    <div class="app-header">
        {logo_html}
        <div>
            <h1>Indecimal AI Assistant</h1>
            <p>Construction marketplace knowledge companion</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_context_card(chunk, index):
    """
    Render a styled card displaying a retrieved document chunk.
    
    Creates an HTML card component showing:
    - Source file name with icon
    - Section name within the document
    - Relevance/similarity score as percentage
    - Cleaned text content of the chunk
    
    Args:
        chunk (dict): Dictionary containing chunk data with keys:
            - 'source': Source filename (e.g., 'doc1.md')
            - 'section': Section name within document
            - 'text': Raw text content
            - 'similarity_score': Float between 0 and 1
        index (int): Index number for the chunk (used for display)
    
    Returns:
        str: HTML string for the context card
    """
    # Convert similarity score (0-1) to percentage string
    score_pct = f"{chunk['similarity_score']:.0%}"
    
    # Clean the markdown and convert to HTML (with <br> for newlines)
    clean_text = clean_markdown_text(chunk['text'], for_html=True)
    
    # Truncate very long chunks to prevent UI overflow
    raw_text = clean_markdown_text(chunk['text'], for_html=False)
    if len(raw_text) > 600:
        # Re-process with truncation and add ellipsis
        truncated = raw_text[:600] + "..."
        clean_text = truncated.replace('\n', '<br>')
    
    # Return HTML card using CSS classes defined above
    return f"""
    <div class="context-card">
        <div class="context-header">
            <div>
                <span class="source-label">{ICONS['file']} {chunk['source']}</span>
                <span class="section-label">{chunk['section']}</span>
            </div>
            <span class="match-score">{score_pct} match</span>
        </div>
        <div class="context-text">{clean_text}</div>
    </div>
    """


# ================================================================================
# MAIN APPLICATION FUNCTION
# ================================================================================

def main():
    """
    Main application entry point.
    
    This function orchestrates the entire Streamlit application:
    1. Renders the header
    2. Sets up the sidebar with info and example questions
    3. Manages chat session state
    4. Handles user input (both chat and sidebar buttons)
    5. Processes queries through the RAG pipeline
    6. Displays responses and retrieved context
    """
    
    # ======================== HEADER ========================
    render_header()
    
    # ======================== SIDEBAR ========================
    with st.sidebar:
        # Explain how the RAG system works
        st.markdown("## How It Works")
        st.markdown("""
        1. **Embed** - Question becomes a vector
        2. **Search** - Find similar document chunks  
        3. **Generate** - AI answers from context only
        """)
        
        st.markdown("---")  # Visual separator
        
        try:
            # Load RAG pipeline (cached after first load)
            with st.spinner("Loading models..."):
                rag = load_rag_pipeline()
            
            # Get system statistics for display
            stats = rag.get_stats()
            
            # -------------------- Knowledge Base Stats --------------------
            st.markdown("## Knowledge Base")
            
            # Render stat cards using custom HTML
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-item">
                    <div class="stat-value">{len(stats['documents_indexed'])}</div>
                    <div class="stat-label">Documents</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['total_chunks']}</div>
                    <div class="stat-label">Chunks</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # -------------------- Model Information --------------------
            st.markdown("## Models")
            st.caption(f"Embeddings: {stats['embedding_model']}")
            st.caption(f"LLM: {stats['llm_model']}")
            st.caption(f"Backend: {stats.get('llm_backend', 'Unknown')}")
            
            st.markdown("---")
            
            # -------------------- Example Questions --------------------
            st.markdown("## Try These")
            
            # List of example questions for users to try
            examples = [
                "What is the Premier package price?",
                "How do you handle delays?",
                "What quality checks are done?",
                "Explain the customer journey",
                "What maintenance is included?"
            ]
            
            # Create a button for each example question
            for q in examples:
                if st.button(q, key=q, use_container_width=True):
                    # Store query in session state for processing
                    st.session_state.pending_query = q
                    
        except Exception as e:
            # Handle errors (e.g., Ollama not running)
            st.error(f"Error: {str(e)}")
            st.info("Make sure Ollama is running.")
            return  # Exit if pipeline fails to load
    
    # ======================== SESSION STATE INITIALIZATION ========================
    # Initialize session state variables if they don't exist
    # Session state persists across reruns within the same browser session
    
    if "messages" not in st.session_state:
        # List to store chat history: [{"role": "user/assistant", "content": "..."}]
        st.session_state.messages = []
    
    if "last_result" not in st.session_state:
        # Store the last RAG query result for context display
        st.session_state.last_result = None
    
    # ======================== QUERY INPUT HANDLING ========================
    # Check for pending query from sidebar button
    query = None
    if "pending_query" in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query  # Clear after reading
    
    # ======================== SCROLLABLE MESSAGE CONTAINER ========================
    # Create a fixed-height container for messages
    # This keeps the chat input visible by preventing the message area from
    # pushing it off-screen when there are many messages
    messages_container = st.container(height=500)
    
    with messages_container:
        # -------------------- Display Chat History --------------------
        # Iterate through all stored messages and display them
        for msg in st.session_state.messages:
            # st.chat_message creates styled message bubbles
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # -------------------- Display Retrieved Context --------------------
        # Show the context chunks that were used for the last answer
        if st.session_state.last_result:
            st.markdown("---")
            
            # Section header with database icon
            st.markdown(
                f'<div class="section-title">{ICONS["database"]} Retrieved Context</div>',
                unsafe_allow_html=True
            )
            st.caption("Document chunks used to generate the answer:")
            
            # Render a card for each retrieved chunk
            for i, chunk in enumerate(st.session_state.last_result["retrieved_chunks"]):
                st.markdown(render_context_card(chunk, i + 1), unsafe_allow_html=True)
            
            # Collapsible section for raw JSON response (for debugging)
            with st.expander("View Raw Response"):
                st.json(st.session_state.last_result)
    
    # ======================== CHAT INPUT ========================
    # Place chat input AFTER the scrollable container so it stays visible
    user_input = st.chat_input("Ask about Indecimal's construction services...")
    
    # Combine sidebar query and chat input (sidebar takes priority)
    if query is None:
        query = user_input
    
    # ======================== QUERY PROCESSING ========================
    # Process the query if one was submitted
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Execute RAG query with timing
        with st.spinner("Searching..."):
            start = time.time()  # Start timer
            result = rag.query(query)  # Call RAG pipeline
            elapsed = time.time() - start  # Calculate duration
        
        # Store result for context display
        st.session_state.last_result = result
        
        # Format answer with timing info
        answer = f"{result['generated_answer']}\n\n*Generated in {elapsed:.1f}s*"
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
        
        # Rerun the app to display the new messages
        # This is necessary because Streamlit runs top-to-bottom
        st.rerun()


# ================================================================================
# APPLICATION ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    # Run the main application when script is executed directly
    main()
