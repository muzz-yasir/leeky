import streamlit as st
import streamlit as st
from rapidfuzz import fuzz
from typing import Dict, Tuple, List, Any, Optional, Union
import difflib
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).resolve().parent
src_path = project_root / 'src'
if src_path.exists():
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
else:
    st.error("Could not find src directory. Make sure the project structure is correct.")
    st.stop()

try:
    from leeky.core.types import TextSource, TemplateType, PromptTemplate
    from leeky.core.prompt_manager import PromptManager
    from leeky.core.decop_ui import render_decop_ui
    from leeky.core.scraper import ArticleScraper
    from leeky.core.engines.base_engine import CompletionEngine
except ImportError as e:
    st.error(f"Failed to import leeky package: {str(e)}")
    st.stop()

class SyncOpenAIEngine(CompletionEngine):
    """Synchronous OpenAI engine implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """Initialize the OpenAI engine."""
        import openai
        openai.api_key = api_key
        self.client = openai.Client()
        self._model = model
        self._max_tokens = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-32k": 32768
        }.get(model, 4096)

    def _format_prompt_with_length_guidance(self, prompt: str, max_tokens: int) -> str:
        """Add length guidance to the prompt."""
        # Convert from tokens to approximate word count for more natural guidance
        approx_words = int(max_tokens / 1.3)  # Convert tokens back to approximate words
        guidance = f"\n\nPlease provide a response that is approximately {approx_words} words in length (about {max_tokens} tokens)."
        return prompt + guidance

    async def complete(self, prompt: str, **kwargs) -> str:
        """Async complete method required by base class but not used."""
        raise NotImplementedError("Use sync_complete instead")

    def sync_complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using OpenAI API synchronously."""
        try:
            params = {**self.default_params, **kwargs}
            target_tokens = params.get("max_tokens")
            
            if target_tokens:
                # Add length guidance to the prompt
                prompt = self._format_prompt_with_length_guidance(prompt, target_tokens)
                
                # Set a higher max_tokens limit to allow for some flexibility
                # while still staying within model limits
                params["max_tokens"] = min(
                    int(target_tokens * 1.2),  # Allow 20% more tokens for flexibility
                    self._max_tokens  # But don't exceed model's limit
                )
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI API error: {str(e)}")
            return ""

    @property
    def name(self) -> str:
        """Get engine name."""
        return f"openai-{self._model}"

    @property
    def max_tokens(self) -> int:
        """Get maximum token limit."""
        return self._max_tokens

    @property
    def default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {
            "temperature": 0.7,
            "max_tokens": None,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }

@st.cache_resource
def init_components():
    """Initialize and cache core components."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
        
    try:
        prompt_manager = PromptManager()
        
        # Load templates from prompts directory
        prompts_dir = project_root / 'prompts'
        instruction_path = prompts_dir / 'instruction_templates.json'
        jailbreak_path = prompts_dir / 'jailbreak_templates.json'
        
        if not instruction_path.exists() or not jailbreak_path.exists():
            st.error("Template files not found. Make sure the prompts directory contains required templates.")
            st.stop()
            
        try:
            prompt_manager.load_templates(
                instruction_path=str(instruction_path),
                jailbreak_path=str(jailbreak_path)
            )
            # Verify templates were loaded
            instruction_templates = prompt_manager.get_all_templates(TemplateType.INSTRUCTION)
            jailbreak_templates = prompt_manager.get_all_templates(TemplateType.JAILBREAK)
            if not instruction_templates or not jailbreak_templates:
                st.error("Failed to load templates - no templates found after loading")
                st.stop()
        except Exception as e:
            st.error(f"Failed to load templates: {str(e)}")
            st.stop()
        
        engine = SyncOpenAIEngine(api_key)
        return prompt_manager, engine
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()

# Initialize components
try:
    prompt_manager, engine = init_components()
except Exception as e:
    st.error(f"Failed to initialize: {str(e)}")
    st.stop()

def create_text_source(content: str, source_name: Optional[str] = None) -> TextSource:
    """Create a TextSource object from content."""
    return TextSource(
        content=content,
        source_id=str(hash(content)),
        metadata={},
        timestamp=datetime.now(),
        source_name=source_name
    )

def generate_prompt_combinations(text_source: TextSource) -> List[Tuple[str, str, str]]:
    """Generate all combinations of jailbreak and instruction prompts."""
    combinations = []
    instruction_templates = prompt_manager.get_all_templates(TemplateType.INSTRUCTION)
    jailbreak_templates = prompt_manager.get_all_templates(TemplateType.JAILBREAK)
    
    if not instruction_templates:
        st.error("No instruction templates found.")
        st.stop()
        
    if not jailbreak_templates:
        st.warning("No jailbreak templates found. Using instruction templates only.")
        # Use each instruction template without a jailbreak
        for i_template in instruction_templates:
            # Skip templates that require source if source_name is not provided
            if isinstance(i_template.parameters, list) and "source" in i_template.parameters and not text_source.source_name:
                #st.warning(f"Skipping template {i_template.name} as it requires source parameter")
                continue
                
            format_args = {"text": text_source.context_portion}
            if text_source.source_name and isinstance(i_template.parameters, list) and "source" in i_template.parameters:
                format_args["source"] = text_source.source_name
            prompt = i_template.template.format(**format_args)
            combinations.append((prompt, "none", i_template.name))
    else:
        # Use all combinations of jailbreak and instruction templates
        for j_template in jailbreak_templates:
            for i_template in instruction_templates:
                try:
                    prompt = prompt_manager.combine_templates(i_template.name, j_template.name)
                    # Skip templates that require source if source_name is not provided
                    if isinstance(i_template.parameters, list) and "source" in i_template.parameters and not text_source.source_name:
                        #st.warning(f"Skipping template {i_template.name} as it requires source parameter")
                        continue
                        
                    format_args = {"text": text_source.context_portion}
                    if text_source.source_name and isinstance(i_template.parameters, list) and "source" in i_template.parameters:
                        format_args["source"] = text_source.source_name
                    prompt = prompt.format(**format_args)
                    combinations.append((prompt, j_template.name, i_template.name))
                except Exception as e:
                    st.warning(f"Failed to combine templates {j_template.name} + {i_template.name}: {str(e)}")
                    continue
    
    if not combinations:
        st.error("No valid prompt combinations could be generated.")
        st.stop()
        
    return combinations

def calculate_similarity(text1: str, text2: str, metric: str) -> float:
    """Calculate similarity between two texts using specified rapidfuzz metric."""
    if metric == "ratio":
        return fuzz.ratio(text1, text2)
    elif metric == "partial_ratio":
        return fuzz.partial_ratio(text1, text2)
    elif metric == "token_sort_ratio":
        return fuzz.token_sort_ratio(text1, text2)
    elif metric == "token_set_ratio":
        return fuzz.token_set_ratio(text1, text2)
    return 0

def generate_colored_diff(completion: str, generated: str) -> str:
    """
    Generate HTML with colored diff between completion and generated text.
    Green: Words that match the completion
    Red: Words that don't match
    """
    d = difflib.Differ()
    diff = list(d.compare(completion.split(), generated.split()))
    
    html_parts = []
    i = 0
    while i < len(diff):
        word = diff[i]
        
        # Process common words and additions (which make up the generated text)
        if word.startswith('  '):  # Common word
            html_parts.append(f'<span style="color: green">{word[2:]}</span>')
        elif word.startswith('+ '):  # Addition in generated text
            html_parts.append(f'<span style="color: red">{word[2:]}</span>')
            
        i += 1
    
    return ' '.join(html_parts)

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #1e1e1e;
            padding: 2rem;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #4CAF50;
            font-size: 2.8rem !important;
            font-weight: 700 !important;
            margin-bottom: 1.5rem !important;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #3b3b3b;
            border-radius: 8px;
            padding: 12px;
            font-size: 1rem;
        }
        .stButton > button {
            background-color: #2e7d32;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 2rem;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
            font-size: 1rem;
        }
        .stButton > button:hover {
            background-color: #1b5e20;
            transform: translateY(-1px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stProgress > div > div > div {
            background-color: #2e7d32;
        }
        .stSlider > div > div > div > div {
            background-color: #2e7d32;
        }
        .stRadio > div {
            padding: 1rem;
            background: transparent;
            border-radius: 8px;
        }
        .stRadio > div > div > div > label {
            color: #ffffff !important;
        }
        .preview-box {
            padding: 15px;
            background-color: #262730;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #3b3b3b;
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #1e1e1e;
            padding: 1rem;
            text-align: center;
            border-top: 1px solid #3b3b3b;
            font-size: 0.9rem;
            z-index: 100;
        }
        .footer a {
            color: #4CAF50;
            text-decoration: none;
            font-weight: 500;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        .stSelectbox > div > div {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        .stNumberInput > div > div > input {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #3b3b3b;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Have You Been Trained On?")
    
    # Add footer
    st.markdown(
        """
        <div class="footer">
            Created by <a href="https://discretedeliberations.xyz" target="_blank">Mustafa Yasir</a> 
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Navigation
    page = st.sidebar.radio(
        "Select Feature",
        ["Text Completion", "Data Contamination Detection"]
    )
    
    if page == "Text Completion":
        st.header("Text Completion")
        st.write("Test if an LLM can complete your text. Can you find the optimal chunking and sentence split to get the best completion?")
        
        # Initialize session state
        if 'input_text' not in st.session_state:
            st.session_state.input_text = ""
            st.session_state.source_name = ""
            st.session_state.show_params = False
            st.session_state.chunk_controls = []
        
        # Input section
        input_type = st.radio("Input Type", ["URL", "Direct Text"])
    
        text = None
        source_name = st.session_state.source_name  # Use session state source name
        
        if input_type == "URL":
            # Initialize URL state if not present
            if 'url' not in st.session_state:
                st.session_state.url = ""
                st.session_state.url_text = None
                st.session_state.url_error = None
            
            url = st.text_input("Enter URL (a news article or blog post works best)", value=st.session_state.url)
            
            # Only scrape if URL changed
            if url != st.session_state.url:
                st.session_state.url = url
                st.session_state.url_text = None
                st.session_state.url_error = None
                
                if url:
                    try:
                        with st.spinner("Fetching content from URL..."):
                            scraper = ArticleScraper()
                            st.session_state.url_text = scraper.scrape_url(url)
                            if not st.session_state.url_text:
                                st.session_state.url_error = "No content found at the provided URL"
                            else:
                                source_name = url  # Use URL as source name
                                st.session_state.source_name = source_name  # Update session state
                    except Exception as e:
                        st.session_state.url_error = f"Error: {str(e)}"
            
            # Show any errors
            if st.session_state.url_error:
                st.error(st.session_state.url_error)
            
            # Use the scraped text
            text = st.session_state.url_text
        else:
            # Text input with session state
            text = st.text_area("Enter Text", height=200, key="text_input", value=st.session_state.input_text)
        
        # Add Start Analysis button
        if st.button("Start Analysis"):
            st.session_state.show_params = True
        
        # Show parameters and analyze button after clicking Start Analysis
        if st.session_state.get('show_params', False):
            if not text:
                st.error("Please enter a URL or text first")
                return
            st.subheader("Analysis Parameters")
            
            # Number of chunks
            num_chunks = st.number_input("Number of chunks", min_value=1, max_value=5, value=2)
            
            # Optional source name input
            source_name = st.text_input("Source Name (optional)", value=source_name or "")
            if source_name != st.session_state.source_name:
                st.session_state.source_name = source_name

            # Create TextSource
            text_source = create_text_source(text, source_name if source_name else None)
            chunks = text_source.split_into_chunks(num_chunks)
            
            # Chunk controls
            st.session_state.chunk_controls = []
            for i, chunk in enumerate(chunks):
                    st.subheader(f"Chunk {i+1}")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns([2, 1])
                
                    with col2:
                        # Split ratio for this chunk
                        split_ratio = st.slider(f"Split Ratio for Chunk {i+1}", 0.0, 1.0, 0.8, 0.01)
                        temp_source = create_text_source(chunk)
                        temp_source.split_for_completion(split_ratio=split_ratio)
                        context_words = len(temp_source.context_portion.split())
                        completion_words = len(temp_source.completion_portion.split())
                    
                    with col1:
                        # Calculate word counts and estimate tokens (rough estimate: 1 word ≈ 1.3 tokens)
                        total_words = context_words + completion_words
                        
                        context_tokens = int(context_words * 1.3)
                        completion_tokens = int(completion_words * 1.3)
                        total_tokens = context_tokens + completion_tokens
                        
                        # Show max token guidance with warning if needed
                        model_max_tokens = engine.max_tokens
                        if completion_tokens > model_max_tokens:
                            st.error(f"⚠️ Completion portion exceeds model's max token limit ({model_max_tokens})")
                        elif completion_tokens > model_max_tokens * 0.8:
                            st.warning(f"⚡ Completion portion is close to model's max token limit ({model_max_tokens})")
                        
                        # Show preview with split
                        st.write(f"Preview (Context | Completion):")
                        context_preview = temp_source.context_portion[:200] + "..." if len(temp_source.context_portion) > 200 else temp_source.context_portion
                        completion_preview = temp_source.completion_portion[:200] + "..." if len(temp_source.completion_portion) > 200 else temp_source.completion_portion
                        preview_html = f"""
                            <div class="preview-box">
                                <span style="color:#4CAF50">{context_preview}</span>
                                <span style="color:#FFD700"> | </span>
                                <span style="color:#FF4B4B">{completion_preview}</span>
                            </div>
                        """
                        st.markdown(preview_html, unsafe_allow_html=True)
                    
                    # Add visual separator between chunks
                    st.markdown("---")
                    
                    st.session_state.chunk_controls.append({
                        'chunk': chunk,
                        'split_ratio': split_ratio
                    })
            
            # Similarity metric selection
            metric = st.selectbox("Similarity Metric", 
                                ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"])
            
            if st.button("Run Analysis"):
                try:
                    # Create a container for all results
                    results_container = st.container()
                    
                    with results_container:
                        # Process each chunk
                        for i, chunk_control in enumerate(st.session_state.chunk_controls):
                            st.subheader(f"Analysis for Chunk {i+1}")
                            
                            # Split this chunk
                            text_source.split_for_completion(
                                split_ratio=chunk_control['split_ratio'],
                                chunk_text=chunk_control['chunk']
                            )
                            
                            with st.spinner("Generating prompt combinations..."):
                                combinations = generate_prompt_combinations(text_source)
                            
                            with st.spinner("Getting completions..."):
                                # Get LLM outputs for each prompt combination
                                results = []
                                progress_bar = st.progress(0)
                                for idx, (prompt, j_name, i_name) in enumerate(combinations):
                                    # Estimate max tokens (rough estimate: 1 word ≈ 1.3 tokens)
                                    max_tokens = int(len(text_source.completion_portion.split()) * 1.3)
                                    output = engine.sync_complete(prompt, max_tokens=max_tokens)
                                    if output:
                                        similarity = calculate_similarity(text_source.completion_portion, output, metric)
                                        results.append({
                                            'jailbreak': j_name,
                                            'instruction': i_name,
                                            'output': output,
                                            'similarity': similarity
                                        })
                                    progress_bar.progress((idx + 1) / len(combinations))
                                
                                if not results:
                                    st.error(f"No successful completions for Chunk {i+1}")
                                    continue
                                
                                # Sort by similarity and get best result
                                results.sort(key=lambda x: x['similarity'], reverse=True)
                                best_result = results[0]
                                
                                # Display the best result
                                st.write(f"Best Result (Similarity: {best_result['similarity']:.2f})")
                                st.subheader("Original")
                                original_text = f"<span style='color: gray'>{text_source.context_portion}</span> <span style='color: green'>{text_source.completion_portion}</span>"
                                st.markdown(original_text, unsafe_allow_html=True)
                                st.subheader("Generated")
                                display_completed = f"<span style='color: gray'>{text_source.context_portion}</span> {generate_colored_diff(text_source.completion_portion, best_result['output'])}"
                                st.markdown(display_completed, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

    elif page == "Data Contamination Detection":
        render_decop_ui(engine)

if __name__ == "__main__":
    main()
