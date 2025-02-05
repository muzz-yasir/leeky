"""Streamlit UI components for DE-COP training data detection."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

from .decop_processor import DecopProcessor
from .decop_types import ProcessingStage, ProcessingStatus, DecopResult, DecopPassage
from .engines.base_engine import CompletionEngine

def render_decop_ui(engine: CompletionEngine):
    """Render the DE-COP training data detection UI.
    
    Args:
        engine: LLM engine instance to use for analysis
    """
    st.title("Data Contamination Detection")
    st.write("""
    Provide URLs to potentially contaminated content and known clean content for comparison. 
    We will quiz ChatGPT with paraphrases of the content to see if it can correctly identify true extracts.
    A higher accuracy for suspect URLs indicates that the model has seen this content before.
    """)
    
    # Initialize session state
    if 'decop_status' not in st.session_state:
        st.session_state.decop_status = None
    if 'decop_result' not in st.session_state:
        st.session_state.decop_result = None
    
    # Input sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Suspect URLs")
        suspect_urls = st.text_area(
            "Enter URLs to test (one per line)",
            height=150,
            key="suspect_urls",
            help="URLs of content that might be in the training data"
        ).strip().split('\n')
        suspect_urls = [url.strip() for url in suspect_urls if url.strip()]
        
    with col2:
        st.subheader("Clean URLs")
        clean_urls = st.text_area(
            "Enter known clean URLs (one per line)",
            height=150,
            key="clean_urls",
            help="URLs of content definitely not in training data (e.g., recent articles)"
        ).strip().split('\n')
        clean_urls = [url.strip() for url in clean_urls if url.strip()]
    
    # Cache directory setup
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "cache", "decop")
    
    # Analysis options
    with st.expander("Analysis Options"):
        st.write("Configure analysis parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            passage_tokens = st.number_input(
                "Tokens per passage",
                min_value=64,
                max_value=256,
                value=128,
                help="Target number of tokens for each passage"
            )
            
        with col2:
            paraphrase_temp = st.slider(
                "Paraphrase temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                help="Temperature for paraphrase generation"
            )
    
    # Button layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Extract passages button
        extract_disabled = not (suspect_urls and clean_urls) or st.session_state.get('passages_updated', False)
        if st.button("Extract Passages", disabled=extract_disabled):
            if not suspect_urls:
                st.error("Please enter at least one suspect URL")
                return
            if not clean_urls:
                st.error("Please enter at least one clean URL")
                return

            processor = DecopProcessor(
                engine=engine,
                cache_dir=cache_dir,
                status_callback=lambda status: _update_status(status)
            )

            # Store processor in session state for later use
            st.session_state.processor = processor
            
            try:
                # Initialize passage containers
                suspect_passages = []
                clean_passages = []
                all_passages = {}
                
                # Extract passages for each URL
                for urls, url_type in [
                    (suspect_urls, "Suspect"),
                    (clean_urls, "Clean")
                ]:
                    for url in urls:
                        text = _scrape_url(url)
                        if text:
                            extracted = processor._extract_passages(text)
                            all_passages[f"{url_type}_{url}"] = extracted
                            
                            # Create DecopPassage objects
                            for passage_text in extracted:
                                passage = DecopPassage(
                                    text=passage_text,
                                    source_url=url,
                                    token_count=processor._count_tokens(passage_text)
                                )
                                if url_type == "Suspect":
                                    suspect_passages.append(passage)
                                else:
                                    clean_passages.append(passage)
                
                # Store in session state
                st.session_state.all_passages = all_passages
                st.session_state.suspect_passages = suspect_passages
                st.session_state.clean_passages = clean_passages
                st.session_state.passages_updated = True
                
                # Force a rerun to update UI
                st.rerun()
            
            except Exception as e:
                st.error(f"Error extracting passages: {str(e)}")
                return
    
    with col2:
        # Reset button
        if st.button("Reset All", key="reset_button"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key not in ['suspect_urls', 'clean_urls']:  # Preserve URL inputs
                    del st.session_state[key]
            st.rerun()
    
    with col3:
        # Run analysis button (only show if passages are updated)
        if st.session_state.get('passages_updated', False):
            if st.button("Run Analysis", key="run_analysis"):
                processor = st.session_state.processor
                suspect_passages = st.session_state.suspect_passages
                clean_passages = st.session_state.clean_passages
                
                # Create progress tracking
                progress_container = st.empty()
                status_container = st.empty()
                
                try:
                    # Generate paraphrases and run analysis
                    all_passages = suspect_passages + clean_passages
                    
                    # Use sync_complete for all operations
                    processor._generate_paraphrases_batch(all_passages, batch_size=5)
                    suspect_results = processor._evaluate_quizzes_batch(suspect_passages, batch_size=3)
                    clean_results = processor._evaluate_quizzes_batch(clean_passages, batch_size=3)
                    
                    stats = processor._compute_comparison_stats(suspect_results, clean_results)
                    
                    result = DecopResult(
                        suspect_passages=suspect_results,
                        clean_passages=clean_results,
                        stats=stats,
                        metadata={
                            "suspect_urls": suspect_urls,
                            "clean_urls": clean_urls,
                            "engine": engine.name,
                            "tokenizer": processor.tokenizer.name
                        },
                        start_time=datetime.now(),
                        end_time=datetime.now()
                    )
                
                    # Store result in session state
                    st.session_state.decop_result = result
                    
                    # Clear progress indicators
                    progress_container.empty()
                    status_container.empty()
                    
                    # Show results
                    _render_results(result)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    return
    
    # Display extracted passages if available
    if st.session_state.get('passages_updated', False):
        # Create a single form for all passages
        with st.form(key="all_passages_form"):
            for url_type in ["Suspect", "Clean"]:
                st.subheader(f"{url_type} Passages")
                passages = st.session_state.suspect_passages if url_type == "Suspect" else st.session_state.clean_passages
                
                for i, passage in enumerate(passages):
                    with st.container():
                        col1, col2 = st.columns([8, 2])
                        
                        with col1:
                            passage.text = st.text_area(
                                f"Passage {i+1} from {passage.source_url}",
                                value=passage.text,
                                height=100,
                                key=f"text_{url_type}_{i}"
                            )
                        
                        with col2:
                            include = st.selectbox(
                                "Status",
                                ["Include", "Exclude"],
                                key=f"include_{url_type}_{i}"
                            )
                            passage.include = (include == "Include")
                        
                        st.markdown("---")
            
            # Update button at the end of the form
            update_clicked = st.form_submit_button("Update Passages")
        
        # Handle form submission
        if update_clicked:
            # Filter included passages
            st.session_state.suspect_passages = [p for p in st.session_state.suspect_passages if getattr(p, 'include', True)]
            st.session_state.clean_passages = [p for p in st.session_state.clean_passages if getattr(p, 'include', True)]
            
            # Update token counts
            processor = st.session_state.processor
            for passages in [st.session_state.suspect_passages, st.session_state.clean_passages]:
                for p in passages:
                    p.token_count = processor._count_tokens(p.text)
            
            # Force a rerun to update UI
            st.rerun()
    
    # Show existing results if available
    if st.session_state.decop_result:
        _render_results(st.session_state.decop_result)

def _update_status(status: ProcessingStatus) -> None:
    """Update UI with current processing status."""
    st.session_state.decop_status = status
    
    # Update progress bar
    progress_container = st.empty()
    with progress_container:
        st.progress(status.progress)
    
    # Update status message
    status_container = st.empty()
    with status_container:
        st.write(status.message)
        if status.error:
            st.error(status.error)

def _scrape_url(url: str) -> str:
    """Scrape content from URL using ArticleScraper."""
    from .scraper import ArticleScraper
    scraper = ArticleScraper()
    return scraper.scrape_url(url)

def _render_results(result: DecopResult) -> None:
    """Render analysis results with visualizations."""
    st.header("Analysis Results")
    
    # Create a container for full width
    with st.container():
        # Overall statistics
        st.subheader("Detection Statistics")
        metrics_container = st.container()
        col1, col2, col3 = metrics_container.columns(3)
        
        with col1:
            st.metric(
                "Suspect Accuracy",
                f"{result.stats.suspect_accuracy:.1%}",
                help="Average accuracy for suspect passages"
            )
        
        with col2:
            st.metric(
                "Clean Accuracy",
                f"{result.stats.clean_accuracy:.1%}",
                help="Average accuracy for clean passages"
            )
        
        with col3:
            st.metric(
                "Difference",
                f"{result.stats.accuracy_difference:+.1%}",
                help="Difference in detection accuracy"
            )
        
        # Statistical significance
        st.subheader("Statistical Analysis")
        stats_container = st.container()
        col1, col2 = stats_container.columns(2)
        with col1:
            st.metric(
                "p-value",
                f"{result.stats.p_value:.3f}",
                help="Statistical significance (p < 0.05 indicates significant difference)"
            )
        
        with col2:
            st.metric(
                "Effect Size",
                f"{result.stats.effect_size:.2f}",
                help="Cohen's d effect size"
            )
    
    # Confidence interval
    ci_low, ci_high = result.stats.confidence_interval
    st.write(f"95% Confidence Interval: [{ci_low:.3f}, {ci_high:.3f}]")
    
    # Accuracy distribution plot
    st.subheader("Accuracy Distribution")
    
    try:
        # Safely extract accuracies with error handling
        suspect_accuracies = []
        clean_accuracies = []
        
        if hasattr(result, 'suspect_passages'):
            suspect_accuracies = [r.accuracy for r in result.suspect_passages if hasattr(r, 'accuracy')]
        if hasattr(result, 'clean_passages'):
            clean_accuracies = [r.accuracy for r in result.clean_passages if hasattr(r, 'accuracy')]
        
        if suspect_accuracies or clean_accuracies:
            fig = go.Figure()
            if suspect_accuracies:
                fig.add_trace(go.Violin(
                    y=suspect_accuracies,
                    name="Suspect",
                    box_visible=True,
                    meanline_visible=True
                ))
            if clean_accuracies:
                fig.add_trace(go.Violin(
                    y=clean_accuracies,
                    name="Clean",
                    box_visible=True,
                    meanline_visible=True
                ))
            
            fig.update_layout(
                title="Distribution of Detection Accuracies",
                yaxis_title="Accuracy",
                showlegend=True
            )
            
            st.plotly_chart(fig)
        else:
            st.warning("No accuracy data available for visualization")
            
    except Exception as e:
        st.error(f"Error generating accuracy distribution plot: {str(e)}")
    
    # Position bias analysis
    st.subheader("Position Bias Analysis")
    
    try:
        # Safely extract position bias data with error handling
        if hasattr(result, 'suspect_passages') and hasattr(result, 'clean_passages'):
            # Combine position bias data
            suspect_bias = {
                pos: sum(r.position_bias[pos] for r in result.suspect_passages if hasattr(r, 'position_bias')) / len(result.suspect_passages)
                for pos in range(4)
            }
            clean_bias = {
                pos: sum(r.position_bias[pos] for r in result.clean_passages if hasattr(r, 'position_bias')) / len(result.clean_passages)
                for pos in range(4)
            }
            
            bias_data = pd.DataFrame({
                'Position': ['A', 'B', 'C', 'D'] * 2,
                'Selection Rate': list(suspect_bias.values()) + list(clean_bias.values()),
                'Type': ['Suspect'] * 4 + ['Clean'] * 4
            })
            
            fig = px.bar(
                bias_data,
                x='Position',
                y='Selection Rate',
                color='Type',
                barmode='group',
                title='Answer Position Selection Rates'
            )
            
            st.plotly_chart(fig)
        else:
            st.warning("No position bias data available for visualization")
            
    except Exception as e:
        st.error(f"Error generating position bias plot: {str(e)}")
    
    # Detailed results in full width container
    with st.container():
        st.subheader("Detailed Results")
        
        # Suspect passages
        with st.expander("Suspect Passages"):
            for i, passage_result in enumerate(result.suspect_passages):
                st.write(f"Passage {i+1} from {passage_result.passage.source_url}")
                st.write("Text:")
                st.write(passage_result.passage.text)
                st.write(f"Accuracy: {passage_result.accuracy:.1%}")
                st.write(f"Confidence: {passage_result.avg_confidence:.1%}")
                st.write("---")
        
        # Clean passages
        with st.expander("Clean Passages"):
            for i, passage_result in enumerate(result.clean_passages):
                st.write(f"Passage {i+1} from {passage_result.passage.source_url}")
                st.write("Text:")
                st.write(passage_result.passage.text)
                st.write(f"Accuracy: {passage_result.accuracy:.1%}")
                st.write(f"Confidence: {passage_result.avg_confidence:.1%}")
                st.write("---")
    
    # Download results
    st.download_button(
        "Download Full Results",
        data=_format_results_json(result),
        file_name=f"decop_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def _format_results_json(result: DecopResult) -> str:
    """Format results as JSON string for download."""
    import json
    from dataclasses import asdict
    
    def _serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    
    return json.dumps(asdict(result), default=_serialize, indent=2)
