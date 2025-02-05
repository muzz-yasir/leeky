"""Streamlit UI components for DE-COP training data detection."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import os

from .decop_processor import DecopProcessor
from .decop_types import ProcessingStage, ProcessingStatus, DecopResult
from .engines.base_engine import CompletionEngine

def render_decop_ui(engine: CompletionEngine):
    """Render the DE-COP training data detection UI.
    
    Args:
        engine: LLM engine instance to use for analysis
    """
    st.title("Training Data Detector")
    st.write("""
    This tool implements the DE-COP method to detect if content was likely used in LLM training.
    Provide URLs to potentially contaminated content and known clean content for comparison.
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
    
    # Run analysis button
    if st.button("Run Analysis", disabled=not (suspect_urls and clean_urls)):
        if not suspect_urls:
            st.error("Please enter at least one suspect URL")
            return
        if not clean_urls:
            st.error("Please enter at least one clean URL")
            return
            
        # Initialize processor
        processor = DecopProcessor(
            engine=engine,
            cache_dir=cache_dir,
            status_callback=lambda status: _update_status(status)
        )
        
        # Create progress tracking
        progress_container = st.empty()
        status_container = st.empty()
        
        try:
            # Run analysis
            result = processor.analyze_urls(
                suspect_urls=suspect_urls,
                clean_urls=clean_urls,
                scraper=lambda url: _scrape_url(url)  # Use existing scraper
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
    
    # Show existing results if available
    elif st.session_state.decop_result:
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
    
    # Overall statistics
    st.subheader("Detection Statistics")
    col1, col2, col3 = st.columns(3)
    
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
    
    col1, col2 = st.columns(2)
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
    
    suspect_accuracies = [r.accuracy for r in result.suspect_passages]
    clean_accuracies = [r.accuracy for r in result.clean_passages]
    
    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=suspect_accuracies,
        name="Suspect",
        box_visible=True,
        meanline_visible=True
    ))
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
    
    # Position bias analysis
    st.subheader("Position Bias Analysis")
    
    # Combine position bias data
    suspect_bias = {
        pos: sum(r.position_bias[pos] for r in result.suspect_passages) / len(result.suspect_passages)
        for pos in range(4)
    }
    clean_bias = {
        pos: sum(r.position_bias[pos] for r in result.clean_passages) / len(result.clean_passages)
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
    
    # Detailed results
    st.subheader("Detailed Results")
    
    with st.expander("Suspect Passages"):
        for i, result in enumerate(result.suspect_passages):
            st.write(f"Passage {i+1} from {result.passage.source_url}")
            st.write(f"Accuracy: {result.accuracy:.1%}")
            st.write(f"Confidence: {result.avg_confidence:.1%}")
    
    with st.expander("Clean Passages"):
        for i, result in enumerate(result.clean_passages):
            st.write(f"Passage {i+1} from {result.passage.source_url}")
            st.write(f"Accuracy: {result.accuracy:.1%}")
            st.write(f"Confidence: {result.avg_confidence:.1%}")
    
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
