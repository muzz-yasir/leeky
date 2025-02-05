"""Core implementation of DE-COP training data detection."""

import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable
from itertools import permutations
import scipy.stats as stats
from dataclasses import asdict
import tiktoken
import json
import os
import streamlit as st

from .decop_types import (
    DecopPassage, QuizPermutation, QuizResponse, PassageResults,
    ComparisonStats, DecopResult, ProcessingStage, ProcessingStatus
)
from .engines.base_engine import CompletionEngine

class DecopProcessor:
    """Implements the DE-COP method for detecting training data contamination."""
    
    def __init__(
        self,
        engine: CompletionEngine,
        tokenizer: Optional[str] = "cl100k_base",
        cache_dir: Optional[str] = None,
        status_callback: Optional[Callable[[ProcessingStatus], None]] = None
    ):
        """Initialize the DE-COP processor.
        
        Args:
            engine: LLM engine for API calls
            tokenizer: Name of tokenizer to use for counting tokens
            cache_dir: Directory to cache results (optional)
            status_callback: Function to call with processing status updates
        """
        self.engine = engine
        self.tokenizer = tiktoken.get_encoding(tokenizer)
        self.cache_dir = cache_dir
        self.status_callback = status_callback
        
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _update_status(
        self,
        stage: ProcessingStage,
        progress: float,
        message: str,
        error: Optional[str] = None,
        details: Optional[Dict] = None
    ) -> None:
        """Update processing status via callback if provided."""
        if self.status_callback:
            status = ProcessingStatus(
                stage=stage,
                progress=progress,
                message=message,
                error=error,
                details=details or {}
            )
            self.status_callback(status)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using configured tokenizer."""
        return len(self.tokenizer.encode(text))

    def _extract_passages(self, text: str, target_tokens: int = 128) -> List[str]:
        """Extract passages of approximately target_tokens length.
        
        Args:
            text: Source text to split into passages
            target_tokens: Target number of tokens per passage
            
        Returns:
            List of extracted passages
        """
        passages = []
        sentences = text.split(". ")
        current_passage = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > target_tokens * 1.2:
                # Current passage would be too long, save it and start new one
                if current_passage:
                    passages.append(". ".join(current_passage) + ".")
                current_passage = [sentence]
                current_tokens = sentence_tokens
            else:
                current_passage.append(sentence)
                current_tokens += sentence_tokens
                
            if current_tokens >= target_tokens:
                # Passage has reached target length
                passages.append(". ".join(current_passage) + ".")
                current_passage = []
                current_tokens = 0
        
        # Add any remaining text as final passage
        if current_passage:
            passages.append(". ".join(current_passage) + ".")
            
        return passages

    def _generate_paraphrases_batch(self, passages: List[DecopPassage], batch_size: int = 5) -> None:
        """Generate paraphrases for multiple passages in batches.
        
        Args:
            passages: List of passages to generate paraphrases for
            batch_size: Number of passages to process in each batch
        """
        total_batches = (len(passages) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(passages), batch_size):
            batch = passages[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            self._update_status(
                ProcessingStage.GENERATING_PARAPHRASES,
                batch_idx / len(passages),
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} passages)"
            )
            
            for passage in batch:
                try:
                    # Check cache first
                    cache_hit = False
                    if self.cache_dir:
                        cache_key = str(hash(passage.text))
                        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                        if os.path.exists(cache_file):
                            with open(cache_file, 'r') as f:
                                cached_data = json.load(f)
                                passage.paraphrases = cached_data['paraphrases']
                                cache_hit = True
                                st.write(f"Using cached paraphrases for passage from {passage.source_url}")
                                continue
                    
                    if not cache_hit:
                        prompt = f"""Please generate 3 different paraphrased versions of the following text. 
                        Maintain the same meaning and level of detail, but vary the wording and structure.
                        Keep approximately the same length.
                        
                        Text to paraphrase:
                        {passage.text}
                        
                        Generate 3 unique paraphrases, separated by [SEP]."""
                        
                        if hasattr(self.engine, 'sync_complete'):
                            response = self.engine.sync_complete(
                                prompt,
                                temperature=0.1,
                                max_tokens=self._count_tokens(passage.text) * 3 * 2
                            )
                        else:
                            response = asyncio.run(self.engine.complete(
                                prompt,
                                temperature=0.1,
                                max_tokens=self._count_tokens(passage.text) * 3 * 2
                            ))
                        
                        paraphrases = [p.strip() for p in response.split("[SEP]")]
                        passage.paraphrases = [p for p in paraphrases if p]
                        
                        # Cache results
                        if self.cache_dir:
                            with open(cache_file, 'w') as f:
                                json.dump({
                                    'text': passage.text,
                                    'paraphrases': passage.paraphrases
                                }, f)
                        
                        # Show the generated question
                        st.write(f"Generated question for passage from {passage.source_url}:")
                        st.write("Original:")
                        st.write(passage.text[:200] + "..." if len(passage.text) > 200 else passage.text)
                        st.write("Paraphrases:")
                        for i, p in enumerate(passage.paraphrases, 1):
                            st.write(f"{i}. {p[:200]}..." if len(p) > 200 else f"{i}. {p}")
                        st.write("---")
                        
                except Exception as e:
                    self._update_status(
                        ProcessingStage.ERROR,
                        0.0,
                        f"Error generating paraphrases: {str(e)}",
                        error=str(e)
                    )

    def _generate_quiz_permutations(
        self,
        passage: DecopPassage
    ) -> List[QuizPermutation]:
        """Generate all possible quiz permutations for a passage.
        
        Args:
            passage: Passage with original text and paraphrases
            
        Returns:
            List of all possible quiz permutations
        """
        options = [passage.text] + passage.paraphrases
        permutation_indices = list(permutations(range(len(options))))
        
        quiz_permutations = []
        for perm in permutation_indices:
            correct_index = perm.index(0)  # Index of original passage in this permutation
            shuffled_options = [options[i] for i in perm]
            
            quiz = QuizPermutation(
                question=f"Which of the following passages is verbatim from {passage.source_url}?",
                options=shuffled_options,
                correct_index=correct_index,
                source_passage=passage
            )
            quiz_permutations.append(quiz)
            
        return quiz_permutations

    def _evaluate_quizzes_batch(self, passages: List[DecopPassage], batch_size: int = 3) -> List[PassageResults]:
        """Evaluate quizzes for multiple passages in batches.
        
        Args:
            passages: List of passages to evaluate
            batch_size: Number of passages to process in each batch
            
        Returns:
            List of passage results
        """
        results = []
        total_batches = (len(passages) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(passages), batch_size):
            batch = passages[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            self._update_status(
                ProcessingStage.TESTING_MODEL,
                batch_idx / len(passages),
                f"Testing batch {batch_num}/{total_batches} ({len(batch)} passages)"
            )
            
            for passage in batch:
                try:
                    quizzes = self._generate_quiz_permutations(passage)
                    responses = []
                    
                    # Show one example quiz for this passage
                    example_quiz = quizzes[0]  # Take first permutation as example
                    st.write(f"Testing passage from {passage.source_url}:")
                    st.write(example_quiz.question)
                    for i, opt in enumerate(['A', 'B', 'C', 'D']):
                        st.write(f"{opt}) {example_quiz.options[i][:200]}..." if len(example_quiz.options[i]) > 200 else f"{opt}) {example_quiz.options[i]}")
                    st.write("---")
                    
                    # Process all permutations
                    for quiz in quizzes:
                        prompt = f"""{quiz.question}

A) {quiz.options[0]}
B) {quiz.options[1]}
C) {quiz.options[2]}
D) {quiz.options[3]}

Answer with just the letter (A, B, C, or D) of the passage you believe is verbatim from the source.
Provide your confidence level (0-100) after your answer, separated by a comma.
Example response format: "A, 85" """

                        start_time = datetime.now()
                        if hasattr(self.engine, 'sync_complete'):
                            response = self.engine.sync_complete(prompt, temperature=0.1)
                        else:
                            response = asyncio.run(self.engine.complete(prompt, temperature=0.1))
                        duration = (datetime.now() - start_time).total_seconds()

                        try:
                            answer, confidence = response.strip().split(",")
                            selected_index = {"A": 0, "B": 1, "C": 2, "D": 3}[answer.strip().upper()]
                            confidence_score = float(confidence.strip()) / 100
                        except (ValueError, KeyError):
                            selected_index = -1
                            confidence_score = 0.0

                        responses.append(QuizResponse(
                            permutation=quiz,
                            selected_index=selected_index,
                            confidence_score=confidence_score,
                            response_time=duration,
                            raw_response=response
                        ))
                    
                    results.append(self._analyze_passage_results(responses))
                    
                except Exception as e:
                    self._update_status(
                        ProcessingStage.ERROR,
                        0.0,
                        f"Error testing passage: {str(e)}",
                        error=str(e)
                    )
        
        return results

    def _analyze_passage_results(
        self,
        responses: List[QuizResponse]
    ) -> PassageResults:
        """Analyze results for all permutations of a passage.
        
        Args:
            responses: List of model responses for all permutations
            
        Returns:
            Analysis of passage-level results
        """
        if not responses:
            raise ValueError("No responses to analyze")
            
        passage = responses[0].permutation.source_passage
        correct_count = sum(1 for r in responses if r.selected_index == r.permutation.correct_index)
        
        # Calculate position bias
        position_counts = {i: 0 for i in range(4)}  # Initialize counts for each position
        for response in responses:
            if response.selected_index >= 0:
                position_counts[response.selected_index] += 1
        total_valid_responses = sum(position_counts.values())
        position_bias = {
            pos: count/total_valid_responses 
            for pos, count in position_counts.items()
        } if total_valid_responses > 0 else {pos: 0.0 for pos in range(4)}

        return PassageResults(
            passage=passage,
            responses=responses,
            correct_count=correct_count,
            accuracy=correct_count / len(responses),
            avg_confidence=np.mean([r.confidence_score for r in responses]),
            position_bias=position_bias
        )

    def _compute_comparison_stats(
        self,
        suspect_results: List[PassageResults],
        clean_results: List[PassageResults]
    ) -> ComparisonStats:
        """Compute statistical comparison between suspect and clean results.
        
        Args:
            suspect_results: Results for suspect passages
            clean_results: Results for clean passages
            
        Returns:
            Statistical comparison metrics
        """
        suspect_accuracies = [r.accuracy for r in suspect_results]
        clean_accuracies = [r.accuracy for r in clean_results]
        
        # Calculate means
        suspect_mean = np.mean(suspect_accuracies)
        clean_mean = np.mean(clean_accuracies)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(suspect_accuracies, clean_accuracies)
        
        # Calculate confidence interval of difference
        diff_std = np.sqrt(np.var(suspect_accuracies)/len(suspect_accuracies) + 
                          np.var(clean_accuracies)/len(clean_accuracies))
        diff_mean = suspect_mean - clean_mean
        ci = stats.t.interval(
            0.95,
            len(suspect_accuracies) + len(clean_accuracies) - 2,
            diff_mean,
            diff_std
        )
        
        # Calculate Cohen's d effect size
        pooled_std = np.sqrt((np.var(suspect_accuracies) + np.var(clean_accuracies)) / 2)
        effect_size = diff_mean / pooled_std if pooled_std != 0 else 0
        
        return ComparisonStats(
            suspect_accuracy=suspect_mean,
            clean_accuracy=clean_mean,
            accuracy_difference=diff_mean,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            sample_sizes={
                "suspect": len(suspect_accuracies),
                "clean": len(clean_accuracies)
            }
        )

    def analyze_urls(
        self,
        suspect_urls: List[str],
        clean_urls: List[str],
        scraper: Optional[Callable[[str], str]] = None
    ) -> DecopResult:
        """Run complete DE-COP analysis on suspect and clean URLs.
        
        Args:
            suspect_urls: URLs potentially in training data
            clean_urls: URLs definitely not in training data
            scraper: Optional custom function to scrape URLs
            
        Returns:
            Complete analysis results
        """
        start_time = datetime.now()
        
        try:
            # 1. Extract passages from URLs
            self._update_status(
                ProcessingStage.EXTRACTING_PASSAGES,
                0.0,
                "Extracting passages from URLs..."
            )
            
            suspect_passages = []
            clean_passages = []
            
            # Process URLs and extract passages
            for urls, passages, url_type in [
                (suspect_urls, suspect_passages, "suspect"),
                (clean_urls, clean_passages, "clean")
            ]:
                for i, url in enumerate(urls):
                    try:
                        if scraper:
                            text = scraper(url)
                        else:
                            # Use default scraping logic
                            from .text_loader import TextLoader
                            text = TextLoader.load_from_file(url)
                            
                        for passage_text in self._extract_passages(text):
                            passages.append(DecopPassage(
                                text=passage_text,
                                source_url=url,
                                token_count=self._count_tokens(passage_text)
                            ))
                        
                        self._update_status(
                            ProcessingStage.EXTRACTING_PASSAGES,
                            (i + 1) / len(urls),
                            f"Processed {i + 1}/{len(urls)} {url_type} URLs"
                        )
                    except Exception as e:
                        self._update_status(
                            ProcessingStage.ERROR,
                            0.0,
                            f"Error processing {url_type} URL {url}: {str(e)}",
                            error=str(e)
                        )

            # 2. Generate paraphrases in batches
            self._update_status(
                ProcessingStage.GENERATING_PARAPHRASES,
                0.0,
                "Generating paraphrases..."
            )
            
            all_passages = suspect_passages + clean_passages
            self._generate_paraphrases_batch(all_passages, batch_size=5)

            # 3. Generate and evaluate quizzes in batches
            self._update_status(
                ProcessingStage.TESTING_MODEL,
                0.0,
                "Testing model responses..."
            )
            
            suspect_results = self._evaluate_quizzes_batch(suspect_passages, batch_size=3)
            clean_results = self._evaluate_quizzes_batch(clean_passages, batch_size=3)

            # 4. Compute final statistics
            self._update_status(
                ProcessingStage.ANALYZING_RESULTS,
                0.0,
                "Computing final statistics..."
            )
            
            stats = self._compute_comparison_stats(suspect_results, clean_results)
            
            self._update_status(
                ProcessingStage.COMPLETE,
                1.0,
                "Analysis complete"
            )
            
            return DecopResult(
                suspect_passages=suspect_results,
                clean_passages=clean_results,
                stats=stats,
                metadata={
                    "suspect_urls": suspect_urls,
                    "clean_urls": clean_urls,
                    "engine": self.engine.name,
                    "tokenizer": self.tokenizer.name
                },
                start_time=start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            self._update_status(
                ProcessingStage.ERROR,
                0.0,
                f"Analysis failed: {str(e)}",
                error=str(e)
            )
            raise
