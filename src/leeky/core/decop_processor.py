"""Core implementation of DE-COP training data detection."""

import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable
from itertools import permutations
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _generate_paraphrases_single(self, passage: DecopPassage) -> None:
        """Generate paraphrases for a single passage."""
        try:
            # Check cache first
            if self.cache_dir:
                cache_key = str(hash(passage.text))
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        passage.paraphrases = cached_data['paraphrases']
                        return

            prompt = f"""Please generate 4 different paraphrased versions of the following text. 
            Maintain the same meaning and level of detail, but vary the wording and structure significantly.
            Keep approximately the same length. Do not number the paraphrases.
            Each paraphrase must be unique and meaningfully different from the others.
            
            Text to paraphrase:
            {passage.text}
            
            Generate 4 unique and distinct paraphrases without numbering, separated by [SEP]."""
            
            response = self.engine.sync_complete(
                prompt,
                temperature=0.1,
                max_tokens=self._count_tokens(passage.text) * 3 * 2
            )
            
            # Process and validate paraphrases
            paraphrases = [p.strip() for p in response.split("[SEP]") if p.strip()]
            valid_paraphrases = [p for p in paraphrases if len(p) >= len(passage.text) * 0.3]
            
            attempts = 0
            max_attempts = 3
            while len(valid_paraphrases) < 3 and attempts < max_attempts:
                attempts += 1
                temperature = min(0.7 + (attempts * 0.1), 0.9)
                
                response = self.engine.sync_complete(
                    prompt,
                    temperature=temperature,
                    max_tokens=self._count_tokens(passage.text) * (3 + attempts) * 2
                )
                additional_paraphrases = [p.strip() for p in response.split("[SEP]") if p.strip()]
                valid_paraphrases.extend([p for p in additional_paraphrases if len(p) >= len(passage.text) * 0.3])
                valid_paraphrases = list(set(valid_paraphrases))
            
            if len(valid_paraphrases) >= 3:
                passage.paraphrases = valid_paraphrases[:3]
            else:
                # If we still don't have enough, take what we have and pad with modified originals
                while len(valid_paraphrases) < 3:
                    modified = passage.text.replace("the", "a").replace("is", "was").replace("are", "were")
                    valid_paraphrases.append(modified)
                passage.paraphrases = valid_paraphrases[:3]
            
            # Cache results
            if self.cache_dir:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'text': passage.text,
                        'paraphrases': passage.paraphrases
                    }, f)
                    
        except Exception as e:
            raise ValueError(f"Error generating paraphrases: {str(e)}")

    def _generate_paraphrases_batch(self, passages: List[DecopPassage], batch_size: int = 5) -> None:
        """Generate paraphrases for multiple passages in parallel batches.
        
        Args:
            passages: List of passages to generate paraphrases for
            batch_size: Number of passages to process in each batch
        """
        total = len(passages)
        processed = 0
        
        # Process passages in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for passage in passages:
                futures.append(executor.submit(self._generate_paraphrases_single, passage))
            
            # Wait for each batch to complete and update progress
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred
                    processed += 1
                    self._update_status(
                        ProcessingStage.GENERATING_PARAPHRASES,
                        processed / total,
                        f"Generated paraphrases for {processed}/{total} passages"
                    )
                except Exception as e:
                    self._update_status(
                        ProcessingStage.ERROR,
                        0.0,
                        f"Error generating paraphrases: {str(e)}",
                        error=str(e)
                    )
            
            # Show example for first passage only
            if passages:
                st.write("Example paraphrase generation:")
                st.write(f"Original: {passages[0].text[:100]}..." if len(passages[0].text) > 100 else passages[0].text)
                if passages[0].paraphrases:
                    st.write(f"Paraphrase: {passages[0].paraphrases[0][:100]}..." if len(passages[0].paraphrases[0]) > 100 else passages[0].paraphrases[0])

    def _generate_quiz_permutations(
        self,
        passage: DecopPassage,
        max_attempts: int = 3,
        num_permutations: int = 8
    ) -> List[QuizPermutation]:
        """Generate a subset of quiz permutations for a passage.
        
        Args:
            passage: Passage with original text and paraphrases
            max_attempts: Maximum attempts to generate valid permutations
            num_permutations: Number of permutations to generate (default 8)
            
        Returns:
            List of quiz permutations
        """
        attempts = 0
        while attempts < max_attempts:
            try:
                # Ensure we have exactly 4 options (original + 3 paraphrases)
                if len(passage.paraphrases) < 3:
                    raise ValueError("Not enough paraphrases available")
                
                options = [passage.text] + passage.paraphrases[:3]
                if len(options) != 4:
                    raise ValueError(f"Invalid number of options: {len(options)}")
                
                # Get all possible permutations
                all_perms = list(permutations(range(len(options))))
                
                # Initialize quiz permutations list
                quiz_permutations = []
                
                # Track correct answer positions to ensure good distribution
                position_counts = {i: 0 for i in range(4)}
                
                # Randomly sample permutations while maintaining position balance
                while len(quiz_permutations) < num_permutations and all_perms:
                    # Get a random permutation
                    perm_idx = random.randrange(len(all_perms))
                    perm = all_perms.pop(perm_idx)
                    
                    correct_index = perm.index(0)  # Index of original passage
                    
                    # Check if this position is overrepresented
                    if len(quiz_permutations) >= 4:
                        max_per_position = (num_permutations + 3) // 4  # Allow slight imbalance
                        if position_counts[correct_index] >= max_per_position:
                            continue
                    
                    # Create quiz and add to list
                    shuffled_options = [options[i] for i in perm]
                    quiz = QuizPermutation(
                        question=f"Which of the following passages appears in the original source?",
                        options=shuffled_options,
                        correct_index=correct_index,
                        source_passage=passage
                    )
                    quiz_permutations.append(quiz)
                    position_counts[correct_index] += 1
                
                if not quiz_permutations:
                    raise ValueError("No valid quiz permutations generated")
                
                return quiz_permutations
                
            except Exception as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise ValueError(f"Failed to generate valid quiz after {max_attempts} attempts: {str(e)}")
                
                # Try to regenerate paraphrases
                self._generate_paraphrases_batch([passage], batch_size=1)
            
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
                    
                    # Show only one example quiz for the first passage
                    if batch_idx == 0 and passage == passages[0]:
                        example_quiz = quizzes[0]
                        st.write("Example quiz generation:")
                        st.write(example_quiz.question)
                        for i, opt in enumerate(['A', 'B', 'C', 'D']):
                            text = example_quiz.options[i]
                            st.write(f"{opt}) {text[:100]}..." if len(text) > 100 else f"{opt}) {text}")
                    
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
                        response = self.engine.sync_complete(prompt, temperature=0.1)
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
