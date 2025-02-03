"""
This module implements the OpenAI engine for any models available through the API, including:
 * text-ada-001
 * text-babbage-001
 * text-curie-001
 * text-davinci-001
 * text-davinci-003
"""

# imports
import logging
import os
import time
from itertools import product, combinations_with_replacement
from pathlib import Path
from typing import Iterator

# packages
import numpy.random
import openai

# leeky imports
from leeky.engines.base_engine import BaseEngine

# set up logging
logger = logging.getLogger(__name__)

# default parameters
DEFAULT_OPENAI_API_MODEL = "gpt-3.5-turbo"

# set default valid parameters
OPENAI_VALID_PARAMETERS = {
    "temperature": [0.0, 0.5, 1.0],
    "max_tokens": [16, 32, 64, 128, 256],
    # Note: 'best_of' is no longer available in chat completion
    # Add new parameters like:
    "presence_penalty": [0.0, 0.5, 1.0],
    "frequency_penalty": [0.0, 0.5, 1.0],
}


class OpenAIEngine(BaseEngine):
    """
    OpenAI text completion engine implementing base engine interface.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_API_MODEL,
        api_key: str | None = None,
        parameters: dict | None = None,
        sleep_time: float = 0.0,
        retry_count: int = 3,
        retry_time: float = 5.0,
        seed: int | None = None,
    ) -> None:
        """
        Constructor for the engine.
        """
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Model & parameters
        self.model = model
        self.parameters = parameters if parameters else {}

        # Retry and sleep settings
        self.sleep_time = sleep_time
        self.retry_count = retry_count
        self.retry_time = retry_time

        # Setup RNG
        self.rng = numpy.random.RandomState(seed or numpy.random.randint(0, 2**32 - 1, dtype=numpy.int64))

    def get_name(self) -> str:
        """
        This method returns the name of the engine.

        Returns:
            str: The name of the engine with namespace:model format.
        """
        return f"openai:{self.model}"

    def get_current_parameters(self) -> dict:
        """
        This method returns the current parameters of the engine.

        Returns:
            dict: The current parameters of the engine.
        """
        return self.parameters

    def get_valid_parameters(self) -> Iterator[dict]:
        """
        This method returns a set of valid parameters.

        Returns:
            Iterator[dict]: An iterator of valid parameters.
        """
        # get all combinations of values from DEFAULT_OPENAI_VALID_PARAMETERS with itertools/functools
        for parameter_combination in product(*OPENAI_VALID_PARAMETERS.values()):
            # convert to dict
            parameter_dict = dict(
                zip(OPENAI_VALID_PARAMETERS.keys(), parameter_combination)
            )
            # yield the parameter dict
            yield parameter_dict

    def set_parameters(self, parameters: dict) -> None:
        """
        This method sets the parameters of the engine.

        N.B.: This does NOT update parameters.  To do so, create a copy with `get_current_parameters` and
        set from the updated copy.

        Args:
            parameters (dict): The parameters to set.
        """
        # set the parameters
        self.parameters = parameters

    def get_completions(self, prompt: str, n: int = 1) -> list[str]:
        response_list = []
        
        for _ in range(n):
            retry_count = 0
            response = None

            while response is None and retry_count < self.retry_count:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": "You are a helpful assistant."},
                                  {"role": "user", "content": prompt}],
                        **self.parameters
                    )

                    if response.choices:
                        response_list.append(response.choices[0].message.content)
                    else:
                        response = None
                except Exception as e:
                    logger.error(f"OpenAI API error (try={retry_count}): {e}")
                    retry_count += 1
                    time.sleep(self.retry_time)

        return response_list

    def get_random_parameters(self) -> dict:
        """
        Returns a random set of parameters.
        """
        return {key: self.rng.choice(values) for key, values in OPENAI_VALID_PARAMETERS.items()}
