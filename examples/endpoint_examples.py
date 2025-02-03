"""
This script provides comprehensive examples of how to use the Leeky API endpoint
with various configurations and use cases.
"""

import requests
from typing import Optional, Dict, Any
from rich import print as rprint

class LeekyEndpoint:
    def __init__(self, endpoint_url: str, api_key: str):
        """
        Initialize the Leeky endpoint client.
        
        Args:
            endpoint_url: The full RunPod endpoint URL
            api_key: Your RunPod API key
        """
        self.endpoint_url = endpoint_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _make_request(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make a request to the endpoint with error handling."""
        try:
            response = requests.post(
                self.endpoint_url,
                headers=self.headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            print("Full response:", response_data)  # Debug print
            
            # Handle both possible response formats
            if isinstance(response_data, dict):
                return response_data.get("output") or response_data
            return response_data
        
        except requests.exceptions.Timeout:
            rprint("[red]Request timed out - the analysis took too long[/red]")
        except requests.exceptions.HTTPError as e:
            rprint(f"[red]HTTP Error: {e}[/red]")
        except requests.exceptions.RequestException as e:
            rprint(f"[red]Error making request: {e}[/red]")
        except KeyError as e:
            rprint(f"[red]Unexpected response format: {e}[/red]")
        except Exception as e:
            rprint(f"[red]Unexpected error: {e}[/red]")
        
        return None

    def basic_analysis(self, text: str, num_samples: int = 3) -> Optional[Dict[str, Any]]:
        """
        Perform a basic analysis with default settings.
        
        Args:
            text: The text to analyze
            num_samples: Number of samples to generate
        """
        data = {
            "input": {
                "text": text,
                "num_samples": num_samples
            }
        }
        return self._make_request(data)

    def advanced_analysis(
        self,
        text: str,
        source: str = "Legal Documentation",
        match_list: Optional[list] = None,
        selected_testers: Optional[list] = None,
        engine_config: Optional[Dict[str, str]] = None,
        num_samples: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Perform an advanced analysis with custom configurations.
        
        Args:
            text: The text to analyze
            source: Source context for contextual_recital
            match_list: List of terms to match in source_recall
            selected_testers: List of specific testers to use
            engine_config: Configuration mapping testers to engines
            num_samples: Number of samples to generate
        """
        data = {
            "input": {
                "text": text,
                "num_samples": num_samples,
                "source": source
            }
        }
        
        if match_list:
            data["input"]["match_list"] = match_list
        if selected_testers:
            data["input"]["selected_testers"] = selected_testers
        if engine_config:
            data["input"]["engine_config"] = engine_config
            
        return self._make_request(data)

    def compare_texts(
        self,
        original_text: str,
        modified_text: str,
        source: str = "Legal Database",
        match_list: Optional[list] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare two texts using source veracity and recall.
        
        Args:
            original_text: The original text
            modified_text: The modified/compared text
            source: Source context
            match_list: List of terms to match
        """
        results = {}
        for text_type, text in [("original", original_text), ("modified", modified_text)]:
            data = {
                "input": {
                    "text": text,
                    "source": source,
                    "match_list": match_list or ["RICO", "law", "criminal"],
                    "selected_testers": ["source_veracity", "source_recall"],
                    "engine_config": {
                        "source_veracity": "openai",
                        "source_recall": "bloom"
                    }
                }
            }
            results[text_type] = self._make_request(data)
        
        return results

def main():
    """Example usage of the LeekyEndpoint class"""
    
    # Initialize the endpoint
    endpoint = LeekyEndpoint(
        endpoint_url="https://api.runpod.ai/v2/7j7d04zs937ynt/run",
        api_key="rpa_X18YC8AFEGOD0VTSOEMVS74OLV0Q1EHP7OEXA1BG1hq2cx"
    )

    # 1. Basic Analysis Example
    rprint("\n[bold cyan]Basic Analysis Example:[/bold cyan]")
    basic_result = endpoint.basic_analysis(
        "We the People of the United States, in Order to form a more perfect Union..."
    )
    if basic_result:
        rprint("Basic Analysis Scores:", basic_result["scores"])

    # # 2. Advanced Analysis Example
    # rprint("\n[bold cyan]Advanced Analysis Example:[/bold cyan]")
    # advanced_result = endpoint.advanced_analysis(
    #     text="The RICO Act provides for extended criminal penalties and a civil cause of action...",
    #     source="Legal Documentation",
    #     match_list=["RICO", "criminal", "penalties", "organization"],
    #     selected_testers=["contextual_recital", "source_recall", "semantic_recital"],
    #     engine_config={
    #         "contextual_recital": "openai",
    #         "source_recall": "gptj",
    #         "semantic_recital": "bloom"
    #     }
    # )
    # if advanced_result:
    #     rprint("Advanced Analysis Scores:", advanced_result["scores"])
        
    #     rprint("\nDetailed Results:")
    #     for test_name, result in advanced_result["full_results"].items():
    #         rprint(f"\n[bold]{test_name}:[/bold]")
    #         rprint(f"Score: {result['score']}")
    #         if 'samples' in result:
    #             rprint("Samples:")
    #             for sample in result['samples']:
    #                 rprint(f"- {sample}")

    # # 3. Text Comparison Example
    # rprint("\n[bold cyan]Text Comparison Example:[/bold cyan]")
    # texts = {
    #     "original": "The RICO Act is a United States federal law that provides for extended criminal penalties.",
    #     "modified": "The RICO Act is a helpful law that maybe does something about crime, probably."
    # }
    
    # comparison_results = endpoint.compare_texts(
    #     original_text=texts["original"],
    #     modified_text=texts["modified"],
    #     source="Legal Database",
    #     match_list=["RICO", "law", "criminal"]
    # )
    
    # if comparison_results:
    #     for text_type, results in comparison_results.items():
    #         rprint(f"\n[bold]{text_type.title()} Text Results:[/bold]")
    #         rprint("Scores:", results["scores"])

if __name__ == "__main__":
    main()