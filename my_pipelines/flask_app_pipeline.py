import requests
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import logging


class Pipeline:
    class Valves(BaseModel):
        endpoint_api: str = "http://10.5.0.2:5001/query"
        timeout: int = 30

    def __init__(self):
        self.name = "Flask App Connector Pipeline"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def query_flask_app(self, user_message: str):
        """
        Queries the Flask app and ensures the response is correctly handled, even if it's empty or unexpected.
        """
        try:
            response = requests.post(
                self.valves.endpoint_api,
                json={"query": user_message},
                timeout=self.valves.timeout
            )
            
            response.raise_for_status()
            response_json = response.json()

            # Check for a direct LLM response or a codebase-related response
            if isinstance(response_json, list) and response_json:
                return response_json[0].get("response", "No response found."), response.status_code
            elif isinstance(response_json, dict) and "error" in response_json:
                return response_json.get("error", "An error occurred."), response.status_code
            else:
                return "No relevant data found for your query.", response.status_code
        except requests.RequestException as e:
            logging.error(f"Request to Flask app failed: {e}")
            return f"Error contacting Flask app: {str(e)}", 500
        except (ValueError, KeyError) as e:
            logging.error(f"Error processing Flask app response: {e}")
            return "Invalid response format from Flask app.", 500

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Processes the user query through the Flask app pipeline.
        """
        print(f"pipe:{__name__}")
        print(messages)
        print(user_message)

        if body.get("title", False):
            return "Flask App Connector Pipeline"
        else:
            result, status_code = self.query_flask_app(user_message)
            return result if status_code == 200 else f"Error: {result}"
