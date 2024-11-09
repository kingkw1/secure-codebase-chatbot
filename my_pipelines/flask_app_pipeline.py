import requests
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

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
        try:
            # Modify the URL and payload as needed for your Flask app
            response = requests.post(
                self.valves.endpoint_api, 
                json={"query": user_message},
                timeout=self.valves.timeout
            )
            
            response.raise_for_status()
            
            return response.json()[0].get("response", ""), response.status_code
        except requests.RequestException as e:
            return str(e), 500

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        print(messages)
        print(user_message)

        if body.get("title", False):
            return "Flask App Connector Pipeline"
        else:        
            result, status_code = self.query_flask_app(user_message)
            return result if status_code == 200 else f"Error: {result}"
