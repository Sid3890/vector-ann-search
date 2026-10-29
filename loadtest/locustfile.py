
import json

from locust import HttpUser, between, task


class SearchUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def search(self):
        payload = {
            "query_vector": [0.0] * 384,
            "top_k": 5,
        }

        with self.client.post(
            "/search",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/search",
            catch_response=True,
            timeout=15,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                return

            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError):
                response.failure("Response was not valid JSON")
                return

            required_fields = {"ids", "distances", "latency_ms", "partial"}

            if not isinstance(data, dict) or not required_fields.issubset(data.keys()):
                response.failure(
                    f"Unexpected search response schema: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
                )
