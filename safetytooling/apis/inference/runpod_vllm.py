import asyncio
import collections
import logging
import time
from pathlib import Path
from traceback import format_exc
import json

import aiohttp

from safetytooling.data_models import LLMResponse, Prompt, StopReason
from safetytooling.utils import math_utils

from .model import InferenceAPIModel

VLLM_MODELS = {
    "dummy": "https://pod-port.proxy.runpod.net/v1/chat/completions",
    # Examples for RunPod serverless endpoints
    # "your-model-id": "https://api.runpod.ai/v2/{endpoint_id}/run",
    # Add more models and their endpoints as needed
}

LOGGER = logging.getLogger(__name__)


class VLLMChatModel(InferenceAPIModel):
    @staticmethod
    def is_runpod_serverless(url: str) -> bool:
        """Detect if URL is RunPod serverless endpoint"""
        return "runpod.ai" in url and ("/run" in url or url.endswith("/run"))
    
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        vllm_base_url: str = "http://localhost:8000/v1/chat/completions",
        vllm_api_key: str | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

        self.vllm_api_key = vllm_api_key
        self.headers = {"Content-Type": "application/json"}
        if self.vllm_api_key is not None:
            self.headers["Authorization"] = f"Bearer {self.vllm_api_key}"

        self.vllm_base_url = vllm_base_url
        # Only add /chat/completions for standard VLLM endpoints, not RunPod serverless
        if self.vllm_base_url.endswith("v1") and not self.is_runpod_serverless(self.vllm_base_url):
            self.vllm_base_url += "/chat/completions"

        self.stop_reason_map = {
            "length": StopReason.MAX_TOKENS.value,
            "stop": StopReason.STOP_SEQUENCE.value,
        }

    async def query(self, model_url: str, payload: dict, session: aiohttp.ClientSession, timeout: int = 1000) -> dict:
        async with session.post(model_url, headers=self.headers, json=payload, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                if "<!DOCTYPE html>" in str(error_text) and "<title>" in str(error_text):
                    reason = str(error_text).split("<title>")[1].split("</title>")[0]
                else:
                    reason = " ".join(str(error_text).split()[:20])
                raise RuntimeError(f"API Error: Status {response.status}, {reason}")
            return await response.json()

    def adapt_runpod_response(self, runpod_response: dict) -> dict:
        """Convert RunPod response format to OpenAI chat completions format"""
        # Convert to OpenAI format
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": runpod_response[0]['choices'][0]['tokens'][0]
                },
                "finish_reason": "stop"
            }]
        }
    
    async def run_query_serverless(self, model_url: str, messages: list, params: dict) -> dict:
        """Handle RunPod serverless API with polling"""
        # Convert URL to RunPod serverless format
        if "/v1/chat/completions" in model_url:
            submit_url = model_url.replace("/v1/chat/completions", "/run")
            base_url = model_url.replace("/v1/chat/completions", "")
        else:
            submit_url = model_url if model_url.endswith("/run") else f"{model_url}/run"
            base_url = model_url.replace("/run", "") if model_url.endswith("/run") else model_url
        
        # Prepare payload for RunPod serverless
        payload = {
            "input": {
                "messages": messages,
                "model": params["model"],
                "sampling_params": {k: v for k, v in params.items() if k != "model"},
            }
        }

        start_time = time.time()
        max_time = 600
        
        async with aiohttp.ClientSession() as session:
            async with session.post(submit_url, headers=self.headers, json=payload, timeout=60) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to submit job: Status {response.status}, {error_text}")
                job_data = await response.json()
            
            job_id = job_data["id"]
            
            # Poll for result
            status_url = f"{base_url}/status/{job_id}"
            
            while time.time() - start_time < max_time:
                await asyncio.sleep(1)  # Poll every 1 second
                
                async with session.get(status_url, headers=self.headers, timeout=30) as status_response:
                    if status_response.status != 200:
                        error_text = await status_response.text()
                        LOGGER.info(f"Status check failed: {error_text}")
                        continue
                    
                    status_data = await status_response.json()
                
                status = status_data.get("status", "UNKNOWN")
                
                if status == "COMPLETED":
                    output = status_data.get("output", {})
                    return self.adapt_runpod_response(output)
                elif status == "FAILED":
                    error_msg = status_data.get("error", "Unknown error")
                    raise RuntimeError(f"RunPod job failed: {error_msg}")
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    continue  # Keep polling
                else:
                    LOGGER.warning(f"Unknown status: {status}")
                    continue
            
            raise RuntimeError(f"RunPod job timed out after {max_time} seconds")
    
    async def run_query_sync(self, model_url: str, messages: list, params: dict) -> dict:
        """Handle synchronous VLLM API calls"""
        payload = {
            "model": params.get("model", "default"),
            "messages": messages,
            **{k: v for k, v in params.items() if k != "model"},
        }

        async with aiohttp.ClientSession() as session:
            response = await self.query(model_url, payload, session)
            if "error" in response:
                if "<!DOCTYPE html>" in str(response) and "<title>" in str(response):
                    reason = str(response).split("<title>")[1].split("</title>")[0]
                else:
                    reason = " ".join(str(response).split()[:20])
                print(f"API Error: {reason}")
                LOGGER.error(f"API Error: {reason}")
                raise RuntimeError(f"API Error: {reason}")
            return response
    
    async def run_query(self, model_url: str, messages: list, params: dict) -> dict:
        """Route to appropriate query method based on URL type"""
        if self.is_runpod_serverless(model_url):
            return await self.run_query_serverless(model_url, messages, params)
        else:
            return await self.run_query_sync(model_url, messages, params)

    @staticmethod
    def convert_top_logprobs(data: dict) -> list[dict]:
        # convert OpenAI chat version of logprobs response to completion version
        top_logprobs = []

        for item in data["content"]:
            # <|end|> is known to be duplicated on gpt-4-turbo-2024-04-09
            # See experiments/sample-notebooks/api-tests/logprob-dup-tokens.ipynb for details
            possibly_duplicated_top_logprobs = collections.defaultdict(list)
            for top_logprob in item["top_logprobs"]:
                possibly_duplicated_top_logprobs[top_logprob["token"]].append(top_logprob["logprob"])

            top_logprobs.append({k: math_utils.logsumexp(vs) for k, vs in possibly_duplicated_top_logprobs.items()})

        return top_logprobs

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        logprobs: int | None = None,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()

        model_url = VLLM_MODELS.get(model_id, self.vllm_base_url)

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response_data = None
        duration = None
        api_duration = None

        kwargs["model"] = model_id
        if logprobs is not None:
            kwargs["top_logprobs"] = logprobs
            kwargs["logprobs"] = True
            kwargs["prompt_logprobs"] = True

        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    response_data = await self.run_query(
                        model_url,
                        prompt.openai_format(),
                        kwargs,
                    )
                    api_duration = time.time() - api_start

                    if not response_data.get("choices"):
                        LOGGER.error(f"Invalid response format: {response_data}")
                        raise RuntimeError(f"Invalid response format: {response_data}")

                    if not all(is_valid(choice["message"]["content"]) for choice in response_data["choices"]):
                        LOGGER.error(f"Invalid responses according to is_valid {response_data}")
                        raise RuntimeError(f"Invalid responses according to is_valid {response_data}")
            except TypeError as e:
                raise e
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break
        else:
            LOGGER.error(f"Failed to get a response from the API after {max_attempts} attempts.")
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        if response_data is None or response_data.get("choices", None) is None:
            LOGGER.error(f"Invalid response format: {response_data}")
            raise RuntimeError(f"Invalid response format: {response_data}")

        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice["message"]["content"],
                stop_reason=choice["finish_reason"],
                api_duration=api_duration,
                duration=duration,
                cost=0,
                logprobs=self.convert_top_logprobs(choice["logprobs"]) if choice.get("logprobs") is not None else None,
            )
            for choice in response_data["choices"]
        ]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
