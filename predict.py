# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

MODEL_NAME = "TheBloke/WizardCoder-Python-34B-V1.0-GPTQ"
MODEL_CACHE = "cache"
use_triton = False

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CACHE, 
            use_fast=True
        )
        self.model = AutoGPTQForCausalLM.from_quantized(
            MODEL_CACHE,
            use_safetensors=True,
            trust_remote_code=False,
            device="cuda:0",
            use_triton=use_triton,
            quantize_config=None,
            inject_fused_attention=False
        )

    def predict(
        self,
        prompt: str = Input(description="Your prompt", default="Tell me about AI"),
        system_prompt: str = Input(description="System prompt that helps guide system behavior", default="Below is an instruction that describes a task. Write a response that appropriately completes the request."),
        temperature: float = Input(description="Randomness of outputs, 0 is deterministic, greater than 1 is random", ge=0, le=5, default=0.7),
        max_new_tokens: int = Input(description="Number of new tokens", ge=1, le=4096 , default=512),
        top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens", ge=0.01, le=1, default=0.95),
        repetition_penalty: float = Input(description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it", ge=0, le=5, default=1.15),
    ) -> str:
        """Run a single prediction on the model"""
        prompt_template=f'''{system_prompt}

### Instruction:
{prompt}

### Response:
'''
        input_ids = self.tokenizer(prompt_template, return_tensors='pt').input_ids.to("cuda")
        outputs = self.model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_new_tokens, top_p=top_p, repetition_penalty=repetition_penalty)
        output = self.tokenizer.decode(outputs[0])
        parts = output.split("### Response:", 1)
        response = parts[1]

        return response
    