"""One full Single trajectory per question, preserving every output token ID."""

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import time


@dataclass(frozen=True)
class Generation:
    prompt_ids: list[int]
    output_ids: list[int]
    user_prompt: str
    chat_prompt: str
    finish_reason: str
    latency_sec: float


class HFSingleBackend:
    method = "single"
    enable_thinking = True

    def __init__(self, model_name: str, revision: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Full-trajectory generation requires a CUDA GPU")
        self.torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, dtype=torch.bfloat16, device_map="auto"
        ).eval()
        device_map = getattr(self.model, "hf_device_map", {})
        if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
            raise RuntimeError(f"Model offloaded outside GPU: {device_map}")
        start = self.tokenizer.encode("<think>", add_special_tokens=False)
        end = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(start) != 1 or len(end) != 1:
            raise ValueError("Expected single-token Qwen3 thinking delimiters")
        self.think_start_id, self.think_end_id = start[0], end[0]
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_ids = set([eos] if isinstance(eos, int) else (eos or []))

    def model_revision(self) -> str | None:
        return getattr(self.model.config, "_commit_hash", None)

    def tokenizer_fingerprint(self) -> str:
        if not hasattr(self, "_tokenizer_fingerprint"):
            data = {"vocab": self.tokenizer.get_vocab(),
                    "chat_template": self.tokenizer.chat_template,
                    "special_tokens": self.tokenizer.special_tokens_map}
            self._tokenizer_fingerprint = hashlib.sha256(
                json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return self._tokenizer_fingerprint

    def generation_signature(self) -> str:
        data = {"generation_config": self.model.generation_config.to_dict(),
                "transformers": version("transformers"), "torch": version("torch"),
                "dtype": str(self.model.dtype)}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def generate_trajectory(self, question: str, prompt_template: str, *, budget: int,
                            seed: int, temperature: float, top_p: float,
                            top_k: int) -> Generation:
        from transformers import set_seed

        if prompt_template.count("{question}") != 1:
            raise ValueError("Prompt template must contain {question} exactly once")
        user_prompt = prompt_template.format(question=question).strip()
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        prompt_ids = self.tokenizer(chat_prompt, add_special_tokens=False,
                                    return_tensors="pt").input_ids[0].tolist()
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(context_limit, int) and len(prompt_ids) + budget > context_limit:
            raise ValueError("Requested trajectory exceeds the model context length")
        set_seed(seed)
        inputs = self.torch.tensor([prompt_ids], device=self.model.device)
        self.torch.cuda.synchronize(self.model.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                inputs, max_new_tokens=budget, do_sample=True,
                temperature=temperature, top_p=top_p, top_k=top_k,
                pad_token_id=(self.tokenizer.pad_token_id
                              if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id),
            )
        self.torch.cuda.synchronize(self.model.device)
        latency = time.perf_counter() - start
        output_ids = output[0, len(prompt_ids):].tolist()
        if not output_ids or len(output_ids) > budget:
            raise ValueError("Backend returned an invalid output length")
        eos_at = next((i for i, token in enumerate(output_ids) if token in self.eos_ids), None)
        if eos_at is not None:
            output_ids = output_ids[:eos_at + 1]
        return Generation(prompt_ids, output_ids, user_prompt, chat_prompt,
                          "eos" if eos_at is not None else "length", latency)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)
