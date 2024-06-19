from transformers import AutoModelForCausalLM, AutoTokenizer

def init_local_model(model_path, device, system_message="You are a helpful assistant."):
     # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

class BaselineModel:
    def __init__(self, model_name='qwen2-1.5b-instruct', system_message="You are a helpful assistant.", user_prompt_format="{problem}", device="cuda"):
        self.device = device
        self.model, self.tokenizer = init_local_model(model_name, device, system_message)
        self.user_prompt_format = user_prompt_format

    def answer(self, problem):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": problem}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    



