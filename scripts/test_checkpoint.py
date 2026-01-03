"""Quick test of a DAPT checkpoint."""
import sys
from pathlib import Path

def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/dapt/checkpoint-500"
    
    from unsloth import FastLanguageModel
    
    print(f"Loading checkpoint: {checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Test prompts - TypeScript code completion
    prompts = [
        "// TypeScript function to fetch data from an API\nasync function fetchData(",
        "interface User {\n  id: number;\n  name: string;\n",
        "const express = require('express');\nconst app = express();\n\napp.get('/',",
    ]
    
    for prompt in prompts:
        print(f"\n{'='*60}\nPrompt:\n{prompt}\n{'-'*60}\nGenerated:")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
