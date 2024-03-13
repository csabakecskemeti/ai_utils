import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


def generate(model, eval_tokenizer):
    """
    Wait for text input from the user and print it.
    Keep waiting until interrupted by Ctrl+C.
    """

    try:
        while True:
            user_input = input("User (Ctrl+C to exit): ")
            model_input = eval_tokenizer(user_input, return_tensors="pt").to("cuda")
            with torch.no_grad():
                chat_response = eval_tokenizer.decode(
                    model.generate(**model_input, max_new_tokens=500, repetition_penalty=1.11)[0],
                    skip_special_tokens=True)
            print(type(chat_response))
            print(f'[USER]: {user_input}\n[CHAT]: {chat_response}\n')
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="A script to demonstrate argument parsing.")

    # Add positional arguments
    parser.add_argument('-bm',
                        '--base_model_id',
                        type=str,
                        help='The base model has LORA trained',
                        default="microsoft/phi-2")
    parser.add_argument('-c',
                        '--checkpoint',
                        type=str,
                        help='Training checkpoint to load',
                        default="/my_checkpoints/checkpoint-2540")
    parser.add_argument('-b', '--use_base', action='store_true', help='Chat with base model')
    parser.add_argument('-sm', '--save_merge', action='store_true', help='Merge and save the merged model')
    parser.add_argument('-msd',
                        '--merge_save_dir',
                        type=str,
                        help='Path to save the merged model',
                        default="lora-merged")

    # Parse the command-line arguments
    args = parser.parse_args()

    base_model_id = args.base_model_id
    checkpoint = args.checkpoint

    print(f'Using LORA checkpoint {checkpoint} with base mode {base_model_id} for generation')
    print(f'Load base model\'s tokenizer: {base_model_id}')
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,  # needed for now, should be fixed soon
    )
    tokenizer.pad_token = tokenizer.eos_token

    eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True,
                                                   use_fast=False)
    eval_tokenizer.pad_token = tokenizer.eos_token

    print(f'Load base model: {base_model_id}')
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Phi2, same as before
        device_map="auto",
        trust_remote_code=True,
        # load_in_8bit=True,
        # torch_dtype=torch.float16,
    )

    if args.use_base:
        print(f'Using base mode {base_model_id} for generation')
        generate(base_model, eval_tokenizer)
    else:
        print(f'Using LORA checkpoint {checkpoint} with base mode {base_model_id} for generation')
        ft_model = PeftModel.from_pretrained(base_model, checkpoint)
        if args.save_merge:
            peft_model = ft_model.merge_and_unload()
            peft_model.save_pretrained(save_directory=args.merge_save_dir)
            tokenizer.save_pretrained(save_directory=args.merge_save_dir)
        ft_model.eval()
        generate(ft_model,eval_tokenizer)
if __name__ == "__main__":
    main()

