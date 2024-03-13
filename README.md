# ai_utils
ai_utils for everyone

## generate.py
A simple python utility to run models, and apply the LoRA weiths from checkpoint.
You can chat with the base model or the LoRA fine-tuned model, also have an option 
to merge the LoRA weaith to the model.

### Usage
-b      --use_base              flag chat with base model
-bm     --base_mode_id          base model from HuggingFace
-c      --checkpoint            LoRA weights checkpoint
-sm     --save_merge            flag to save the merged base model and LoRA weights
-msd    --merge_save_dir        Local path to save the merged model

Example:
python generate.py -b -bm DevQuasar/vintage-nextstep_os_systemadmin-ft-phi2
 
