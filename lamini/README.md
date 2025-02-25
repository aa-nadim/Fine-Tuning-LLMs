# Limini

```bash
python3 -m venv .venv
source .venv/bin/activate

source .venv/Scripts/activate # for windows

pip install -r requirements.txt

python app.py

pip install lamini

```



```
curl --location "https://api.lamini.ai/v1/completions" \
--header "Authorization: Bearer " \
--header "Content-Type: application/json" \
--data '{
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you?<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
}'

```