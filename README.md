1. Uninstall all installed packages in Colab: `pip freeze | cut -d "@" -f1 | xargs pip uninstall -y`
2. Install all libs from requirements: `pip install -r requirements.txt`
3. Install lm-eval: 

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
``` 
4. Add .env file with your Hugging Face token: `HF_TOKEN=your_token`
5. Check config.yaml and modify if needed
6. Run evaluation: `python run_evaluation.py --quantization-method [awq, gptq, bitsandbytes]`

For a quick test run add `--smoke-test`.

For evaluation on a full MMLU dataset add `--mode full`.