test_case="10-1-100-t"

python -m src.agent.main\
    --task abstask\
    --template abstask_plan\
    --model "meta-llama/Llama-3.1-8B-Instruct"\
    --scheduler parallel\
    --max_retry 2\
    --test_case "${test_case}"\
    --output_dir "data/result/Llama-3.1-8B-Instruct"\