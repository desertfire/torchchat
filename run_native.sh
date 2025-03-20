#LD_LIBRARY_PATH=/home/binbao/local/miniconda3/envs/pytorch-3.11/lib/ cmake-out/aoti_run exportedModels/llama3_1_artifacts.pt2 -z `python3 torchchat.py where llama3.1`/tokenizer.model -i "Once upon a time"

cmake-out/aoti_libtorch_free_run exportedModels/llama3_1_artifacts_libtorch_free.pt2 -z `python3 torchchat.py where llama3.1`/tokenizer.model -i "Once upon a time"
