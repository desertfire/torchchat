# python3 torchchat.py export llama3.1 --output-aoti-package-path exportedModels/llama3_1_artifacts.pt2
rm -rf cmake-out
CMAKE_PREFIX_PATH=~/local/pytorch torchchat/utils/scripts/build_native.sh aoti
