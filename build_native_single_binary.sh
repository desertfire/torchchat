#AOT_INDUCTOR_DEBUG_COMPILE=1
#AOT_INDUCTOR_PACKAGE_CPP_ONLY=1 AOT_INDUCTOR_LIBTORCH_FREE=1 python3 torchchat.py export llama3.1 --output-aoti-package-path exportedModels/llama3_1_artifacts_libtorch_free.src.pt2
rm -rf cmake-out
torchchat/utils/scripts/build_native.sh aoti_libtorch_free_single_binary
