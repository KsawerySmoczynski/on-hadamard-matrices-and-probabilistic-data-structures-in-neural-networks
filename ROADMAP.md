# Experiments to try
A. MNIST
    1. Hadamard + one invertable hidden dim -> try to learn identity from random weights
    2. Hadamard + one invertable hidden dim lora conditioned by positional patch matrix -> identity
    3. Hadamard + one invertable hidden dim lora conditioned by positional patch matrix + one hidden mapping dim -> numbers mapping
    4. Hadamard + one invertable hidden dim lora conditioned by positional patch matrix + one hidden mapping with lora conditioned by positional patch matrix dim (non invertible) -> numbers mapping

If succeds:

If suceeds:
B. Try with NanoEMDE data

If suceeds:
C. Try with Monad data

Notes:
* https://github.com/HazyResearch/structured-nets/tree/master CUDA implementation of hadamard matrix would require update of the cuda kernel & separate build.
