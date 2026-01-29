
uv venv /tmp/venvtiny
source /tmp/venvtiny

# As of early 2026, Tiny cuda nn v2 seems to require a recent cuda version. Seems to work with cuda 13.0
# The torch version must use that cuda version as well otherwise conflicts will happen at the tiny cuda nn install
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Then install tiny-cuda-nn
uv pip install setuptools
uv pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# And test 
uv pip install commentjson imageio

# Clone the git of tiny-cuda-nn
# because running the sample needs more than just the python script
git clone https://github.com/NVlabs/tiny-cuda-nn.git
cd tiny-cuda-nn/
python samples/mlp_learning_an_image_pytorch.py
