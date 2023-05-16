# Improving BLIP-2 by employing ChatGPT3 in context learning
This GitHub repository details all our code to both reproduce some of the original experiments in the paper "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" 
as well our additions. Since SalesForce has packaged the LAVIS repository such that it can be installed as Pip package, we have not cloned their code in this repository.
We do need to build the LAVIS repository from source, since they updated the master branch regarding a GitHub issue (re)raised by us, which is not yet included in the PyPI version.
As such we have only included the minimal required code to reproduce two of their experiments (see the notebook and blogspot for details). 

## Environment installation
To get conflicting dependencies to function we needed to modify the environment more than a simple YAML file would allow.
For installation specifically on a Lisa cluster environment we made a script to automate the installation process.
```bash
bash create_env.sh
```
### Manual installation
If the script fails, you can follow the manual steps below. First, create a new conda environment named 'dl2' for example.
```bash
conda create -n dl2 python=3.10.11 -y
```
Activate the environment and install the CUDA Toolkit
```bash
source activate dl2
conda install cudatoolkit -y
```

Install dependencies required for the HuggingFace model quantization, OpenAI and various other. Also build the LAVIS package from source
```bash
pip install ipykernel accelerate==0.18.0 bitsandbytes==0.38.1 openai nbconvert
pip install git+https://github.com/salesforce/LAVIS.git
```

Download 'en_core_web_sm'
```bash
python -m spacy download en
```

Modify the installed version of BitsAndBytes if you get errors indicating 'missing symbols'. Our CUDA version was '117' for example.
```bash
cd <your conda environments location>/envs/dl2/lib/python3.10/site-packages/bitsandbytes
cp libbitsandbytes_cuda<your cuda version>.so libbitsandbytes_cpu.so
```
