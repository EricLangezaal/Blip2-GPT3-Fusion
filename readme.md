# Improving BLIP-2 by employing GPT-3 in context learning
This GitHub repository details all our code to both reproduce some of the original experiments in the paper "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" 
as well our additions. Since SalesForce has packaged the LAVIS repository such that it can be installed as Pip package, we have not cloned their code in this repository.
We do need to build the LAVIS repository from source, since they updated the master branch regarding a GitHub issue (re)raised by us, which is not yet included in the PyPI version.
As such we have only included the minimal required code to reproduce two of their experiments (see the notebook and blogspot for details). 

## 1. Environment installation
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

Install dependencies required for the HuggingFace model quantization, OpenAI and various others. Also build the LAVIS package from source
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

## 2. Downloading datasets
By default the Lavis library is able to download the VQAV2, OKVQA and GQA datasets whenever it needs them. It will however attempt to do so in a root level folder, which will often lead to permission issues (for example on the Lisa cluster). As such, we created custom scripts to download all datasets to legal locations in our ecosystem.

To download the VQA/OKVQA dataset (which share the same images), please move to the `src/data/` folder and run:
```bash
download_coco.py
```
To download the GQA dataset
```bash
download_gqa.py
```

Since these downloads likely take over 15 minutes, you can also schedule `run_dataset.job` from that folder.

## 3. Rerunning an evaluation experiment
Using our codebase reproducing an experiment from the original BLIP-2 paper on VQAV2, OKVQA and GQA can be achieved by running:
```bash
cd src/
python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path reproducing/configs/<experiment_config>.yaml 
```
Where `<experiment_config>` is the name of one of the experiment configuration files for the respective model and dataset.

In a similar manner, it is possible to evaluate our custom pipeline on OKVQA. Our ablation study which let GPT-3 pick three questions for BLIP-2 can also be tested separately using its configuration file. To evaluate our final pipeline on the OKVQA test set:
```bash
cd src/
python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path extensions/configs/okvqa_flant5xl_caption_gpt3.yaml
```
There is also a Slurm job file available called `run_eval.job`, make sure to also modify the configuration file path if applicable.
