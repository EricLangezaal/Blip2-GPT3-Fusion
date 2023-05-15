if ! command -v conda &> /dev/null
then
    module load 2021
    module load Anaconda3/2021.05
fi
conda create -n dl2 python=3.10.11 -y
source activate dl2
conda install cudatoolkit -y
pip install ipykernel accelerate==0.18.0 bitsandbytes==0.38.1 openai
pip install git+https://github.com/salesforce/LAVIS.git
python -m spacy download en
cd ~/.conda/envs/dl2/lib/python3.10/site-packages/bitsandbytes
cp libbitsandbytes_cuda117.so libbitsandbytes_cpu.so
