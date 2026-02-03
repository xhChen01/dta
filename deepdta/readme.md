python = 3.10

rdkit = 2023.09.6

pytorch = 2.7.1

pyg:
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.1+cpu.html
pip install torch-geometric

sklearn = 1.3.2

openpyxl = 3.1.5
conda install -c conda-forge openpyxl lxml pillow -y


将蛋白质和药物序列打乱以后，性能下降，说明序列信息是有价值的。

