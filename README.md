## Artifact for Understanding and Bridging the Gaps in Current GNN Performance Optimizations

### Getting started

#### Environment

* Python 3.7
* CUDA 10.1
* libcusparse.so.10
* libcurand.so.10
* libcublas.so.10
* Python packages
  * DGL 0.4.3post2: pip3 install dgl-cu101==0.4.3post2
  * Pytorch 1.6.0: pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
  * torch_scatter 2.0.5: pip3 install torch-scatter==2.0.5+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
  * datasketch 1.5.1: pip3 install datasketch==1.5.1
  * matplotlib # for figure generation
  * seaborn # for figure generation

#### Hardware

* Tesla V100-PCIE-32GB

#### Get the code

```bash
git clone git@github.com:xxcclong/GNN-Computing.git
```

#### Setting up

```bash
cd artifact
mkdir build && cd build
cmake ..
make -j16
cp fig7.out ../Figure7/
cp fig8.out ../Figure8/
cp fig9.out ../Figure9/
cp fig10a.out ../Figure10/
cp fig10b.out ../Figure10/
cp fig11.out ../Figure11/
```

#### Data preparation

```bash
# get the compressed data
# put them into artifact/data/
wget -O data.zip https://cloud.tsinghua.edu.cn/f/2eebc696ce054681a6a4/?dl=1
# or download from onedrive: https://1drv.ms/u/s!Apc72a8BNm47f8k2kJEwBTdB-_o?e=JZ5zPd
# or download from dropbox: https://www.dropbox.com/s/d75okzxgy1uwyqk/data.zip?dl=0
unzip data.zip
```

After it, the file structure should be as follows
```
.
|-- CMakeLists.txt
|-- Figure10
|-- Figure11
|-- Figure7
|-- Figure8
|-- Figure9
|-- README.md
|-- data
|-- data_pyg
|-- include
`-- src
```

For every dataset (taking `arxiv` for example), we have

* arxiv.config: with some graph information, such as the number of nodes and edges.
* arxiv.graph: two lines, the first line has the poiters of the range of the neighbors, the second line has the neighbor indexes. (similar to CSR format)
* arxiv.reorder_thres_0.2: the preprocessed reorder file, containing the number of `0` to `num_v - 1`, indicating the new node order of the graph.


### Reproduce

#### Figure 7

```bash
cd Figure7
./run.sh
python3 draw_fig7.py # get fig7.pdf
```

P.S.

1. PyG will expand the on-node tensor to edges, as a result, it will lead to out of memory. So you can find there are "RuntimeError: CUDA out of memory." during the test. However, the script can continue running, and the "out of memory" will be shown as "out of support" in the generated figure.
2. The generated figure will not have breaks, so it looks unsimilar with the one in paper, but them have similar numbers.
3. We can use `python3 dgl_prof.py --model sagelstm --gpu 0 --syn-name datasetname` to run GraphSAGE-LSTM using DGL. But due to its implementation, the CPU scheduling time is too much. So we re-implement it using CUDA, with negligible CPU overhead, for a fair comparison.

#### Figure 8

```bash
cd Figure8
./run.sh
python3 draw_fig8.py # get fig8.pdf
```

#### Figure 9

```bash
cd Figure9
./run.sh
python3 draw_fig9.py # get fig9.pdf
```

#### Figure 10

```bash
cd Figure10
./run.sh
python3 draw_fig10a.py # get fig10a.pdf
python3 draw_fig10b.py # get fig10b.pdf
```

#### Figure 11

```bash
cd Figure11
./run.sh
python3 draw_fig11.py # get fig11.pdf
```

#### Preprocessing graph

```bash
cd script
python3 cluster2.py arxiv # can replace arxiv with other dataset names
# the reorder file will be in data/arxiv_new_reorder_thres_0.2
```

### Publication

Kezhao Huang, Jidong Zhai, Zhen Zheng, Youngmin Yi, and Xipeng Shen. 2021. Understanding and Bridging the Gaps in Current GNN Performance Optimizations. In Proceedings of PPoPP ’21: 26rd ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, Virtual Event, Republic of Korea,  February 27–March 3, 2021 (PPoPP'21), 14 pages. https://doi.org/10.1145/3437801.3441585 


### Contact

If meet some problems, feel free to send E-mail to `hkz20@mails.tsinghua.edu.cn` and `xxcclong@gmail.com`, we will reply as soon as possible.
