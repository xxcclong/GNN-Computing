dsets=(arxiv collab citation ddi protein ppa reddit.dgl products)
rm -f results/gcn_pyg.log
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 python3 pyg.py --dset ${dsets[i]} --gpu 0 --model GCN  2>&1 | tee -a results/gcn_pyg.log
done    

rm -f results/gat_pyg.log
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 python3 pyg.py --dset  ${dsets[i]} --gpu 0 --model GAT  2>&1 | tee -a results/gat_pyg.log
done
cat results/gcn_pyg.log | ack "figure|memory" | awk '{print $5}' > results/pyg_gcn_results.log
cat results/gat_pyg.log | ack "figure|memory" | awk '{print $5}' > results/pyg_gat_results.log
