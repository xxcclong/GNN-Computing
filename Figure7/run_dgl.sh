dsets=(arxiv collab citation ddi protein ppa reddit.dgl products)

rm -f results/gcn_dgl.log
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 python3 dgl_prof.py --dataset syn --syn-name  ${dsets[i]} --gpu 0 --model GCN  2>&1 | tee -a results/gcn_dgl.log
done    

rm -f results/gat_dgl.log
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 python3 dgl_prof.py --dataset syn --syn-name  ${dsets[i]} --gpu 0 --model GAT  2>&1 | tee -a results/gat_dgl.log
done

rm -f results/sage_dgl.log
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 ./fig7.out --dataset ${dsets[i]}_sample_16 --feature-len 32 --nei 16 2>&1 | tee -a results/sage_dgl.log
done

cat results/gcn_dgl.log | grep figure | awk '{print $5}' > results/dgl_gcn_results.log
cat results/gat_dgl.log | grep figure | awk '{print $5}' > results/dgl_gat_results.log
cat results/sage_dgl.log | ack "timing_dgl|Cuda failure" | awk '{print $9}' > results/dgl_sage_results.log
