dsets=(arxiv collab citation ddi protein ppa reddit.dgl products)


rm -f ./results/gat_our.log
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 python3 our.py --syn-name  ${dsets[i]} --gpu 0 --model our_GAT  2>&1 | tee -a ./results/gat_our.log
done    

rm -f ./results/gat_dgl.log
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 python3 our.py --syn-name  ${dsets[i]} --gpu 0 --model our_GCN  2>&1 | tee -a ./results/gcn_our.log
done


rm -f ./results/sage_our.log
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 ./fig7.out --dataset ${dsets[i]}_sample_16 --feature-len 32 --nei 16 2>&1 | tee -a ./results/sage_our.log
done


cat ./results/gcn_our.log | grep figure | awk '{print $6}' > ./results/our_gcn_results.log
cat ./results/gat_our.log | grep figure | awk '{print $6}' > ./results/our_gat_results.log
cat ./results/sage_our.log | ack "timing_our|Cuda failure" | awk '{print $9}' > ./results/our_sage_results.log
