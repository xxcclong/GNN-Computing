dsets=(arxiv collab citation ddi protein ppa reddit.dgl products)
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 ./fig8.out --dataset  ${dsets[i]} --feature-len 32  --reorder _thres_0.2  2>&1 # | grep "hkz_" | awk '{print $5}'
done    
