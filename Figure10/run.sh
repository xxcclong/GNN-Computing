dsets=(arxiv collab citation ddi protein ppa reddit.dgl products)

rm -f output_figure10a
rm -f output_figure10a_base
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 ./fig10a.out --dataset ${dsets[i]} --feature-len 32 --nei 32  2>&1  | tee -a output_figure10a_base
    CUDA_VISIBLE_DEVICES=0 ./fig10a.out --dataset ${dsets[i]} --feature-len 32 --nei 32 --reorder _thres_0.2 2>&1  | tee -a output_figure10a
done

rm -f output_figure10b
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 ./fig10b.out --dataset ${dsets[i]} --feature-len 32 --outfea 32 --nei 64  2>&1 | tee -a ./output_figure10b
done

cat ./output_figure10a_base | grep "t_base" | awk '{print $8}' > results/fig10a_base.log
cat ./output_figure10a | grep "t_adapter" | awk '{print $8}' > results/fig10a_base_adapter.log
cat ./output_figure10a | grep "t_linear"  | awk '{print $8}' > results/fig10a_base_adapter_linear.log

cat ./output_figure10b | grep "t_base"  | awk '{print $8}' > results/fig10b_base.log
cat ./output_figure10b | grep "t_linear"  | awk '{print $8}' > results/fig10b_base_adapter_linear.log
