dsets=(arxiv collab citation ddi protein ppa reddit.dgl products)

rm -f fig9_output
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 nvprof --kernels "aggr_gcn_target|aggr_gcn"  --metrics "l2_tex_hit_rate"  ./fig9.out --dataset ${dsets[i]} --feature-len 32 --reorder _thres_0.2  2>&1 | ack -i "l2_tex_hit_rate|kernel" | tee -a fig9_output 
done
cat fig9_output | grep "hit_rate" | awk '{gsub("\%+","");print $9}' | sed -n 'n;p' > results/fig9_LAS.log
cat fig9_output | grep "hit_rate" | awk '{gsub("\%+","");print $9}' | sed -n 'p;n' > results/fig9_NG_LAS.log

for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 nvprof --kernels "aggr_gcn_target"  --metrics "l2_tex_hit_rate"  ./fig9.out --dataset ${dsets[i]} --feature-len 32   2>&1 | ack -i "l2_tex_hit_rate"  | awk '{gsub("\%+","");print $9}' | tee -a results/fig9_NG.log
done

rm -f fig9_output_best_prior
for i in `seq 0 7`;
do
nvprof --kernels "csrMmt_hyb_core" --metrics "l2_tex_hit_rate"  python3 dgl_prof_gcn.py --gpu 0 --dset ${dsets[i]}  2>&1 | grep "l2_tex_hit_rate" | tee -a fig9_output_best_prior
done

cat fig9_output_best_prior | grep "hit_rate" | awk '{gsub("\%+","");print $9}' > results/fig9_best_prior.log
