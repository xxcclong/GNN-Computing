dsets=(arxiv collab citation ddi protein ppa reddit.dgl products)

rm -f output_figure11
for i in `seq 0 7`;
do
    CUDA_VISIBLE_DEVICES=0 ./fig11.out --dataset ${dsets[i]}_sample_16 --feature-len 32 --nei 16  2>&1  | tee -a output_figure11
done

cat ./output_figure11 | grep "timing_dgl" | awk '{print $9}' > results/fig11_base.log
cat ./output_figure11 | grep "timing_sparse" | awk '{print $9}' > results/fig11_SF.log
cat ./output_figure11 | grep "timing_our" | awk '{print $9}' > results/fig11_SF_RE.log
