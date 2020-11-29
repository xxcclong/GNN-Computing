#ifndef AGGREGATOR_H
#define AGGREGATOR_H

#define __shfl(a, b, c) __shfl_sync(0xffffffff, a, b, c)
#define __shfl_down(a, b) __shfl_down_sync(0xffffffff, a, b)

#include "util.h"
#include "data.h"
#include "graph_schedule.h"

__global__ void convertCSRToEdgelist(int *ptr, int *idx, int *edgelist, int num_v)
{
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= num_v)
        return;
    int begin = ptr[row], end = ptr[row + 1];
    for (int i = begin + lane; i < end; i += 32)
    {
        edgelist[i * 2] = idx[i];
        edgelist[i * 2 + 1] = row;
    }
}

class Aggregator
{
public:
    Aggregator(int *host_out_ptr, int *host_out_idx, int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in, int out_feat_out) : h_ptr(host_out_ptr), h_idx(host_out_idx), d_ptr(dev_out_ptr), d_idx(dev_out_idx), num_v(out_num_v), num_e(out_num_e), feat_in(out_feat_in), feat_out(out_feat_out)
    {
        if (h_ptr == NULL)
        {
            h_ptr = new int[num_v + 1];
            checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (h_idx == NULL)
        {
            h_idx = new int[num_e];
            checkCudaErrors(cudaMemcpy(h_idx, d_idx, (num_e) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        return;
    }
    Aggregator(CSRSubGraph g, int out_feat_in, int out_feat_out) : feat_in(out_feat_in), feat_out(out_feat_out)
    {
        d_ptr = g.ptr;
        d_idx = g.idx;
        num_v = g.num_v;
        num_e = g.num_e;
        h_ptr = new int[num_v + 1];
        h_idx = new int[num_e];
        checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_idx, d_idx, (num_e) * sizeof(int), cudaMemcpyDeviceToHost));
        d_vset = g.vertexset;
        // dbg(h_ptr[num_v]);
        // dbg(h_idx[num_e - 1]);
        // dbg(num_v);
        // dbg(num_e);
    }
    ~Aggregator()
    {
        safeFree(d_ptr);
        safeFree(d_idx);
        safeFree(d_ptr_scheduled);
        safeFree(d_idx_scheduled);
        safeFree(d_target_scheduled);
        safeFree(d_vset);
    }
    virtual void schedule(Schedule s, int *param)
    {
        std::vector<int> ptr_vec;
        std::vector<int> idx_vec;
        std::vector<int> target_vec;
        sche = s;
        safeFree(d_ptr_scheduled);
        safeFree(d_idx_scheduled);
        safeFree(d_target_scheduled);
        switch (s)
        {
        case locality:
            locality_schedule(h_ptr, h_idx, param[0], num_v, &ptr_vec, &idx_vec, &target_vec, n);
            locality_partition_num = param[0];
            break;
        case neighbor_grouping:
            neighbor_grouping_schedule(h_ptr, h_idx, param[0], num_v, num_e, &ptr_vec, &idx_vec, &target_vec);
            neighbor_group_size = param[0];
            break;
        case locality_neighbor_grouping:
            localityNeighborGrouping(h_ptr, h_idx, param[0], param[1], num_v, &ptr_vec, &idx_vec, &target_vec, n);
            locality_partition_num = param[0];
            neighbor_group_size = param[1];
            break;
        default:
            break;
        }
        num_target = target_vec.size();
        dbg(num_target);
        copyVec2Dev(&ptr_vec, d_ptr_scheduled);
        copyVec2Dev(&idx_vec, d_idx_scheduled);
        copyVec2Dev(&target_vec, d_target_scheduled);
    }
    virtual double run(float *vin, float *vout, int BLOCK_SIZE, bool scheduled)
    {
        assert(false);
        return -1;
    }
    virtual double run(float *v1, float *v2, float *outval, int BLOCK_SIZE, bool scheduled)
    {
        assert(false);
        return -1;
    }
    virtual double runEdgeWise(float *vin, float *vout, int BLOCK_SIZE, bool scheduled)
    {
        assert(false);
        return -1;
    }
    void csr2edgelist()
    {
        safeFree(d_edgelist);
        checkCudaErrors(cudaMalloc2((void **)&d_edgelist, 2 * num_e * sizeof(int)));
        int BLOCK_SIZE = 64;
        int target_in_block = BLOCK_SIZE / 32;
        convertCSRToEdgelist<<<(num_v + target_in_block - 1) / target_in_block, BLOCK_SIZE>>>(d_ptr, d_idx, d_edgelist, num_v);
    }

    int feat_in = 0;
    int feat_out = 0;
    int num_target = 0;

protected:
    int *d_ptr = NULL;
    int *d_idx = NULL;

    int *d_ptr_scheduled = NULL;
    int *d_idx_scheduled = NULL;
    int *d_target_scheduled = NULL;

    int *h_ptr = NULL;
    int *h_idx = NULL;

    int *d_vset = NULL;
    int *h_vset = NULL;

    int *d_edgelist = NULL;

    int num_v = 0;
    int num_e = 0;

    int neighbor_group_size = 0;
    int locality_partition_num = 0;

    Schedule sche = nop;
};
#endif
