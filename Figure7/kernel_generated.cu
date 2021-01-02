#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/util.h"
#include <vector>

#include "../include/aggr_gcn.h"
#include "../include/aggr_gat.h"
#define ERROR_CODE (-1)


int64_t GCN_init_impl(int* ptr, int* idx, float* val, int num_v, int num_e)
{
    Aggregator *at = new Aggregator_GCN(NULL, NULL, ptr, idx, num_v, num_e, 32, 32, val);
    return (int64_t)at;
}

void GCN_update_val_impl(int64_t at, float *val)
{
	((Aggregator_GCN *)at)->updateval(val);
}

void GCN_run_impl(int64_t at, float *feat, float *out_feat, int blocksize, int scheduled, int featlen)
{
	// dbg(featlen);
	((Aggregator_GCN *)at)->run_with_feat(feat, out_feat, blocksize, scheduled, featlen);
	// dbg(featlen);
	return;
}

void GCN_schedule_impl(int64_t at, int *arr)
{
	dbg(arr[0]);
	((Aggregator_GCN *)at)->schedule(neighbor_grouping, arr);
	return;
}

int64_t GAT_init_impl(int* ptr, int* idx, int num_v, int num_e)
{
    Aggregator *at = new Aggregator_GAT(NULL, NULL, ptr, idx, num_v, num_e, 32, 32);
    return (int64_t)at;
}

void GAT_run_impl(int64_t at, float *feat, float* att, float *out_feat, int blocksize, int scheduled, int featlen)
{
	((Aggregator_GAT *)at)->run_with_feat(feat, att, out_feat, blocksize, scheduled, featlen);
}

void GAT_run_u_add_v_impl(int64_t at, float* att, float* outval, int blocksize)
{
	((Aggregator_GAT *)at)->run_u_add_v(att, outval, blocksize);
}

void GAT_run_add_to_center_impl(int64_t at, float* inval, float* outatt, int blocksize)
{
	((Aggregator_GAT *)at)->run_add_to_center(inval, outatt, blocksize);
}

void GAT_run_div_each_impl(int64_t at, float* inatt, float* inoutval, int blocksize)
{
	((Aggregator_GAT *)at)->run_div_each(inatt, inoutval, blocksize);
}



void GAT_schedule_impl(int64_t at, int *arr)
{
	dbg(arr[0]);
    ((Aggregator_GAT *)at)->schedule(neighbor_grouping, arr);
    return;
}
