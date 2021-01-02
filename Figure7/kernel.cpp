#include <torch/extension.h>

#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/util.h"
#include "../include/data.h"



int64_t GCN_init_impl(int *ptr, int *idx, float *val, int num_v, int num_e);

void GCN_update_val_impl(int64_t at, float *val);

void GCN_run_impl(int64_t at, float *feat, float *out_feat, int blocksize, int scheduled, int featlen);

void GCN_schedule_impl(int64_t at, int *arr);

int64_t GAT_init_impl(int *ptr, int *idx, int num_v, int num_e);

void GAT_run_impl(int64_t at, float *feat, float *att, float *out_feat, int blocksize, int scheduled, int featlen);

void GAT_run_u_add_v_impl(int64_t at, float* att, float* outval, int blocksize);

void GAT_run_add_to_center_impl(int64_t at, float* inval, float* outatt, int blocksize);

void GAT_run_div_each_impl(int64_t at, float* inatt, float* inoutval, int blocksize);



void GAT_schedule_impl(int64_t at, int *arr);

std::vector<torch::Tensor> new_load(std::string dset, std::string reorder = "", int devid = 0)
{
  int num_v = 1, num_e = 1;
  int *indptr = NULL, *indices = NULL;
  load_graph(dset, num_v, num_e, indptr, indices, 1, reorder);
  auto options =
      torch::TensorOptions()
          .dtype(torch::kInt32)
          // .device(torch::kCUDA)
          .device(torch::kCUDA, devid)
          .requires_grad(false);
  dbg(num_v);
  dbg(num_e);
  dbg(reorder);

  n = num_v;
  m = num_e;

  auto ptrs = torch::zeros({num_v + 1}, options);
  auto idxs = torch::zeros({num_e}, options);
  assert(indptr != NULL);
  dbg(indptr[1]);
  dbg(indices[1]);
  checkCudaErrors(cudaMemcpy(ptrs.data<int>(), indptr, (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(idxs.data<int>(), indices, (num_e) * sizeof(int), cudaMemcpyHostToDevice));
  if (indptr != NULL)
    delete[] indptr;
  if (indices != NULL)
    delete[] indices;
  return {ptrs, idxs};
}

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


int64_t GCN_init(
    torch::Tensor ptrs,
    torch::Tensor idxs,
    torch::Tensor val)
{
  return GCN_init_impl(ptrs.data<int>(), idxs.data<int>(), val.data<float>(), ptrs.size(0) - 1, idxs.size(0));
}

void GCN_update_val(
    int64_t at,
    torch::Tensor val)
{
  GCN_update_val_impl(at, val.data<float>());
}

void GCN_run(
    int64_t at,
    torch::Tensor feat,
    torch::Tensor outfeat,
    int blocksize,
    int scheduled)
{
  GCN_run_impl(at, feat.data<float>(), outfeat.data<float>(), blocksize, scheduled, feat.size(1));
}

void GCN_schedule(
    int64_t at,
    int neighbor_num)
{
  int arr[] = {neighbor_num};
  GCN_schedule_impl(at, arr);
  return;
}

int64_t GAT_init(
    torch::Tensor ptrs,
    torch::Tensor idxs)
{
  return GAT_init_impl(ptrs.data<int>(), idxs.data<int>(), ptrs.size(0) - 1, idxs.size(0));
}

void GAT_run(
    int64_t at,
    torch::Tensor feat,
    torch::Tensor att,
    torch::Tensor outfeat,
    int blocksize,
    int scheduled)
{
  GAT_run_impl(at, feat.data<float>(), att.data<float>(), outfeat.data<float>(), blocksize, scheduled, feat.size(1));
}

void GAT_run_u_add_v(
    int64_t at,
    torch::Tensor att,
    torch::Tensor outval,
    int blocksize)
{
  GAT_run_u_add_v_impl(at, att.data<float>(), outval.data<float>(), blocksize);
}

void GAT_run_add_to_center(
    int64_t at,
    torch::Tensor inval,
    torch::Tensor outatt,
    int blocksize)
{
  GAT_run_add_to_center_impl(at, inval.data<float>(), outatt.data<float>(), blocksize);
}

void GAT_run_div_each(
    int64_t at,
    torch::Tensor inatt,
    torch::Tensor inoutval,
    int blocksize)
{
  GAT_run_div_each_impl(at, inatt.data<float>(), inoutval.data<float>(), blocksize);
}

void GAT_schedule(
    int64_t at,
    int neighbor_num)
{
  int arr[] = {neighbor_num};
  GAT_schedule_impl(at, arr);
  return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("new_load", &new_load, "new_load");
  m.def("gcn_init", &GCN_init, "gcn_init");
  m.def("gcn_update_val", &GCN_update_val, "gcn_update_val");
  m.def("gcn_run", &GCN_run, "gcn_run");
  m.def("gcn_schedule", &GCN_schedule, "gcn_schedule");
  m.def("gat_init", &GAT_init, "gat_init");
  m.def("gat_run", &GAT_run, "gat_run");
  m.def("gat_schedule", &GAT_schedule, "gat_schedule");
  m.def("gat_run_u_add_v", &GAT_run_u_add_v, "gat_run_u_add_v");
  m.def("gat_run_add_to_center", &GAT_run_add_to_center, "gat_run_add_to_center");
  m.def("gat_run_div_each", &GAT_run_div_each, "gat_run_div_each");
}
