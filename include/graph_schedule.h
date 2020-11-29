#ifndef GRAPH_SCHEDULE_H
#define GRAPH_SCHEDULE_H
#include <vector>
#include "util.h"

// #define ANALYSE_RESULT

enum Schedule
{
    locality,
    neighbor_grouping,
    locality_neighbor_grouping,
    nop
};

// reorder the computing sequence is needed, so need to pass in **val**, to reorder everything needed
void locality_schedule(int *ptr, int *idx, int par_num, int num_v, std::vector<int> *ptr_vec, std::vector<int> *idx_vec, std::vector<int> *target_vec, int total_num_v, float *val = NULL, std::vector<float> *val_vec = NULL)
{
#ifdef ANALYSE_RESULT
    vector<int> task_size_vec;
#endif
    timestamp(t0);
    ptr_vec->push_back(0);
    for (int par = 0; par < par_num; ++par)
    {
        int llim = par * (total_num_v / par_num);
        int ulim = llim + total_num_v / par_num;
        if (par == par_num - 1)
            ulim = total_num_v;
        for (int i = 0; i < num_v; ++i)
        {
            int begin = ptr[i];
            int end = ptr[i + 1];
            int cnt = 0;
            for (int j = begin; j < end; ++j)
            {
                if (idx[j] >= llim && idx[j] < ulim)
                {
                    cnt++;
                    idx_vec->push_back(idx[j]);
                    if (val != NULL)
                        val_vec->push_back(val[j]);
                }
                // else if (idx[j] >= ulim)
                // {
                //     if (cnt != 0)
                //     {
                //         ptr_vec->push_back(ptr_vec->back() + cnt);
                //         target_vec->push_back(i);
                //     }
                //     break;
                // }
            }
            if (cnt != 0)
            {
                ptr_vec->push_back(ptr_vec->back() + cnt);
                target_vec->push_back(i);
#ifdef ANALYSE_RESULT
                task_size_vec.push_back(cnt);
#endif
            }
        }
    }
    timestamp(t1);
    double locality_schedule_time = getDuration(t0, t1);
    dbg(locality_schedule_time);
#ifdef ANALYSE_RESULT
    int under_utilization = 0;
    int under_u_aver = 0;
    int mmax = 0;
    int mmin = 999999999;
    for (auto item : task_size_vec)
    {
        if (item < 32)
            ++under_utilization;
        if (item > mmax)
            mmax = item;
        if (item < mmin)
            mmin = item;
    }
    dbg(under_utilization);
    dbg(((float)under_u_aver) / under_utilization);
    dbg(mmax);
    dbg(mmin);
    dbg(target_vec->size());
    dbg(num_v);
    dbg(ptr_vec->back());
#endif
}

void neighbor_grouping_schedule(int *ptr, int *idx, int neighbor_num, int num_v, int num_e, std::vector<int> *ptr_vec, std::vector<int> *idx_vec, std::vector<int> *target_vec)
{
    assert(ptr != NULL);
    assert(idx != NULL);
#ifdef ANALYSE_RESULT
    vector<int> task_size_vec;
#endif
    timestamp(t0);
    ptr_vec->push_back(0);
    for (int i = 0; i < num_v; ++i)
    {
        int left = ptr[i];
        while (ptr[i + 1] - left > neighbor_num)
        {
            left += neighbor_num;
            ptr_vec->push_back(left);
            target_vec->push_back(i);
#ifdef ANALYSE_RESULT
            task_size_vec.push_back(neighbor_num);
#endif
        }
        if (ptr[i + 1] != left)
        {
#ifdef ANALYSE_RESULT
            task_size_vec.push_back(ptr[i + 1] - ptr_vec->back());
#endif
            ptr_vec->push_back(ptr[i + 1]);
            target_vec->push_back(i);
        }
    }
    dbg(target_vec->size());
    dbg(num_e);
    idx_vec->resize(num_e);
    memcpy(&((*idx_vec)[0]), idx, num_e * sizeof(int));
    timestamp(t1);
    double neighbor_grouping_schedule_time = getDuration(t0, t1);
    dbg(neighbor_grouping_schedule_time);
#ifdef ANALYSE_RESULT
    int under_utilization = 0;
    int under_u_aver = 0;
    int zero = 0;
    int mmax = 0;
    int mmin = 999999999;
    for (auto item : task_size_vec)
    {
        if (item < 32)
            ++under_utilization;
        if (item > mmax)
            mmax = item;
        if (item < mmin)
            mmin = item;
        if (item == 0)
            ++zero;
    }
    dbg(under_utilization);
    dbg(((float)under_u_aver) / under_utilization);
    dbg(mmax);
    dbg(mmin);
    dbg(target_vec->size());
    dbg(num_v);
    dbg(ptr_vec->back());
    dbg(zero);
#endif
}

void localityNeighborGrouping(int *ptr, int *idx, int par_num, int neighbor_num, int num_v, std::vector<int> *ptr_vec, std::vector<int> *idx_vec, std::vector<int> *target_vec, int total_num_v, float *val = NULL, std::vector<float> *val_vec = NULL)
{
#ifdef ANALYSE_RESULT
    vector<int> task_size_vec;
#endif
    timestamp(t0);
    ptr_vec->push_back(0);
    for (int par = 0; par < par_num; ++par)
    {
        int llim = par * (total_num_v / par_num);
        int ulim = llim + total_num_v / par_num;
        if (par == par_num - 1)
            ulim = total_num_v;
        for (int i = 0; i < num_v; ++i)
        {
            int begin = ptr[i];
            int end = ptr[i + 1];
            int cnt = 0;
            for (int j = begin; j < end; ++j)
            {
                if (idx[j] >= llim && idx[j] < ulim)
                {
                    cnt++;
                    idx_vec->push_back(idx[j]);
                    if (val != NULL)
                        val_vec->push_back(val[j]);
                    if (cnt == neighbor_num)
                    {
                        ptr_vec->push_back(ptr_vec->back() + cnt);
                        target_vec->push_back(i);
#ifdef ANALYSE_RESULT
                        task_size_vec.push_back(cnt);
#endif
                        cnt = 0;
                    }
                }
                // else if (idx[j] >= ulim)
                // {
                //     if (cnt != 0)
                //     {
                //         ptr_vec->push_back(ptr_vec->back() + cnt);
                //         target_vec->push_back(i);
                //     }
                //     break;
                // }
            }
            if (cnt != 0)
            {
                ptr_vec->push_back(ptr_vec->back() + cnt);
                target_vec->push_back(i);
#ifdef ANALYSE_RESULT
                task_size_vec.push_back(cnt);
#endif
            }
        }
    }
    timestamp(t1);
    dbg(getDuration(t0, t1));
#ifdef ANALYSE_RESULT
    int under_utilization = 0;
    int zero = 0;
    int mmax = 0;
    int mmin = 999999999;
    int under_u_aver = 0;
    for (auto item : task_size_vec)
    {
        if (item < 32)
        {
            ++under_utilization;
            under_u_aver += item;
        }
        if (item > mmax)
            mmax = item;
        if (item < mmin)
            mmin = item;
        if (item == 0)
            ++zero;
    }
    dbg(under_utilization);
    dbg(((float)under_u_aver) / under_utilization);
    dbg(mmax);
    dbg(mmin);
    dbg(target_vec->size());
    dbg(num_v);
    dbg(ptr_vec->back());
    dbg(zero);
#endif
}

#endif