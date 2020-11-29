#include "../include/data.h"

// the map[i]^th node should be placed in the i^th place 
void reorderCSR(const int *ptr, const int *idx, const int *map, const int *reverse_map, int num_v, int num_e, int*& newptr, int*& newidx)
{
    dbg(num_v);
    dbg(num_e);
    if(newptr == NULL) 
    {
        dbg("allocate");
        newptr = new int[num_v + 1];
    }
    if(newidx == NULL) newidx = new int[num_e];

    newptr[0] = 0;
    int begin = 0;
    for(int i = 0; i < num_v; ++i)
    {
        int range = ptr[map[i] + 1] - ptr[map[i]];
        int base = ptr[map[i]];
        for(int j = 0; j < range; ++j)
        {
            newidx[begin + j] = reverse_map[idx[base + j]];
        }
        begin += range;
        newptr[i + 1] = begin;
    }
    dbg("finish allocate");
}

void load_graph(std::string dset, int &num_v, int &num_e, int* &indptr, int* &indices, bool shuffle, std::string reorder_subfix)
{
    dbg("loading");
    std::string basedir = "../data/";
    auto inputgraph = basedir + dset + ".graph";
    auto ptrfile = inputgraph + ".ptrdump";
    auto edgefile = inputgraph + ".edgedump";
    auto configpath = basedir + dset + ".config";

    assert(fexist(configpath));
    FILE *fin(fopen(configpath.c_str(), "r"));
    fscanf(fin, "%d", &num_v);
    fscanf(fin, "%d", &num_e);
    fclose(fin);

    indptr = new int[num_v + 1];
    indices = new int[num_e];

    // ptr
    if(fexist(ptrfile))
    {
        FILE *f1(fopen(ptrfile.c_str(), "r"));
        fread(indptr, (num_v + 1) * sizeof(int), 1, f1);
        fclose(f1);
    }
    else
    {
        dbg("reading non-processed file");
        FILE *tmpfin(fopen(inputgraph.c_str(), "r"));
        fin = tmpfin;
        for (int i = 0; i < num_v + 1; ++i) {
            fscanf(fin, "%d", indptr + i);
        }
        ptrfile = inputgraph + ".ptrdump";
        FILE *f2(fopen(ptrfile.c_str(), "w"));
        fwrite((void*)indptr, (num_v + 1) * sizeof(int), 1, f2);
        fclose(f2);
    }
    if(indptr[num_v] != num_e)
    {
        dbg(indptr[num_v]);
        dbg(num_e);
        assert(indptr[num_v] == num_e);
    }

    // idx
    if(fexist(edgefile))
    {
        FILE *f2(fopen(edgefile.c_str(), "r"));
        fread(indices, num_e * sizeof(int), 1, f2);
        fclose(f2);
    }
    else
    {
        for (int i = 0; i < num_e; ++i) {
            fscanf(fin, "%d", indices + i);
        }
        edgefile = inputgraph + ".edgedump";
        FILE *f2(fopen(edgefile.c_str(), "w"));
        fwrite((void*)indices, num_e * sizeof(int), 1, f2);
        fclose(f2);
        fclose(fin);
    }

    // std::string reorderfile = basedir + dset + ".reorder_thres_0.2";
    if (strlen(reorder_subfix.c_str()) > 0)
        reorderfile = basedir + dset + ".reorder" + reorder_subfix;
    if(shuffle && strlen(reorderfile.c_str()) > 1 && fexist(reorderfile))
    {
        dbg("reorder:" + reorderfile);
        FILE *tmpfin(fopen(reorderfile.c_str(), "r"));
        int* newptr = NULL;//new int[num_v + 1];
        int* newidx = NULL;//new int[num_e];

        int row = 0;
        rows = new int[num_v];
        reverse_rows = new int[num_v];
        for(int i = 0; i < num_v; ++i)
        {
            fscanf(tmpfin, "%d", &row);
            rows[i] = row;
            reverse_rows[row] = i;
        }
        reorderCSR(indptr, indices, rows, reverse_rows, num_v, num_e, newptr, newidx);
        // for(int i = 0; i < n; ++i)
        // {
        //     int range = indptr[rows[i] + 1] - indptr[rows[i]];
        //     int base = indptr[rows[i]];
        //     for(int j = 0; j < range; ++j)
        //     {
        //         newidx[begin + j] = reverse_rows[indices[base + j]];
        //     }
        //     begin += range;
        //     newptr[i + 1] = begin;
        // }
        // delete[] rows;
        // delete[] reverse_rows;
        
        delete[] indptr;
        delete[] indices;
        indptr = newptr;
        indices = newidx;
    }
    else
    {
        dbg("unreordered");
        dbg(reorderfile);
    }
}
