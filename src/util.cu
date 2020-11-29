#include "../include/util.h"

int GPUNUM = 1;
int NEINUM = -1;

// ncclComm_t* comms = new ncclComm_t[GPUNUM];

int n = -1, m = -1, feature_len = 0;
int *rows = NULL;
int *reverse_rows = NULL;
vector<void *> registered_ptr;
int outfea = 0;
int total_size = 0;
string inputfeature = "";
string inputweight = "";
string inputgraph = "";
string inputtransgraph = "";
string partialgraphs = "";
string edgefile = "";
string ptrfile = "";
string partitionfile = "";
string reorderfile = "";

void argParse(int argc, char ** argv, int* p_limit, int* p_limit2)
{
    args::ArgumentParser parser("GNN parameters", "");
    args::ValueFlag<string> arg_dataset(parser, "dataset", "", {"dataset"});
    args::ValueFlag<string> arg_datadir(parser, "datadir", "", {"datadir"});
    args::ValueFlag<string> arg_partition(parser, "partition-path", "", {"partition-path"});
    args::ValueFlag<string> arg_reorder(parser, "reorder", "", {"reorder"});

    args::ValueFlag<int> arg_gpunum(parser, "gpu-num", "", {"gpu-num"});
    args::ValueFlag<int> arg_neinum(parser, "nei", "", {"nei"});
    // args::ValueFlag<int> arg_vertexnum(parser, "vertex-num", "", {"vertex-num"});
    // args::ValueFlag<int> arg_edgenum(parser, "edge-num", "", {"edge-num"});
    args::ValueFlag<int> arg_featurelen(parser, "feature-len", "", {"feature-len"});
    args::ValueFlag<int> arg_outfea(parser, "outfea", "", {"outfea"});

    args::ValueFlag<int> arg_limit(parser, "limit", "", {"limit"});
    args::ValueFlag<int> arg_limit2(parser, "limit2", "", {"limit2"});
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        exit(0);
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }
    string ddir = "";
    if(bool{arg_datadir})
        ddir = args::get(arg_datadir);
    else
        ddir = "../data/";
    assert(bool{arg_dataset});
    string dset = args::get(arg_dataset);
    inputgraph = ddir + dset + ".graph";
    ptrfile = inputgraph + ".ptrdump";
    edgefile = inputgraph + ".edgedump";
    string configpath = ddir + dset + ".config";
    assert(fexist(configpath));
    FILE *fin(fopen(configpath.c_str(), "r"));
    fscanf(fin, "%d", &n);
    fscanf(fin, "%d", &m);
    fclose(fin);
    if(fexist(inputgraph))
    {
        if(!fexist(ptrfile)) ptrfile = "";
        if(!fexist(edgefile)) edgefile = "";
    }
    else
    {
        assert(fexist(ptrfile));
        assert(fexist(edgefile));
    }
    if(bool{arg_partition}) 
    {
        partitionfile = args::get(arg_partition);
        assert(fexist(partitionfile));
    }
    assert(bool{arg_featurelen});
    feature_len = args::get(arg_featurelen);
    if(bool{arg_outfea})
    {
        outfea = args::get(arg_outfea);
    }
    if(bool{arg_gpunum}) GPUNUM = args::get(arg_gpunum);
    if(bool{arg_neinum}) NEINUM = args::get(arg_neinum);
    reorderfile = ddir + dset + ".reorder";
    //assert(!(bool{arg_reorder} && !fexist(reorderfile)));
    std::string tmpreorder = "";// = args::get(arg_reorder);
    if(bool{arg_reorder})
    {
        tmpreorder = args::get(arg_reorder);
    }
    if(strlen(tmpreorder.c_str()) > 1)
    {
        reorderfile += tmpreorder;
    }
    if(bool{arg_reorder})
    {
        assert(fexist(reorderfile));
    }
    else
    {
        reorderfile = "";
    }
    
    if(p_limit != NULL)
    {
        assert(bool{arg_limit});
        *p_limit = args::get(arg_limit);
    }

    if(p_limit2 != NULL)
    {
        assert(bool{arg_limit2});
        *p_limit2 = args::get(arg_limit2);
    }
    dbg(dset);
    inputgraph = dset;
    // fprintf(stderr, 
    //     "*****************************************\n"
    //     "dataset: %s\n"
    //     "graphdir: %s\n"
    //     "ptrdir: %s\n"
    //     "edgedir: %s\n"
    //     "partitionfile: %s\n"
    //     "GPUnum: %d\n"
    //     "n: %d\n"
    //     "m: %d\n"
    //     "feature len: %d\n"
    //     "*****************************************\n"
    //     ,dset.c_str(), inputgraph.c_str(), ptrfile.c_str(), edgefile.c_str(), partitionfile.c_str(), GPUNUM, n, m, feature_len);
}


// ************************************************************
// variables for single train
int *gptr, *gidx;
float *gval;

// layer2

cublasHandle_t cublasH = NULL;
cusparseHandle_t cusparseH = NULL;
cudaStream_t stream = NULL;

// cudnn

// ************************************************************
// var for multi train
int *numVertex = new int[GPUNUM], *numEdge = new int[GPUNUM];

int **gptrs, **gidxs;
float **gvals;


int **trans_gptrs, **trans_gidxs;
float **trans_gvals;


cublasHandle_t* cublasHs = new cublasHandle_t[GPUNUM];
cusparseHandle_t* cusparseHs = new cusparseHandle_t[GPUNUM];
cudaStream_t* streams = new cudaStream_t[GPUNUM];

