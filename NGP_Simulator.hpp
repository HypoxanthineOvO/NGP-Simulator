#ifndef NGP_SIMULATOR_HPP
#define NGP_SIMULATOR_HPP


#include <vector>
#include <string>

#include "utils.hpp"

#include <camera.hpp>
#include <hash.hpp>
#include <sh.hpp>
#include <mlp.hpp>

class Simulator {
public:
    Simulator();
    Simulator(
        std::string Scene_Name,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<OccupancyGrid> occupancy_grid,
        std::shared_ptr<MLP> sig_mlp, std::shared_ptr<MLP> color_mlp,
        std::shared_ptr<HashEncoding> hash_encoding, std::shared_ptr<SHEncoding> sh_encoding,
        int MAX_T_COUNT = 1024
    );

    void loadParameters(std::string path);

    void render();
    void printHistory();

    void setSimulationFrequency(int frequency) {
        history.frequency = frequency;
    }
private:
    // Statistics
    struct History {
        std::string scene_name;
        int frequency; // Frequency of the simulation. MHz
        int cycleCount;
        std::vector<Vec3f> rgbs;
        std::vector<float> opacities;
    } history;

    // Process Variables
    int rayCount;
    int MAX_RAY_COUNT;
    int MAX_T_COUNT;

    // Simulation Units
    enum Stage {
        RAYMARCHING,
        HASHENCODING,
        SHENCODING,
        SIGMAMLP,
        COLORMLP,
        VOLUMERENDERING
    };
    enum ModuleState {
        WAIT_FOR_INPUT,
        DONE_AN_EXECUTION
    };
    enum ModuleState module_state[6] = {
        /* RAY MARCHING */ WAIT_FOR_INPUT,
        /* HASH ENCODING */ WAIT_FOR_INPUT,
        /* SH ENCODING */ WAIT_FOR_INPUT,
        /* SIGMA MLP */ WAIT_FOR_INPUT,
        /* COLOR MLP */ WAIT_FOR_INPUT,
        /* VOLUME RENDERING */ WAIT_FOR_INPUT
    };
    int latency[6] = {
        /* RAY MARCHING */ 1,
        /* HASH ENCODING */ 1,
        /* SH ENCODING */ 1,
        /* SIGMA MLP */ 1,
        /* COLOR MLP */ 1,
        /* VOLUME RENDERING */ 1
    };
    int waitCounter[6] = {
        /* RAY MARCHING */ 0,
        /* HASH ENCODING */ 0,
        /* SH ENCODING */ 0,
        /* SIGMA MLP */ 0,
        /* COLOR MLP */ 0,
        /* VOLUME RENDERING */ 0
    };
    void initialize();
    struct FeaturePool {
        // Ray Marching
        int rayID;
        int rayMarchingID;
        std::vector<int> valid_pixel;
        std::vector<int> t_count;
        std::vector<float> ts;
        
        // Hash Encoding
        int HashRayID;
        // SH Encoding
        int SHRayID;
        // Sigma MLP
        int SigmaRayID;
        // Color MLP
        int ColorRayID;
        // Volume Rendering
        int VolumeRayID;
        std::vector<Vec3f> colors;
        std::vector<float> opacities;
    } featurePool;

    // Note: All the fifo are input fifo.
    void init_valid_pixel();
    void rayMarching();
    std::shared_ptr<Camera> camera;
    std::shared_ptr<OccupancyGrid> occupancy_grid;
    struct ET_Data {
        int rayID;
    };
    FIFO<ET_Data> etFifo;
    void hashEncoding();
    std::shared_ptr<HashEncoding> hash_enc;
    struct Hash_in_Reg {
        int rayID;
        Vec3f input;
    };
    FIFO<Hash_in_Reg> hash_in_Fifo;
    struct Hash_out_Reg {
        int rayID;
        Vec32f output;
    };
    FIFO<Hash_out_Reg> hash_out_Fifo;
    void shEncoding();
    std::shared_ptr<SHEncoding> sh_enc;
    struct SH_in_Reg {
        int rayID;
        Vec3f input;
    };
    FIFO<SH_in_Reg> sh_in_Fifo;
    struct SH_out_Reg {
        int rayID;
        Vec16f output;
    };
    FIFO<SH_out_Reg> sh_out_Fifo;
    void sigmaMLP();
    std::shared_ptr<MLP> sig_mlp;
    struct SigMLP_in_Reg {
        int rayID;
        Vec32f input;
    };
    FIFO<SigMLP_in_Reg> sigmlp_in_Fifo;
    struct SigMLP_out_Reg {
        int rayID;
        Vec16f output;
    };
    FIFO<SigMLP_out_Reg> sigmlp_out_Fifo;
    void colorMLP();
    std::shared_ptr<MLP> col_mlp;
    struct Col_MLP_From_Hash {
        int rayID;
        Vec16f input;
    };
    FIFO<Col_MLP_From_Hash> colmlpFifo_Hash;
    struct Col_MLP_From_SH {
        int rayID;
        Vec16f input;
    };
    FIFO<Col_MLP_From_SH> colmlpFifo_SH;
    struct Col_MLP_out_Reg {
        int rayID;
        Vec4f output;
    };
    FIFO<Col_MLP_out_Reg> colmlp_out_Fifo;
    
    void volumeRendering();
    struct VR_in_Reg {
        int rayID;
        Vec4f input;
    };
    FIFO<VR_in_Reg> vr_in_Fifo;
    struct VR_out_Reg {
        int rayID;
    };
    FIFO<VR_out_Reg> vr_out_Fifo;
};



#endif // NGP_SIMULATOR_HPP