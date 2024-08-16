#ifndef NGP_SIMULATOR_HPP
#define NGP_SIMULATOR_HPP


#include <vector>
#include "utils.hpp"

#include <camera.hpp>
#include <hash.hpp>
#include <sh.hpp>
#include <mlp.hpp>

class Simulator {
public:
    Simulator();
    Simulator(
        std::shared_ptr<Camera> camera,
        std::shared_ptr<OccupancyGrid> occupancy_grid,
        std::shared_ptr<MLP> sig_mlp, std::shared_ptr<MLP> color_mlp,
        std::shared_ptr<HashEncoding> hash_encoding, std::shared_ptr<SHEncoding> sh_encoding
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
        int frequency; // Frequency of the simulation. MHz
        int cycleCount;
        std::vector<Vec3f> rgbs;
        std::vector<float> opacities;
    } history;

    // Process Variables
    int rayCount;
    int MAX_RAY_COUNT;

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
        /* HASH ENCODING */ 10,
        /* SH ENCODING */ 4,
        /* SIGMA MLP */ 4,
        /* COLOR MLP */ 6,
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
        int rayMarchingID;
        float now_t;
        Vec3f HashInput;
        Vec3f SHInput;
        
        // Hash Encoding
        int HashRayID;
        Vec32f HashOutput;
        
        // SH Encoding
        int SHRayID;
        Vec16f SHOutput;
        
        // Sigma MLP
        int SigmaRayID;
        Vec16f SigmaOutput;
        
        // Color MLP
        int ColorRayID;
        Vec4f ColorOutput; // RGB + Alpha, noticed rgb and alpha are all raw data.
        // Volume Rendering
        int VolumeRayID;
        Vec3f color;
        float opacity;
    } featurePool;

    // Note: All the fifo are input fifo.
    void rayMarching();
    // Strategy
    
    std::shared_ptr<Camera> camera;
    std::shared_ptr<OccupancyGrid> occupancy_grid;
    struct RM_Reg {
        int rayID;
    };
    FIFO<RM_Reg> rmFifo;
    void hashEncoding();
    std::shared_ptr<HashEncoding> hash_enc;
    struct Hash_Reg {
        int rayID;
        Vec3f input;
    };
    FIFO<Hash_Reg> hashFifo;
    void shEncoding();
    std::shared_ptr<SHEncoding> sh_enc;
    struct SH_Reg {
        int rayID;
        Vec3f input;
    };
    FIFO<SH_Reg> shFifo;
    void sigmaMLP();
    std::shared_ptr<MLP> sig_mlp;
    struct SigMLP_Reg {
        int rayID;
        Vec32f input;
    };
    FIFO<SigMLP_Reg> sigmlpFifo;
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
    
    void volumeRendering();
    struct VR_Reg {
        int rayID;
        Vec4f input;
    };
    FIFO<VR_Reg> vrFifo;
};



#endif // NGP_SIMULATOR_HPP