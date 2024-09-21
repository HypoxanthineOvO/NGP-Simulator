#include "NGP_Simulator.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>


std::string PATH = "./configs/base.json";
int RESOLITION = 800;
std::string NAME = "lego";
std::string DATA_PATH;
int ID = 0;
int FREQUENCY = 100;
int max_t_count = 1024;

int main(int argc, char** argv) {
    std::cout << "Running Scene " << NAME << std::endl;

    nlohmann::json configs, camera_configs;

    if (argc > 1) {
        NAME = argv[1];
    }
    if (argc > 2) {
        FREQUENCY = std::stoi(argv[2]);
    }
    if (argc > 3) {
        max_t_count = std::stoi(argv[3]);
    }
    
    std::ifstream fin;
    fin.open(PATH);
    fin >> configs;
    fin.close();
    printf("Read Configs from [%s]\n", PATH.c_str());

    DATA_PATH = "./data/nerf_synthetic/" + NAME + "/" + "transforms_test.json";
    fin.open(DATA_PATH);
    fin >> camera_configs;
    fin.close();
    puts("Cameara Configs Loaded");

    /* Generate Camera and Image */
    std::shared_ptr<Image> img = 
        std::make_shared<Image>(RESOLITION, RESOLITION);
    std::shared_ptr<Camera> camera =
        std::make_shared<Camera>(
            camera_configs, img, ID
        );

    /* Generate NGP Runner */
    std::shared_ptr<OccupancyGrid> ocgrid =
        std::make_shared<OccupancyGrid>(
            128, -0.5, 1.5
        );
    // Two MLP
    std::shared_ptr<MLP> sigma_mlp =
        std::make_shared<MLP>(
            32, 16, configs.at("network")
        );
    std::shared_ptr<MLP> color_mlp =
        std::make_shared<MLP>(
            32, 16, configs.at("rgb_network")
        );
    // Hash Encoding
    std::shared_ptr<HashEncoding> hashenc =
        std::make_shared<HashEncoding>(
            configs.at("encoding")
        );
    // SH Encoding
    std::shared_ptr<SHEncoding> shenc =
        std::make_shared<SHEncoding>(
            configs.at("dir_encoding").at("nested")[0]
        );
    
    Simulator sim(
        NAME, camera, ocgrid, sigma_mlp, color_mlp, hashenc, shenc, max_t_count
    );
    sim.setSimulationFrequency(FREQUENCY);
    sim.loadParameters("./snapshots/Hash19_Float/" + NAME + ".msgpack");
    sim.render();
    sim.printHistory();
    return 0;
}