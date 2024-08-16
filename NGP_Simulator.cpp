#include "NGP_Simulator.hpp"
#include <iostream>

Simulator::Simulator(): rayCount(0), MAX_RAY_COUNT(1),
 camera(nullptr), occupancy_grid(nullptr),
hash_enc(nullptr), sh_enc(nullptr), sig_mlp(nullptr), col_mlp(nullptr) {
    // Initialize Statistics
    history.scene_name = "Unknown";
    history.cycleCount = 0;
    history.frequency = 1;
}

Simulator::Simulator(
        std::string Scene_Name,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<OccupancyGrid> occupancy_grid,
        std::shared_ptr<MLP> sig_mlp, std::shared_ptr<MLP> color_mlp,
        std::shared_ptr<HashEncoding> hash_encoding, std::shared_ptr<SHEncoding> sh_encoding
):
    rayCount(0), camera(camera), occupancy_grid(occupancy_grid),
    sig_mlp(sig_mlp), col_mlp(color_mlp),
    hash_enc(hash_encoding), sh_enc(sh_encoding)
    {
        MAX_RAY_COUNT = camera->getResolution().x() * camera->getResolution().y();

        // Initialize Statistics
        history.scene_name = Scene_Name;
        history.cycleCount = 0;
        history.rgbs.resize(camera->getResolution().x() * camera->getResolution().y());
        history.opacities.resize(camera->getResolution().x() * camera->getResolution().y());
        history.frequency = 1;
    }

void Simulator::loadParameters(std::string path) {
    using namespace nlohmann;
    std::ifstream input_msgpack_file(path, std::ios::in | std::ios::binary);
    json data = json::from_msgpack(input_msgpack_file);

    json::binary_t params = data["snapshot"]["params_binary"];
    
    int size_hashnet = sig_mlp->getNumParams(), size_rgbnet = col_mlp->getNumParams(),
        size_hashgrid = hash_enc->getNumParams();
    std::vector<float> sig_mlp_params(size_hashnet), color_mlp_params(size_rgbnet),
        hashgrid_params(size_hashgrid);
    int num_of_params = params.size();

    if (num_of_params / 2 != (size_hashgrid + size_hashnet + size_rgbnet)){
        std::cout << "Mismatched Snapshot and Config!" << std::endl;
        exit(1);
    }
    
    for(int i = 0; i < num_of_params; i += 2){
        uint32_t value = params[i] | (params[i + 1] << 8);
        int index = i / 2;
        float value_float = utils::from_int_to_float16(value);
        if(index < size_hashnet) {
            sig_mlp_params[index] = value_float;
        }
        else if(index < size_rgbnet + size_hashnet) {
            color_mlp_params[index - size_hashnet] = value_float;
        }
        else {
            hashgrid_params[index - size_hashnet - size_rgbnet] = value_float;
        }
    }
    hash_enc->loadParameters(hashgrid_params);
    sig_mlp->loadParameters(sig_mlp_params);
    col_mlp->loadParameters(color_mlp_params);

    json::binary_t density_grid_params = data["snapshot"]["density_grid_binary"];

    int num_of_params_ocgrid = density_grid_params.size();
    int size_ocgrid = occupancy_grid->getNumParams(), resolution = occupancy_grid->getResolution();

    std::vector<int> oc_params(size_ocgrid, 0);
    for(int i = 0; i < num_of_params_ocgrid; i += 2){
        uint32_t value = density_grid_params[i] | (density_grid_params[i + 1] << 8);
        float value_float = utils::from_int_to_float16(value);
        int index = utils::inv_morton(i / 2, resolution);
        if(value_float > 0.01) oc_params[index] = 1;
        else oc_params[index] = 0;
    }
    occupancy_grid->loadParameters(oc_params);
}

void Simulator::render() {
    initialize();

    while (true) {
        rayMarching();
        hashEncoding();
        shEncoding();
        sigmaMLP();
        colorMLP();
        volumeRendering();

        history.cycleCount++;
        rayCount = featurePool.rayMarchingID;
        if (rayCount >= MAX_RAY_COUNT) {
            break;
        }
        
        if (history.cycleCount % 10000 == 0) {
            printf("Cycle Count: %d\n", history.cycleCount);
            printf("Ray Count: %d\n", rayCount);
        }
    }

    // Write history data to image
    std::shared_ptr<Image> img = camera->getImage();
    for (int i = 0; i < img->getResolution().x(); i++) {
        for (int j = 0; j < img->getResolution().y(); j++) {
            Vec3f color = history.rgbs[i * img->getResolution().y() + j];
            img->setPixel(i, img->getResolution().y() - 1 - j, color);
        }
    }
    img->writeImgToFile("output.png");

    std::shared_ptr<Image> img_depth = std::make_shared<Image>(img->getResolution().x(), img->getResolution().y());
    for (int i = 0; i < img->getResolution().x(); i++) {
        for (int j = 0; j < img->getResolution().y(); j++) {
            float depth = featurePool.ts[i * img->getResolution().y() + j];
            if (depth >= RAY_DEFAULT_MAX) depth = 0.0;
            img_depth->setPixel(i, img->getResolution().y() - 1 - j, Vec3f(depth, depth, depth));
        }
    }
    img_depth->writeImgToFile("depth.png");
}

void Simulator::printHistory() {
    puts("========== Simulation History ==========");
    printf("Run Scene: %s\n", history.scene_name.c_str());
    printf("Simulation Frequency: %d MHz\n", history.frequency);
    printf("Cycle Count: %d\n", history.cycleCount);
    printf("Cycle Per Ray: %.6f\n", static_cast<float>(history.cycleCount) / static_cast<float>(rayCount));
    float cycle_period = 1.0 / (static_cast<float>(history.frequency) * 1e6);
    float total_time = cycle_period * history.cycleCount;
    printf("Simulation Time: %.6f s\n", total_time);
    float fps = 1.0 / total_time;
    printf("FPS: %.6f\n", fps);
    float equ_time_to_800_800 = total_time / MAX_RAY_COUNT * 800 * 800;
    float equ_fps_to_800_800 = 1.0 / equ_time_to_800_800;
    printf("Equivalent FPS to 800x800: %.6f\n", equ_fps_to_800_800);

    // Write history data to file
    std::string freq_str = std::to_string(history.frequency);
    std::string file_name = "History_" + freq_str + "MHz_" + history.scene_name + ".txt";
    std::ofstream fout(file_name);
    fout << "========== Simulation History ==========\n";
    fout << "Run Scene: " << history.scene_name << "\n";
    fout << "Simulation Frequency: " << history.frequency << " MHz\n";
    fout << "Cycle Count: " << history.cycleCount << "\n";
    fout << "Cycle Per Ray: " << static_cast<float>(history.cycleCount) / static_cast<float>(rayCount) << "\n";
    fout << "Simulation Time: " << total_time << " s\n";
    fout << "FPS: " << fps << "\n";
    fout << "Equivalent FPS to 800x800: " << equ_fps_to_800_800 << "\n";
    fout.close();
}

void Simulator::initialize() {
    waitCounter[RAYMARCHING] = latency[RAYMARCHING] - 1;

    // Ray Marching
    featurePool.rayMarchingID = 0;
    featurePool.ts = std::vector<float>(MAX_RAY_COUNT, RAY_DEFAULT_MIN);
    // Hash Encoding
    featurePool.HashRayID = -1;
    // SH Encoding
    featurePool.SHRayID = -1;
    // Sigma MLP
    featurePool.SigmaRayID = -1;
    // Color MLP
    featurePool.ColorRayID = -1;
    // Volume Rendering
    featurePool.VolumeRayID = -1;
    featurePool.colors = std::vector<Vec3f>(MAX_RAY_COUNT, Vec3f::Zero());
    featurePool.opacities = std::vector<float>(MAX_RAY_COUNT, 0.0);
}

void Simulator::rayMarching() {
    if (waitCounter[RAYMARCHING] > 0) {
        waitCounter[RAYMARCHING]--;
        return;
    }
    else {
        if (!rmFifo.isEmpty()) {
            RM_Reg rm = rmFifo.read();
            if (featurePool.rayMarchingID == rm.rayID) {
                // Terminate this ray. Jump to next ray at next cycle. reset t
                featurePool.rayMarchingID++;
                return;
            }
            else if (featurePool.rayMarchingID < rm.rayID) {
                puts("Error: Ray ID Mismatch");
                exit(1);
            }
            else {
                // Still Run this ray
            }
        }


        if (!hash_in_Fifo.isFull() && !sh_in_Fifo.isFull()) {

            // Do Ray Marching
            int ray_id = featurePool.rayMarchingID;
            Vec2i resolution = camera->getResolution();
            float ray_id_x = ray_id / resolution.y(), ray_id_y = ray_id % resolution.y();
            Ray ray = camera->generateRay(ray_id_x, ray_id_y);
            
            float t = featurePool.ts[ray_id];
            while (!occupancy_grid->isOccupy(ray(t)) && t < RAY_DEFAULT_MAX + EPS) {
                t += NGP_STEP_SIZE;
            }

            // If t > RAY_DEFAULT_MAX, then skip this ray
            if (t >= RAY_DEFAULT_MAX) {
                // Write data to history
                history.rgbs[featurePool.rayMarchingID] = featurePool.colors[featurePool.rayMarchingID];
                history.opacities[featurePool.rayMarchingID] = featurePool.opacities[featurePool.rayMarchingID];
                featurePool.rayMarchingID++;
                t = RAY_DEFAULT_MIN;
                return;
            }

            Vec3f pos = ray(t), dir = ray.getDirection();

            Hash_in_Reg hash;
            SH_in_Reg sh;
            hash.rayID = featurePool.rayMarchingID;
            hash.input = pos;
            sh.rayID = featurePool.rayMarchingID;
            sh.input = (dir + Vec3f(1, 1, 1)) / 2;
            
            hash_in_Fifo.write(hash);
            sh_in_Fifo.write(sh);
            
            featurePool.ts[ray_id] = t + NGP_STEP_SIZE;
            waitCounter[RAYMARCHING] = latency[RAYMARCHING] - 1;
        } 
    }
}

void Simulator::hashEncoding() {
    if (waitCounter[HASHENCODING] > 0) {
        waitCounter[HASHENCODING]--;
        return;
    }
    else {
        if (module_state[HASHENCODING] == DONE_AN_EXECUTION) {
            if (!sigmlp_in_Fifo.isFull() && !hash_out_Fifo.isEmpty()) {
                Hash_out_Reg hash = hash_out_Fifo.read();
                
                sigmlp_in_Fifo.write(SigMLP_in_Reg{hash.rayID, hash.output});

                module_state[HASHENCODING] = WAIT_FOR_INPUT;
            }
            // ELSE: WAIT FOR WRITING
        }
        if (module_state[HASHENCODING] == WAIT_FOR_INPUT) {
            if (!hash_in_Fifo.isEmpty() && !hash_out_Fifo.isFull()) {
                Hash_in_Reg hash = hash_in_Fifo.read();

                Vec3f input_point = hash.input;
                Vec32f output = hash_enc->encode(input_point);

                // Read Hash Input
                hash_out_Fifo.write(Hash_out_Reg{hash.rayID, output});
                
                waitCounter[HASHENCODING] = latency[HASHENCODING] - 1;
                module_state[HASHENCODING] = DONE_AN_EXECUTION;
            }
        }
    }
}

void Simulator::shEncoding() {
    if (waitCounter[SHENCODING] > 0) {
        waitCounter[SHENCODING]--;
        return;
    }
    else {
        if (module_state[SHENCODING] == DONE_AN_EXECUTION) {
            if (!colmlpFifo_SH.isFull() && !sh_out_Fifo.isEmpty()) {
                SH_out_Reg sh = sh_out_Fifo.read();
                colmlpFifo_SH.write(Col_MLP_From_SH{sh.rayID, sh.output});

                module_state[SHENCODING] = WAIT_FOR_INPUT;
            }
        }
        if (module_state[SHENCODING] == WAIT_FOR_INPUT) {
            if (!sh_in_Fifo.isEmpty() && !sh_out_Fifo.isFull()) {
                SH_in_Reg sh = sh_in_Fifo.read();
                

                //Vec3f input_dir = featurePool.SHInput;
                Vec3f input_dir = sh.input;
                Vec16f output = sh_enc->encode(input_dir);
                
                sh_out_Fifo.write(SH_out_Reg{sh.rayID, output});

                waitCounter[SHENCODING] = latency[SHENCODING] - 1;
                module_state[SHENCODING] = DONE_AN_EXECUTION;
            }
        }

    }
}

void Simulator::sigmaMLP() {
    if (waitCounter[SIGMAMLP] > 0) {
        waitCounter[SIGMAMLP]--;
        return;
    }
    else {
        if (module_state[SIGMAMLP] == DONE_AN_EXECUTION) {
            if (!colmlpFifo_Hash.isFull() && !sigmlp_out_Fifo.isEmpty()) {
                SigMLP_out_Reg color = sigmlp_out_Fifo.read();
                colmlpFifo_Hash.write(Col_MLP_From_Hash{color.rayID, color.output});

                module_state[SIGMAMLP] = WAIT_FOR_INPUT;
            }
        }
        if (module_state[SIGMAMLP] == WAIT_FOR_INPUT) {
            if (!sigmlp_in_Fifo.isEmpty() && !sigmlp_out_Fifo.isFull()) {
                SigMLP_in_Reg sigmlp = sigmlp_in_Fifo.read();

                //Vec32f input = featurePool.HashOutput;
                Vec32f input = sigmlp.input;
                Vec16f output = sig_mlp->inference(input);
                
                sigmlp_out_Fifo.write(SigMLP_out_Reg{sigmlp.rayID, output});
                
                waitCounter[SIGMAMLP] = latency[SIGMAMLP] - 1;
                module_state[SIGMAMLP] = DONE_AN_EXECUTION;
            }
        }
    }
}

void Simulator::colorMLP() {
    if (waitCounter[COLORMLP] > 0) {
        waitCounter[COLORMLP]--;
        return;
    }
    else {
        if (module_state[COLORMLP] == DONE_AN_EXECUTION) {
            if (!vr_in_Fifo.isFull() && !colmlp_out_Fifo.isEmpty()) {
                Col_MLP_out_Reg feature = colmlp_out_Fifo.read();
                
                vr_in_Fifo.write(VR_in_Reg{feature.rayID, feature.output});

                module_state[COLORMLP] = WAIT_FOR_INPUT;
            }
        }
        if (module_state[COLORMLP] == WAIT_FOR_INPUT) {
            if (!colmlpFifo_Hash.isEmpty() && !colmlpFifo_SH.isEmpty() ) {
                Col_MLP_From_Hash color1 = colmlpFifo_Hash.read();
                Col_MLP_From_SH color2 = colmlpFifo_SH.read();

                // Read Color Input
                int color1RayID = color1.rayID, color2RayID = color2.rayID;
                if (color1RayID != color2RayID) {
                    puts("Error: Ray ID Mismatch");
                    exit(1);
                }

                Vec16f input1 = color1.input, input2 = color2.input;
                Vec32f input = Vec32f::Zero();
                for (int i = 0; i < 16; i++) {
                    input[i] = input1[i];
                    input[i + 16] = input2[i];
                }
                Vec3f output = col_mlp->inference(input);
                float alpha = input1[0];

                colmlp_out_Fifo.write(Col_MLP_out_Reg{color1RayID, Vec4f(output[0], output[1], output[2], alpha)});

                waitCounter[COLORMLP] = latency[COLORMLP] - 1;
                module_state[COLORMLP] = DONE_AN_EXECUTION;
            }
        }
    }
}

void Simulator::volumeRendering() {
    if (waitCounter[VOLUMERENDERING] > 0) {
        waitCounter[VOLUMERENDERING]--;
        return;
    }
    else {
        if (module_state[VOLUMERENDERING] == DONE_AN_EXECUTION) {
            if (!vr_out_Fifo.isEmpty()) {
                VR_out_Reg vr = vr_out_Fifo.read();
                int rayID = vr.rayID;
                if (featurePool.opacities[rayID] >= 0.99) {
                    // Write data to history
                    history.rgbs[rayID] = featurePool.colors[rayID];
                    history.opacities[rayID] = featurePool.opacities[rayID];
                    // Write data to rayMarching
                    rmFifo.write(RM_Reg{rayID});
                    module_state[VOLUMERENDERING] = WAIT_FOR_INPUT;
                    return;
                }
            }
            
            module_state[VOLUMERENDERING] = WAIT_FOR_INPUT;
        }
        if (module_state[VOLUMERENDERING] == WAIT_FOR_INPUT) {
            if (!vr_in_Fifo.isEmpty()) {
                VR_in_Reg vr = vr_in_Fifo.read();
                int rayID = vr.rayID;
                float opacity = featurePool.opacities[rayID]; // TODO: FIND WHY THIS MAKE SENSE

                Vec4f rgba_raw = vr.input;
                float T = 1 - opacity;
                float alpha = 1 - expf(-expf(rgba_raw[3]) * NGP_STEP_SIZE);
                float weight = alpha * T;
                Vec3f color = utils::sigmoid(rgba_raw.head(3));


                featurePool.opacities[rayID] += weight;
                featurePool.colors[rayID] += weight * color;

                vr_out_Fifo.write(VR_out_Reg{rayID});
                
                waitCounter[VOLUMERENDERING] = latency[VOLUMERENDERING] - 1;
                module_state[VOLUMERENDERING] = DONE_AN_EXECUTION;
            }
        }
    }
}