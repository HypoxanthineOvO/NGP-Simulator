#include "qhash.hpp"
#include "qsh.hpp"
#include "qmlp.hpp"
#include "FixedPoint.hpp"

std::string to_string(float x) {
    return std::to_string(x);
}

int main() {
    std::shared_ptr<HashEncoding> hash_encoding = std::make_shared<HashEncoding>(2, 16, 19, 16);
    std::shared_ptr<SHEncoding> sh_encoding = std::make_shared<SHEncoding>(4, 3);
    std::shared_ptr<MLP> mlp = std::make_shared<MLP>(3, 4, 1, 4);
    printf("PARAMS: %d\n", mlp->getNumParams());


    std::vector<QuantNGP::MLPParams> mlp_params = {
        3.1, 5.2, 4.3, 10.4, 
        77.5, 5.6, 2.7, 0.8, 
        1.9, 13.0, 1.1, 3.2,
        1.3, 1.4, 2.5, 6.6,
        3.1, 5.2, 4.3, 1.4, 
        77.5, 5.6, 2.7, 0.8, 
        1.9, 13.0, 1.1, 3.2
    };
    mlp->loadParameters(mlp_params);

    Vector<QuantNGP::RMData> input_vec(3);
    input_vec(0) = 0.1;
    input_vec(1) = 0.2;
    input_vec(2) = 0.3;

    for (int i = 0; i < 3; i++) {
        printf("input_vec(%d): %s\n", i, to_string(input_vec(i)).c_str());
    }

    Eigen::Matrix<QuantNGP::HashData, 32, 1> hash_result = hash_encoding->encode(input_vec);
    for (int i = 0; i < 32; i++) {
        printf("hash result(%d): %s\n", i, to_string(hash_result(i)).c_str());
    }

    Eigen::Matrix<QuantNGP::SHData, 16, 1> result = sh_encoding->encode(input_vec);

    for (int i = 0; i < 16; i++) {
        printf("sh result(%d): %s\n", i, to_string(result(i)).c_str());
    }

    Eigen::Matrix<QuantNGP::MLPData, 4, 1> mlp_result = mlp->inference(result);
    for (int i = 0; i < 4; i++) {
        printf("mlp result(%d): %s\n", i, to_string(mlp_result(i)).c_str());
    }
}