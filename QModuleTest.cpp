#include "sh.hpp"
#include "mlp.hpp"
#include "FixedPoint.hpp"

std::string to_string(float x) {
    return std::to_string(x);
}

int main() {
    std::shared_ptr<SHEncoding> sh_encoding = std::make_shared<SHEncoding>(4, 3);
    std::shared_ptr<MLP> mlp = std::make_shared<MLP>(3, 16, 2, 16);

    using TestType = SHEncoding::DATA;

    Vector<TestType> input_vec(3);
    input_vec(0) = 0.1;
    input_vec(1) = 0.2;
    input_vec(2) = 0.3;

    for (int i = 0; i < 3; i++) {
        printf("input_vec(%d): %s\n", i, to_string(input_vec(i)).c_str());
    }

    Eigen::Matrix<SHEncoding::DATA, 16, 1> result = sh_encoding->encode(input_vec);

    // for (int i = 0; i < 16; i++) {
    //     printf("result(%d): %s\n", i, to_string(result(i)).c_str());
    // }

    Eigen::Matrix<MLP::DATA, 16, 1> mlp_result = mlp->inference(result);
    for (int i = 0; i < 16; i++) {
        printf("result(%d): %s\n", i, to_string(mlp_result(i)).c_str());
    }
}