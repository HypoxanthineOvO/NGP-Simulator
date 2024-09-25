#ifndef SHENCODING_HPP_
#define SHENCODING_HPP_

#include <vector>

#include "utils.hpp"
#include "FixedPoint.hpp"

class SHEncoding{
public:
    using DATA = FixedPoint<10, 10>;
    using Input = Eigen::Matrix<DATA, 3, 1>;
    using Output = Eigen::Matrix<DATA, 16, 1>;
    SHEncoding() = delete;
    SHEncoding(const nlohmann::json& configs):
    SHEncoding(
        utils::get_int_from_json(configs, "degree"),
        utils::get_int_from_json(configs, "n_dims_to_encode")
    ){}
    SHEncoding(int degree, int n_dims_to_encode):
        degree(degree), n_dims_to_encode(n_dims_to_encode){};
    Output encode(Input dir);

    // Get Output Dimension
    int getOutDim() const{
        return degree * degree;
    }
private:
    int degree;
    int n_dims_to_encode;
};

#endif // SHENCODING_HPP_

