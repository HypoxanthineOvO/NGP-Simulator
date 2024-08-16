#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>
#include <memory> // It's for Ubuntu and other Linux OS using GCC
#include <iostream>

#include "nlohmann/json.hpp"

using Vec2f = Eigen::Vector2f;
using Vec2i = Eigen::Vector2i;

using Vec3f = Eigen::Vector3f;
using Vec3i = Eigen::Vector3i;

using Vec4f = Eigen::Vector4f;
using Vec4i = Eigen::Vector4i;

using Mat3f = Eigen::Matrix3f;
using Mat3i = Eigen::Matrix3i;
using Mat4f = Eigen::Matrix4f;
using Mat4i = Eigen::Matrix4i;

using Vec16f = Eigen::Matrix<float, 16, 1>;
using Vec32f = Eigen::Matrix<float, 32, 1>;

using VecXf = Eigen::VectorXf;
using MatXf = Eigen::MatrixXf;

// Basic Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float INV_PI = 0.31830988618379067154f;
constexpr float EPS = 1e-5f;
constexpr float RAY_DEFAULT_MIN = 0.1;
constexpr float RAY_DEFAULT_MAX = 3.0;

// Constants for Instant NGP
constexpr float SQRT_3 = 1.73205080756887729352f;
constexpr int NGP_STEPs = 1024;
constexpr float NGP_STEP_SIZE = SQRT_3 / NGP_STEPs;

inline Eigen::VectorXf stdvectorToEigenVector(std::vector<float>& stdv){
    return Eigen::Map<Eigen::VectorXf>(stdv.data(), stdv.size());
}

template <typename T>
class FIFO {
public:
    FIFO(): fifoSize(2) {}
    FIFO(int size): fifoSize(size) {}

    bool write(T data) {
        if (fifo.size() < fifoSize) {
            fifo.push_back(data);
            return true;
        }
        return false;
    }
    T read() {
        T data;
        if (fifo.size() > 0) {
            data = fifo[0];
            fifo.erase(fifo.begin());
            return data;
        }
        return data;
    }
    T peek() {
        T data;
        if (fifo.size() > 0) {
            data = fifo[0];
            return data;
        }
        return data;
    }
    bool isFull() {
        return fifo.size() == fifoSize;
    }
    bool isEmpty() {
        return fifo.size() == 0;
    }
    void printFIFO() {
        printf("Size: %d / %d\n", fifo.size(), fifoSize);
    }

private:
    int fifoSize;
    std::vector<T> fifo;
};


namespace utils {

	static inline float clamp01(float v) {
		if (v > 1) v = 1;
		else if (v < 0) v = 0;
		return v;
	}

	static inline uint8_t gammaCorrection(float radiance) {
		return static_cast<uint8_t>(255.f * clamp01(powf(radiance, 1.f / 2.2f)));
	}

	static inline uint8_t trans(float radiance){
		return static_cast<uint8_t>(255.f * clamp01(radiance));
	}

	static inline float radians(float x) { return x * PI / 180; }

	static inline Vec3f deNan(const Vec3f& vec, float val) {
		Vec3f tmp = vec;
		if (vec.x() != vec.x()) tmp.x() = val;
		if (vec.y() != vec.y()) tmp.y() = val;
		if (vec.z() != vec.z()) tmp.z() = val;
		return tmp;
	}
	static VecXf sigmoid(VecXf input){
		VecXf output(input.size());
		for(int i = 0; i < input.size(); i++){
			output(i) = 1.0f / (1.0f + std::exp(-input(i)));
		}
		return output;
	}

	static inline uint32_t as_uint(const float x){
		return *(uint32_t*)(&x);
	}

	static inline float as_float(const uint32_t x) {
		return *(float*)&x;
	}

	static float from_int_to_float16(const uint32_t& x){
		const uint32_t exponent = (x & 0x7C00) >> 10,
			mantissa = (x & 0x03FF) << 13,
			v = as_uint((float)(mantissa)) >> 23;
			return as_float(
				(x&0x8000)<<16 | 
				(exponent != 0)*((exponent + 112) << 23 | mantissa) |
				((exponent == 0)&(mantissa != 0)) * ((v - 37) << 23|((mantissa << (150-v)) & 0x007FE000)));
	}

	static inline int inv_Part_1_By_2(int x){
			x = ((x >> 2) | x) & 0x030C30C3;
			x = ((x >> 4) | x) & 0x0300F00F;
			x = ((x >> 8) | x) & 0x030000FF;
			x = ((x >>16) | x) & 0x000003FF;
			return x;
	}


	static int inv_morton(int input, int resolution){
		int x = inv_Part_1_By_2(input &        0x09249249);
		int y = inv_Part_1_By_2((input >> 1) & 0x09249249);
		int z = inv_Part_1_By_2((input >> 2) & 0x09249249);
		
		return x * resolution * resolution + y * resolution + z;
	}

	static Mat4f nerf_matrix_to_ngp(MatXf pose, float scale = 0.33, Vec3f offset = Vec3f(0.5, 0.5, 0.5)){
		Mat4f out_mat;
		out_mat << pose(1, 0) , -pose(1, 1) , -pose(1, 2) , pose(1, 3) * scale + offset(0) , \
			pose(2, 0) , -pose(2, 1) , -pose(2, 2) , pose(2, 3) * scale + offset(1) , \
			pose(0, 0) , -pose(0, 1) , -pose(0, 2) , pose(0, 3) * scale + offset(2) ,\
			0 , 0 , 0 , 1;
		return out_mat;
	}
	static int get_int_from_json(const nlohmann::json& config, std::string name){
		int value;
		return config.at(name).get_to(value);
	}

	static inline float min(float v1, float v2){
		if (v1 < v2) return v1;
		else return v2;
	}
	static inline float max(float v1, float v2){
		if (v1 > v2) return v1;
		else return v2;
	}
}

#endif // UTILS_HPP_