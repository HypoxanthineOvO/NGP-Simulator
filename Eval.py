from typing import List
from math import ceil, floor
def estimateMLPCyc(mlp_size: List[int], mac_size: int = 64, show = True) -> int:
    """
    Estimate the number of cycles required for a given MLP size
    """
    size = len(mlp_size)
    tot_cyc = 0
    for i in range(size - 1):
        layer_bef = mlp_size[i]
        layer_aft = mlp_size[i + 1]
        # e.g.: bef, aft = 32, 64, mac = 32 -> 2;
        # e.g.: bef, aft = 33, 64, mac = 32 -> 3;
        
        cyc = ceil(layer_bef * layer_aft / (mac_size * mac_size))
        if (show):
            print(f"Layer {i}: {layer_bef} -> {layer_aft}, {cyc} cycles")
        tot_cyc += cyc
    if (show):
        print(f"Total cycles: {tot_cyc}")
    return tot_cyc

def getMACAreaPerf(res_0: int, res_1: int, freq: int):
    ref_res = 256
    ref_freq = 400
    ref_area = 29.69
    ref_power = 0.7924
    
    area = (res_0 * res_1) / (ref_res ** 2) * ref_area
    power = (res_0 * res_1) / (ref_res ** 2) * (freq / ref_freq) * ref_power
    
    return area, power

if __name__ == "__main__":
    NetworkSize = [
        [32, 64], [64, 16],
        [32, 64], [64, 64], [64, 3]
    ]
    
    # Use 32 * 32 MAC
    cyc = estimateMLPCyc([32, 64, 16], 32, show = False) + estimateMLPCyc([32, 64, 64, 3], 32, show = False)
    print("Using 32x32 MAC")
    print(f"Total cycles: {cyc}")
    a, p = getMACAreaPerf(32, 32, 200)
    print(f"Area: {round(a, 2)} mm^2, Power: {round(p * 1000, 2)} mW")
    
    # Use 64 * 64 MAC
    cyc = estimateMLPCyc([32, 64, 16], 64, show = False) + estimateMLPCyc([32, 64, 64, 3], 64, show = False)
    print("Using 64x64 MAC")
    print(f"Total cycles: {cyc}")
    a, p = getMACAreaPerf(64, 64, 200)
    print(f"Area: {round(a, 2)} mm^2, Power: {round(p * 1000, 2)} mW")
    
    # In 1 Cycle:
    ## 32 * 64, 64 * 16, 32 * 64, 64 * 64, 64 * 3
    print("In 1 Cycle:")
    a_1, p_1 = getMACAreaPerf(32, 64, 200)
    a_2, p_2 = getMACAreaPerf(64, 16, 200)
    a_3, p_3 = getMACAreaPerf(32, 64, 200)
    a_4, p_4 = getMACAreaPerf(64, 64, 200)
    a_5, p_5 = getMACAreaPerf(64, 3, 200)
    a = a_1 + a_2 + a_3 + a_4 + a_5
    p = p_1 + p_2 + p_3 + p_4 + p_5
    print(f"Area: {round(a, 2)} mm^2, Power: {round(p * 1000, 2)} mW")