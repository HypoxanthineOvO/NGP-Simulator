import os


scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
if __name__ == '__main__':
    freq = 200
    max_t = 8
    fps = []
    psnrs = []
    
    print(f"At {freq} MHz, with max_t = {max_t}")
    for scene in scenes:
        os.system(f"./main {scene} {freq} {max_t}")
    for scene in scenes:
        with open(f"./History_200MHz_{scene}.txt") as f:
            lines = f.readlines()
            fps_line = lines[6]
            psnr_line = lines[9]
            # FPS: xxxx
            fps.append(float(fps_line.split(": ")[1].strip()))
            psnrs.append(float(psnr_line.split(": ")[1].strip()))
            print(f"Scene: {scene}, FPS: {fps[-1]}")
    print(f"Mean FPS: {round(sum(fps) / len(fps), 4)}")
    print(f"Mean PSNR: {round(sum(psnrs) / len(psnrs), 4)}")