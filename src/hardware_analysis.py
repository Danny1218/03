import platform, torch

def analyze_hardware():
    print('OS:', platform.system(), platform.release())
    cuda = torch.cuda.is_available()
    print('CUDA available:', cuda)
    if cuda:
        print('GPU:', torch.cuda.get_device_name(0))
        print('VRAM (GB):', torch.cuda.get_device_properties(0).total_memory/(1024**3))

if __name__ == '__main__':
    analyze_hardware() 