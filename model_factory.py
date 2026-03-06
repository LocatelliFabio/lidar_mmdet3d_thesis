from mmdet3d.apis import init_model

def get_model():
    config = 'mmdet3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py'
    checkpoint = 'mmdet3d/checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth'
    model = init_model(config, checkpoint, device='cuda:0')

    return model