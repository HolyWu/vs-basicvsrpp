import importlib.resources

with importlib.resources.path('vsbasicvsrpp', 'spynet.pth') as p:
    spynet_path = str(p)

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=128,
        num_blocks=25,
        is_low_res_input=False,
        spynet_pretrained=spynet_path))
