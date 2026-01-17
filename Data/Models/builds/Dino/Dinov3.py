import torch
from convnext import ConvNeXt

convnext_sizes = {
    "Tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        checkpoint = "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    ),
    "Small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        checkpoint = "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"
    )
}


if __name__ == "__main__":
    for size, args in convnext_sizes.items():
        state_init = torch.load(args["checkpoint"])
        model = ConvNeXt(depths=args["depths"][:layers+1], dims=args["dims"][:layers+1])
        state = {}
        for key in model.state_dict().keys():
            state[key] = state_init[key]
        model.load_state_dict(state)
        example = torch.zeros((1,3,512,512)) 
        traced_script_module = torch.jit.trace(model, (example, torch.tensor([4]), torch.tensor([]), torch.tensor([])))
        traced_script_module.save(f"./DinoV3_{size}.pt")