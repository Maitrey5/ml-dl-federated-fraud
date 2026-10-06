import torch


def get_parameters(model):
    return [
        parameter.detach().cpu().numpy()
        for parameter in model.parameters()
    ]


def set_parameters(model, parameters):
    for model_parameter, new_parameter in zip(
        model.parameters(),
        parameters
    ):
        model_parameter.data = torch.tensor(
            new_parameter,
            dtype=model_parameter.dtype
        )