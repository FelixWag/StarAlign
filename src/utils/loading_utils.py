import torch
from pathlib import Path


def load_model(model, checkpoint_or_path, client_id_to_load, device, load_global, which_model='latest_states',
               strict=True):
    assert which_model in ['best_valid_loss', 'best_valid_accuracy', 'best_valid_mean_accuracy', 'latest_states'] \
           or 'best_valid_accuracy_client_' in which_model or 'best_valid_loss_client_' in which_model \
           or 'best_valid_accuracy_' in which_model
    if isinstance(checkpoint_or_path, Path):
        checkpoint_or_path = (checkpoint_or_path / 'model' / which_model).with_suffix('.pt')
        checkpoint_or_path = torch.load(checkpoint_or_path, map_location=device)
    if load_global:
        model.load_state_dict(checkpoint_or_path[f'model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint_or_path[f'model_state_client_{client_id_to_load}_dict'], strict=strict)
    model.to(device)


def load_all_models(global_model, client_models, checkpoint, client_ids_to_load, device, which_model='latest_states'):
    # First load the checkpoint on the right GPU
    ckpt = torch.load(((checkpoint / 'model' / which_model).with_suffix('.pt')), map_location=device)

    # Load the global model. We don't neet to load the global optimizer since it does not exist.
    load_model(model=global_model, checkpoint_or_path=ckpt, client_id_to_load=None, device=device,
               which_model=which_model, load_global=True)

    # Load the client models and optimizers
    for c_id, c_model in zip(client_ids_to_load, client_models, strict=True):
        load_model(model=c_model, checkpoint_or_path=ckpt, client_id_to_load=c_id,
                   device=device, which_model=which_model, load_global=False)
