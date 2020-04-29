parameters_dict = {
    "buffer_size": int(1e5),
    "batch_size": 128,
    "gamma": 0.99,
    "tau": 1e-3,
    "lr_actor": 1e-4,
    "lr_critic": 1e-4,
    "weight_decay": 0,
    "n_updates": 10,
    "update_every_nsteps": 20,
    "nhidden_actor": [128, 64],
    "nhidden_critic": [128, 64],
}