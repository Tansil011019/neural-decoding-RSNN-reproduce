from efficient_rsnn_bmi.base.readout import CustomReadoutGroup

def get_custom_readouts(cfg):
    ro_list = []
    for ro, specs in cfg.model["readouts"].items():
        if "tau_mem" in specs:
            tau_mem = specs["tau_mem"]
        else:
            tau_mem = cfg.model.tau_mem_readout
        if "tau_syn" in specs:
            tau_syn = specs["tau_syn"]
        else:
            tau_syn = cfg.model.tau_syn_readout

        if specs["type"] == "default":
            ro_group = CustomReadoutGroup(
                cfg.data.nb_outputs,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                het_timescales=cfg.model.het_timescales_readout,
                learn_timescales=cfg.model.learn_timescales_readout,
                initial_state=-1e-2,
                is_delta_syn=False,
            )
        elif specs["type"] == "delta":
            ro_group = CustomReadoutGroup(
                cfg.data.nb_outputs,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                het_timescales=cfg.model.het_timescales_readout,
                learn_timescales=cfg.model.learn_timescales_readout,
                initial_state=-1e-2,
                is_delta_syn=True,
            )

        ro_group.set_name(ro)
        ro_list.append(ro_group)

    return ro_list