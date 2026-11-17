from solarian.services import Service


class UmiPolicyService(Service):
    """A UmiTrainer checkpoint, served by solarian-diffusion's ``DiffusionPolicyServer``.

    Its Hydra ``_target_`` is ``solarian_umi.diffusion_policy...``, so
    the image needs this package next to the server's own.
    """

    server_module = "solarian_diffusion.policy_server"
    default_port = 8001
    checkpoint_arg = "--input"

    @classmethod
    def modal_spec(cls):
        from solarian_diffusion.trainer import DiffusionTrainer

        from solarian_umi.trainer import UmiTrainer

        spec = UmiTrainer.modal_spec()
        spec.local_python_packages = list(dict.fromkeys(
            [*spec.local_python_packages, *DiffusionTrainer.modal_spec().local_python_packages]
        ))
        return spec
