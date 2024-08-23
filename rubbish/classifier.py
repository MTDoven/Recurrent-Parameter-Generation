



class ClassifierMambaDiffusion(MambaDiffusion):
    config = {}

    def __init__(self, sequence_length):
        super().__init__(sequence_length)
        self.condition_extractor, output_dim = ResNet18()
        assert self.config["d_condition"] == output_dim
        self.register_buffer("device_sign_buffer", torch.zeros(1))

    def forward(self, output_shape=None, x_0=None, condition=None, **kwargs):
        condition = self.condition_extractor(condition.to(self.device_sign_buffer.device))
        return super().forward(output_shape=output_shape, x_0=x_0, condition=condition, **kwargs)