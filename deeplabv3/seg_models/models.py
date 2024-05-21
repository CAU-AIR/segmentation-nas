import segmentation_models_pytorch as smp

def load_model(args, model_name):
    if model_name == 'DeepLabv3':
        ENCODER = 'resnet101'
        ENCODER_WEIGHTS = 'imagenet'
        ACTIVATION = 'sigmoid'

        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            # encoder_weights=ENCODER_WEIGHTS,
            encoder_weights=None,
            activation=ACTIVATION,
        )

    return model