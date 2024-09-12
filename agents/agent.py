from agents.image_modules import make_image_module
from agents.agent_base import *
from torchvision.transforms import Normalize


# BEV AGENTS

class RGBBEVAgent(BaseAgent):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 3 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder = make_image_module(config, in_channels=in_channels)
        self.modality = 'rgb_birds_eye_view'
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


class RGBBEVDoubleAgent(BaseDoubleAgent):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 3 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder.update(
            dict(actor=make_image_module(config, in_channels=in_channels),
                 critic=make_image_module(config, in_channels=in_channels)))
        self.modality = 'rgb_birds_eye_view'
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


class RGBBEVAgentLSTM(BaseAgentLSTM):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 3 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder = make_image_module(config, in_channels=in_channels)
        self.modality = 'rgb_birds_eye_view'
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


class RGBBEVDoubleAgentLSTM(BaseDoubleAgentLSTM):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 3 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder.update(
            dict(actor=make_image_module(config, in_channels=in_channels),
                 critic=make_image_module(config, in_channels=in_channels)))
        self.modality = 'rgb_birds_eye_view'
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


# BEV GRAY AGENTS

class GrayBEVAgent(BaseAgent):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 1 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder = make_image_module(config, in_channels=in_channels)
        self.modality = 'rgb_birds_eye_view'  # env makes grayscale image, sensor is rgb
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


class GrayBEVDoubleAgent(BaseDoubleAgent):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 1 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder.update(
            dict(actor=make_image_module(config, in_channels=in_channels),
                 critic=make_image_module(config, in_channels=in_channels)))
        self.modality = 'rgb_birds_eye_view'  # env makes grayscale image, sensor is rgb
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


class GrayBEVAgentLSTM(BaseAgentLSTM):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        config.rl.image.conv_arch[0][0] = 1 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        self.modality_encoder = make_image_module(config, in_channels=1)
        self.modality = 'rgb_birds_eye_view'  # env makes grayscale image, sensor is rgb
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


class GrayBEVDoubleAgentLSTM(BaseDoubleAgentLSTM):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 1 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder.update(
            dict(actor=make_image_module(config, in_channels=in_channels),
                 critic=make_image_module(config, in_channels=in_channels)))
        self.modality = 'rgb_birds_eye_view'  # env makes grayscale image, sensor is rgb
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


# RGB AGENTS

class RGBAgent(BaseAgent):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        self.modality_encoder = make_image_module(config)
        self.modality = 'camera'
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


class RGBDoubleAgent(BaseDoubleAgent):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        self.modality_encoder.update(dict(actor=make_image_module(config), critic=make_image_module(config)))
        self.modality = 'camera'
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


class RGBAgentLSTM(BaseAgentLSTM):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        self.modality_encoder = make_image_module(config)
        self.modality = 'camera'
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


class RGBDoubleAgentLSTM(BaseDoubleAgentLSTM):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        self.modality_encoder.update(dict(actor=make_image_module(config), critic=make_image_module(config)))
        self.modality = 'camera'
        self.modality_transforms.add_module('0_1', Normalize(0, 255))


# Multi Layer BEV
class MultiBEVAgent(BaseAgent):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 6 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder = make_image_module(config, in_channels=in_channels)
        self.modality = 'multi_birds_eye_view'
        # Note: No normalization for multi-layer BEV since sensor is already normalized


class MultiBEVDoubleAgent(BaseDoubleAgent):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 6 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder.update(
            dict(actor=make_image_module(config, in_channels=in_channels),
                 critic=make_image_module(config, in_channels=in_channels)))
        self.modality = 'multi_birds_eye_view'


class MultiBEVAgentLSTM(BaseAgentLSTM):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 6 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder = make_image_module(config, in_channels=in_channels)
        self.modality = 'multi_birds_eye_view'


class MultiBEVDoubleAgentLSTM(BaseDoubleAgentLSTM):
    def __init__(self, config, envs):
        super().__init__(config, envs)
        in_channels = 6 * (
            (config.rl.frame_stack.n_frames - 1) // config.rl.frame_stack.skip_frames + 1
            if config.rl.frame_stack.use is True else 1)
        config.rl.image.conv_arch[0][0] = in_channels
        self.modality_encoder.update(
            dict(actor=make_image_module(config, in_channels=in_channels),
                 critic=make_image_module(config, in_channels=in_channels)))
        self.modality = 'multi_birds_eye_view'
