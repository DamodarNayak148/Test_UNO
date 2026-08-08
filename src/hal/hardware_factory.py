import yaml
from typing import Dict, Any
from src.hal.base_camera import BaseCamera
from src.hal.base_audio import BaseAudio
from src.hal.base_led import BaseLED
from src.hal.base_pantilt import BasePanTilt
from src.hal.base_buttons import BaseButtons

from src.hal.drivers_sim.sim_camera import SimCamera
from src.hal.drivers_sim.sim_audio import SimAudio
from src.hal.drivers_sim.sim_led import SimLED
from src.hal.drivers_sim.sim_pantilt import SimPanTilt
from src.hal.drivers_sim.sim_buttons import SimButtons

class HardwareSuite:
    """Container suite bundling camera, audio, led, pan_tilt, and buttons drivers."""

    def __init__(self, camera: BaseCamera, audio: BaseAudio, led: BaseLED, pan_tilt: BasePanTilt, buttons: BaseButtons):
        self.camera = camera
        self.audio = audio
        self.led = led
        self.pan_tilt = pan_tilt
        self.buttons = buttons

class HardwareFactory:
    """Factory creating and configuring the Hardware Suite based on application config."""

    @staticmethod
    def create_hardware_suite(config: Dict[str, Any]) -> HardwareSuite:
        mode = config.get("driver_mode", "simulated").lower()

        if mode == "simulated":
            hw_conf = config.get("hardware", {})
            cam_conf = hw_conf.get("camera", {})
            led_conf = hw_conf.get("leds", {})
            pt_conf = hw_conf.get("pan_tilt", {})

            camera = SimCamera(
                device_index=cam_conf.get("device_index", 0),
                width=cam_conf.get("width", 640),
                height=cam_conf.get("height", 480),
                fps=cam_conf.get("fps", 30)
            )

            audio = SimAudio(
                voice_rate=config.get("ai", {}).get("tts_voice_rate", 170)
            )

            led = SimLED(
                count=led_conf.get("count", 12),
                brightness=led_conf.get("default_brightness", 0.8)
            )

            pan_tilt = SimPanTilt(
                pan_range=(pt_conf.get("pan_min", -90), pt_conf.get("pan_max", 90)),
                tilt_range=(pt_conf.get("tilt_min", -45), pt_conf.get("tilt_max", 45))
            )

            buttons = SimButtons()

            return HardwareSuite(camera, audio, led, pan_tilt, buttons)

        elif mode == "uno_q":
            raise NotImplementedError("Arduino UNO Q hardware drivers will be attached when board arrives.")
        else:
            raise ValueError(f"Unknown driver_mode '{mode}' in config.")
