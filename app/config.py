class Settings:
    def __init__(self):
        # Configuration properties with static defaults (fully overridden by CLI flags)
        self.log_level = "info"
        self.target_lang = "ENG"
        self.source_lang = "JPN"

        self.detection_size = 2048
        self.text_threshold = 0.5
        self.box_threshold = 0.7
        self.unclip_ratio = 2.3

        self.det_invert = False
        self.det_gamma_correct = False
        self.det_rotate = False
        self.det_auto_rotate = False

        self.verbose = False

        # Server configuration
        self.server_host = "127.0.0.1"
        self.server_port = 7861
        self.server_blocking_init = True


settings = Settings()
