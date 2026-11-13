class Settings:
    # Camera settings
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    FPS_LIMIT: int = 30

    # MediaPipe Hands settings
    MAX_NUM_HANDS: int = 2
    MIN_DETECTION_CONFIDENCE: float = 0.65
    MIN_TRACKING_CONFIDENCE: float = 0.65
    MODEL_COMPLEXITY: int = 0  # 0 = Lite (real-time low-latency CPU), 1 = Full

    # Stabilizer / Debounce settings
    DEBOUNCE_WINDOW_SIZE: int = 5
    TRANSITION_THRESHOLD: int = 4

    # Streaming settings
    STREAM_MAX_WIDTH: int = 800   # Streamlined 800px width yields ~1ms JPEG encoding & sub-20KB frames
    STREAM_JPEG_QUALITY: int = 65  # Visually lossless preview with ultra-fast encoding and zero buffer lag

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

settings = Settings()
