import logging
import warnings
import os

def setup_logging(level=logging.INFO):
    """
    Configura logging global.
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

    logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger("app")
    logger.info("Logging configurado correctamente")
    return logger