from toolkit.extension import Extension


class MossImageSIFRExtension(Extension):
    uid = "moss_image_sifr"
    name = "MOSS Image SIFR"

    @classmethod
    def get_process(cls):
        from .trainer import MossImageSIFRTrainer

        return MossImageSIFRTrainer


AI_TOOLKIT_EXTENSIONS = [MossImageSIFRExtension]
