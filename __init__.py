from .nodes import enhance, color, filters, io, transfrom, models, morphology

NODE_CLASS_MAPPINGS = {
    **io.NODE_CLASS_MAPPINGS,
    **color.NODE_CLASS_MAPPINGS,
    **enhance.NODE_CLASS_MAPPINGS,
    **filters.NODE_CLASS_MAPPINGS,
    **transfrom.NODE_CLASS_MAPPINGS,
    **morphology.NODE_CLASS_MAPPINGS,
    **models.NODE_CLASS_MAPPINGS,
}