from .nodes import enhance, color, filters, io

NODE_CLASS_MAPPINGS = {
    **io.NODE_CLASS_MAPPINGS,
    **color.NODE_CLASS_MAPPINGS,
    **enhance.NODE_CLASS_MAPPINGS,
    **filters.NODE_CLASS_MAPPINGS,
}