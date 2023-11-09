from .nodes import enhance, core, color

NODE_CLASS_MAPPINGS = {
    **core.NODE_CLASS_MAPPINGS,
    **color.NODE_CLASS_MAPPINGS,
    **enhance.NODE_CLASS_MAPPINGS
}