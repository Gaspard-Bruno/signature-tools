try:
    from .src.signature.comfy_nodes import enhance, color, filters, io, transfrom, models, morphology, misc, processor
except:
    print(f"Error importing modules")

    import subprocess
    try:
        subprocess.check_call(['pip3', 'install', '-e', '.'])
    except subprocess.CalledProcessError:
        print("Installation failed. Please install the dependencies manually.")

    # Retry the import after attempting installation
    from .src.signature.comfy_nodes import enhance, color, filters, io, transfrom, models, morphology, misc, processor

NODE_CLASS_MAPPINGS = {
    **processor.NODE_CLASS_MAPPINGS,
    **models.NODE_CLASS_MAPPINGS,
    **io.NODE_CLASS_MAPPINGS,
    **transfrom.NODE_CLASS_MAPPINGS,
    **color.NODE_CLASS_MAPPINGS,
    **enhance.NODE_CLASS_MAPPINGS,
    **filters.NODE_CLASS_MAPPINGS,
    **morphology.NODE_CLASS_MAPPINGS,
    **misc.NODE_CLASS_MAPPINGS,
}

__all__ = ['NODE_CLASS_MAPPINGS']

MANIFEST = {
    "name": "Signature Tools",
    "version": (1,0,0),
    "author": "marcojoao",
    "project": "https://github.com/Gaspard-Bruno/signature-tools",
    "description": "Image processing tools, AI models and more",
}