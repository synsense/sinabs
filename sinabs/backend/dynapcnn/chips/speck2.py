from subprocess import CalledProcessError
try:
    import samna
    from samna.speck2.configuration import SpeckConfiguration
except (ImportError, ModuleNotFoundError, CalledProcessError):
    SAMNA_AVAILABLE = False
else:
    SAMNA_AVAILABLE = True

from .dynapcnn import DynapcnnConfigBuilder


# Since most of the configuration is identical to DYNAP-CNN, we can simply inherit this class

class Speck2ConfigBuilder(DynapcnnConfigBuilder):

    @classmethod
    def get_samna_module(cls):
        return samna.speck2

    @classmethod
    def get_default_config(cls) -> "SpeckConfiguration":
        return SpeckConfiguration()
    
    @classmethod
    def get_input_buffer(cls):
        return samna.BasicSourceNode_speck2_event_input_event()

    @classmethod
    def get_output_buffer(cls):
        return samna.BasicSinkNode_speck2_event_output_event()
