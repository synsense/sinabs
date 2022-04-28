from pbr.version import VersionInfo

__version__ = VersionInfo("sinabs-dynapcnn").release_string()


from .dynapcnn_network import DynapcnnNetwork, DynapcnnCompatibleNetwork # second one for compatibility purposes
