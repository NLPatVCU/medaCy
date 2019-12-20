from medacy.tools.get_metamap import get_metamap_path


# See if MetaMap has been set for this installation
metamap_path = get_metamap_path()
have_metamap = metamap_path != 0

# Specify why MetaMap tests may be skipped
reason = "This test can only be performed if the path to MetaMap has been configured for this installation"
